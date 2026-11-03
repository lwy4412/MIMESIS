from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


D = 128
K = 4
BATCH_SIZE = 1
SEED = 42
EPOCHS = 300
LEARNING_RATE = 0.0008
OUTPUT_DIR = Path(__file__).resolve().parent / "output_E"


class GraphItemNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.mean_scale = nn.Parameter(torch.ones(channels))

    def forward(self, x, batch, num_graphs):
        if not x.numel():
            return x
        mean = scatter(x, batch, dim=0, dim_size=num_graphs, reduce="mean")
        centered = x - mean[batch] * self.mean_scale

        var = scatter(
            centered.square(), batch, dim=0, dim_size=num_graphs, reduce="mean"
        )
        normalized = centered * torch.rsqrt(var[batch] + 1e-5)
        return normalized * self.weight + self.bias


class LinearGraphNorm(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            nn.Linear(in_features, out_features, bias=False),
            GraphItemNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x, batch, num_graphs):
        x = self[0](x)
        x = self[1](x, batch, num_graphs)
        return self[2](x)


class MPNN_EN(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr=None)
        self.out = LinearGraphNorm(2 * in_channels, in_channels)

    def forward(self, x_src, x_tgt, edge_index, batch_node, num_graphs):
        return self.propagate(
            edge_index, x=(x_src, x_tgt),
            size=(x_src.size(0), x_tgt.size(0)),
            batch_graph=batch_node, num_graphs=num_graphs,
        )

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x[1], aggr_out], dim=1)
        return self.out(tmp, batch_graph, num_graphs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNN_EE(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr=None)
        self.edge_mlp = LinearGraphNorm(4 * in_channels, in_channels)
        self.out = LinearGraphNorm(2 * in_channels, in_channels)

    def forward(self, x, edge_index, edge_attr, batch_element, num_graphs):
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr,
            batch_graph=batch_element,
            batch_message=batch_element[edge_index[1]], num_graphs=num_graphs,
        )

    def message(self, x_j, x_i, edge_attr, batch_message, num_graphs):
        msg = torch.cat([x_j, x_j - x_i, x_j * edge_attr, x_i * edge_attr], dim=1)
        return self.edge_mlp(msg, batch_message, num_graphs)

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.out(tmp, batch_graph, num_graphs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNN_NN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__(aggr=None)
        self.message_mlp = LinearGraphNorm(
            node_dim * 3 + edge_dim, hidden_dim
        )
        self.update_mlp = LinearGraphNorm(
            node_dim + hidden_dim, out_dim
        )

    def forward(self, x, edge_index, edge_attr, batch_node, num_graphs):
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, batch_graph=batch_node,
            batch_message=batch_node[edge_index[1]], num_graphs=num_graphs,
        )

    def message(self, x_j, x_i, edge_attr, batch_message, num_graphs):
        tmp = torch.cat([x_j, x_j - x_i, x_j * edge_attr, x_i * edge_attr], dim=1)
        return self.message_mlp(tmp, batch_message, num_graphs)

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x, aggr_out], dim=1)
        delta = self.update_mlp(tmp, batch_graph, num_graphs)
        return F.leaky_relu(delta + x, 0.2)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNNBlock(nn.Module):
    def __init__(
        self, node_dim, edge_dim, hidden_dim, out_dim,
        update_edge_attr=False):
        super().__init__()
        self.mpnn = MPNN_NN(
            node_dim, edge_dim, hidden_dim, out_dim)
        self.update_edge_attr = update_edge_attr
        if update_edge_attr:
            self.edge_updater = LinearGraphNorm(
                2 * out_dim + edge_dim, edge_dim
            )

    def forward(self, x, edge_index, edge_attr_ji, batch_node, num_graphs):
        if self.update_edge_attr:
            src, tgt = edge_index
            edge_ji_input_ = torch.cat([x[src], x[tgt], edge_attr_ji], dim=1)
            edge_delta = self.edge_updater(
                edge_ji_input_, batch_node[tgt], num_graphs
            )
            edge_attr_ji_new = F.leaky_relu(edge_delta + edge_attr_ji, 0.2)
            updated_x = self.mpnn(
                x, edge_index, edge_attr_ji_new, batch_node, num_graphs
            )
            return updated_x, edge_attr_ji_new
        updated_x = self.mpnn(
            x, edge_index, edge_attr_ji, batch_node, num_graphs
        )
        return updated_x, edge_attr_ji


class StackedMPNN(nn.Module):
    def __init__(
        self, num_layers, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        layers = []
        for layer in range(num_layers):
            layers.append(MPNNBlock(
                node_dim, edge_dim, hidden_dim, out_dim,
                update_edge_attr=(layer > 0)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr_ji, batch_node, num_graphs):
        for layer in self.layers:
            x, edge_attr_ji = layer(
                x, edge_index, edge_attr_ji, batch_node, num_graphs
            )
        return x


class Model(nn.Module):
    def __init__(self, num_hiddens, num_layers):
        super().__init__()
        self.ve_lin = nn.Sequential(
            nn.Linear(10, num_hiddens, bias=False),
            nn.SiLU(),
        )
        self.vn_lin = nn.Sequential(
            nn.Linear(8, num_hiddens, bias=False),
            nn.SiLU(),
        )
        self.eee_lin = nn.Sequential(
            nn.Linear(3, num_hiddens, bias=False),
            nn.SiLU(),
        )
        self.enn_lin = nn.Sequential(
            nn.Linear(4, num_hiddens, bias=False),
            nn.SiLU(),
        )
        self.ee_mpnn = MPNN_EE(num_hiddens)
        self.en_mpnn = MPNN_EN(num_hiddens)
        self.nn_mpnn = StackedMPNN(
            num_layers, num_hiddens, num_hiddens, num_hiddens, num_hiddens)
        self.out = nn.Sequential(
            nn.Linear(num_hiddens, 2 * num_hiddens),
            nn.SiLU(),
            nn.Linear(2 * num_hiddens, 2),
        )

    def forward(
        self, x_dict, edge_index_dict, edge_attr_dict, batch_dict, num_graphs,
    ):
        x_element = x_dict["element"]
        x_node = x_dict["node"]
        batch_element = batch_dict["element"]
        batch_node = batch_dict["node"]
        edge_index_en = edge_index_dict["element", "to", "node"]
        edge_index_ee = edge_index_dict["element", "to", "element"]
        edge_attr_ee = edge_attr_dict["element", "to", "element"]
        edge_index_nn = edge_index_dict["node", "to", "node"]
        edge_attr_nn = edge_attr_dict["node", "to", "node"]

        x_e = self.ve_lin(x_element)
        x_n = self.vn_lin(x_node)
        edge_attr_ee = self.eee_lin(edge_attr_ee)
        edge_attr_nn = self.enn_lin(edge_attr_nn)
        x_e_update = self.ee_mpnn(
            x_e, edge_index_ee, edge_attr_ee, batch_element, num_graphs
        )
        x_n_update = self.en_mpnn(
            x_e_update, x_n, edge_index_en, batch_node, num_graphs
        )
        x_n_mpnn = self.nn_mpnn(
            x_n_update, edge_index_nn, edge_attr_nn, batch_node, num_graphs
        )
        return self.out(x_n_mpnn)


def loss_function(prediction, graph):
    target = graph["node"].y
    return (prediction - target).abs().mean()


def train_epoch(model, train_iter, optimizer, device):
    model.train()
    total_loss = 0.0
    for graph in train_iter:
        graph = graph.to(device)
        prediction = model(
            graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict,
            graph.batch_dict, graph.num_graphs,
        )
        loss = loss_function(prediction, graph)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(train_iter)


def train():
    sample_path = Path(__file__).resolve().parent / "sample_E1_B1.pt"
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    graphs = [sample["graph"]]
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    model = Model(num_hiddens=D, num_layers=K).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0.01 * LEARNING_RATE,
    )
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_iter, optimizer, device)
        scheduler.step()
        print(f"Epoch {epoch}, DD loss = {train_loss:.6g}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": EPOCHS},
        OUTPUT_DIR / "model.pt",
    )


if __name__ == "__main__":
    train()
