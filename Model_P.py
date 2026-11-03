from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


D = 128
K = 4
L = 6
BATCH_SIZE = 1
SEED = 42
EPOCHS = 600
LEARNING_RATE = 0.0004
OUTPUT_DIR = Path(__file__).resolve().parent / "output_P"


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
        self.message_mlp = LinearGraphNorm(4 * in_channels, in_channels)
        self.out = LinearGraphNorm(2 * in_channels, in_channels)

    def forward(self, x, edge_index, edge_attr, batch_element, num_graphs):
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr,
            batch_graph=batch_element,
            batch_message=batch_element[edge_index[1]], num_graphs=num_graphs,
        )

    def message(self, x_j, x_i, edge_attr, batch_message, num_graphs):
        msg = torch.cat([x_j, x_j - x_i, x_j * edge_attr, x_i * edge_attr], dim=1)
        return self.message_mlp(msg, batch_message, num_graphs)

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.out(tmp, batch_graph, num_graphs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNN_NN(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr=None)
        self.message_mlp = LinearGraphNorm(
            hidden_dim * 4, hidden_dim
        )
        self.update_mlp = LinearGraphNorm(
            hidden_dim * 2, hidden_dim
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


class MPNN_NI(MessagePassing):
    def __init__(self, input_dim):
        super().__init__(aggr=None)
        self.message_mlp = LinearGraphNorm(2 * input_dim, input_dim)
        self.out = LinearGraphNorm(2 * input_dim, input_dim)

    def forward(self, x_src, x_tgt, edge_index, edge_attr, batch_integration, num_graphs):
        return self.propagate(
            edge_index, x=(x_src, x_tgt), edge_attr=edge_attr,
            size=(x_src.size(0), x_tgt.size(0)), batch_graph=batch_integration,
            batch_message=batch_integration[edge_index[1]], num_graphs=num_graphs,
        )

    def message(self, x_j, edge_attr, batch_message, num_graphs):
        tmp = torch.cat([x_j, x_j * edge_attr], dim=1)
        return self.message_mlp(tmp, batch_message, num_graphs)

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x[1], aggr_out], dim=1)
        return self.out(tmp, batch_graph, num_graphs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNN_II(MessagePassing):
    def __init__(self, input_dim):
        super().__init__(aggr=None)
        self.message_mlp = LinearGraphNorm(2 * input_dim, input_dim)
        self.out = LinearGraphNorm(2 * input_dim, input_dim)

    def forward(self, x, edge_index, edge_attr, batch_integration, num_graphs):
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, batch_graph=batch_integration,
            batch_message=batch_integration[edge_index[1]], num_graphs=num_graphs,
        )

    def message(self, x_j, edge_attr, batch_message, num_graphs):
        tmp = torch.cat([x_j, x_j * edge_attr], dim=1)
        return self.message_mlp(tmp, batch_message, num_graphs)

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.out(tmp, batch_graph, num_graphs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNN_IN(MessagePassing):
    def __init__(self, input_dim):
        super().__init__(aggr=None)
        self.message_mlp = LinearGraphNorm(2 * input_dim, input_dim)
        self.out = LinearGraphNorm(2 * input_dim, input_dim)

    def forward(self, x_src, x_tgt, edge_index, edge_attr, batch_node, num_graphs):
        return self.propagate(
            edge_index, x=(x_src, x_tgt), edge_attr=edge_attr,
            size=(x_src.size(0), x_tgt.size(0)), batch_graph=batch_node,
            batch_message=batch_node[edge_index[1]], num_graphs=num_graphs,
        )

    def message(self, x_j, edge_attr, batch_message, num_graphs):
        tmp = torch.cat([x_j, x_j * edge_attr], dim=1)
        return self.message_mlp(tmp, batch_message, num_graphs)

    def update(self, aggr_out, x, batch_graph, num_graphs):
        tmp = torch.cat([x[1], x[1] - aggr_out], dim=1)
        return self.out(tmp, batch_graph, num_graphs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")


class MPNN_IN_(MessagePassing):
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


class NNBlock(nn.Module):
    def __init__(self, hidden_dim, update_edge_attr=False):
        super().__init__()
        self.mpnn = MPNN_NN(hidden_dim)
        self.update_edge_attr = update_edge_attr
        if update_edge_attr:
            self.edge_updater = LinearGraphNorm(hidden_dim * 3, hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch_node, num_graphs):
        if self.update_edge_attr:
            src, tgt = edge_index
            edge_input_ = torch.cat([x[src], x[tgt], edge_attr], dim=1)
            delta = self.edge_updater(edge_input_, batch_node[tgt], num_graphs)
            edge_attr_new = F.leaky_relu(delta + edge_attr, 0.2)
            updated_x = self.mpnn(x, edge_index, edge_attr_new, batch_node, num_graphs)
            return updated_x, edge_attr_new
        updated_x = self.mpnn(x, edge_index, edge_attr, batch_node, num_graphs)
        return updated_x, edge_attr


class NNStacked(nn.Module):
    def __init__(self, num_layers, hidden_dim, first_block=False):
        super().__init__()
        layers = []
        for layer in range(num_layers):
            layers.append(NNBlock(
                hidden_dim, update_edge_attr=(not first_block or layer > 0)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr, batch_node, num_graphs):
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr, batch_node, num_graphs)
        return x


class NINBlock(nn.Module):
    def __init__(self, nn_num_layers, hidden_dim, first_block):
        super().__init__()
        self.nn_mpnn = NNStacked(nn_num_layers, hidden_dim, first_block)
        self.ni_mpnn = MPNN_NI(hidden_dim)
        self.ii_mpnn = MPNN_II(hidden_dim)
        self.stress_cal = LinearGraphNorm(hidden_dim * 2, hidden_dim)
        self.in_mpnn = MPNN_IN(hidden_dim)
        self.in_mpnn_ = MPNN_IN_(hidden_dim)
        self.du_cal = LinearGraphNorm(hidden_dim * 2, hidden_dim)
        self.first_block = first_block

    def forward(
        self, fext, x_n_o, x_n_u, edge_index_nn, edge_attr_nn, x_ip,
        edge_index_ni, edge_attr_ni, edge_index_ii, edge_attr_ii,
        edge_index_in, edge_attr_in, batch_node, batch_integration, num_graphs,
    ):
        if self.first_block:
            x_n_ = x_n_o
        else:
            tmp = torch.cat([x_n_o, x_n_u - x_n_o], dim=1)
            x_n_ = self.du_cal(tmp, batch_node, num_graphs)
        x_n_u = self.nn_mpnn(x_n_, edge_index_nn, edge_attr_nn, batch_node, num_graphs)
        stiffness_info = self.ni_mpnn(
            x_n_u, x_ip, edge_index_ni, edge_attr_ni, batch_integration, num_graphs
        )
        stiffness_mat = self.ii_mpnn(
            stiffness_info, edge_index_ii, edge_attr_ii, batch_integration, num_graphs
        )
        stress_info = torch.cat([stiffness_info, stiffness_info * stiffness_mat], dim=1)
        stress = self.stress_cal(stress_info, batch_integration, num_graphs)
        r_info = self.in_mpnn(stress, fext, edge_index_in, edge_attr_in, batch_node, num_graphs)
        du_info = self.in_mpnn_(stiffness_mat, r_info, edge_index_in, batch_node, num_graphs)
        updated_x_n = F.leaky_relu(du_info + x_n_u, 0.2)
        return updated_x_n, stiffness_info


class NINStacked(nn.Module):
    def __init__(self, nn_num_layers, nin_num_layers, hidden_dim):
        super().__init__()
        layers = []
        for layer in range(nin_num_layers):
            layers.append(NINBlock(nn_num_layers, hidden_dim, layer == 0))
        self.layers = nn.ModuleList(layers)
        self.tran_fext = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
        )

    def forward(
        self, x_n, edge_index_nn, edge_attr_nn, x_ip, edge_index_ni, edge_attr_ni,
        edge_index_ii, edge_attr_ii, edge_index_in, edge_attr_in,
        batch_node, batch_integration, num_graphs,
    ):
        fext = self.tran_fext(x_n)
        u = x_n
        for layer in self.layers:
            u, peeq = layer(
                fext, x_n, u, edge_index_nn, edge_attr_nn, x_ip,
                edge_index_ni, edge_attr_ni, edge_index_ii, edge_attr_ii,
                edge_index_in, edge_attr_in, batch_node, batch_integration, num_graphs,
            )
        return u, peeq


class Model(nn.Module):
    def __init__(
        self, nn_num_layers, nin_num_layers, hidden_dim,
    ):
        super().__init__()
        self.ve_lin = nn.Sequential(
            nn.Linear(11, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.vn_lin = nn.Sequential(
            nn.Linear(12, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.vi_lin = nn.Sequential(
            nn.Linear(10, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.eee_lin = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.enn_lin = nn.Sequential(
            nn.Linear(4, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.eni_lin = nn.Sequential(
            nn.Linear(6, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.ein_lin = nn.Sequential(
            nn.Linear(6, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.eii_lin = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.ee_mpnn = MPNN_EE(hidden_dim)
        self.en_mpnn = MPNN_EN(hidden_dim)
        self.nin = NINStacked(nn_num_layers, nin_num_layers, hidden_dim)
        self.disp_out = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2),
        )
        self.peeq_out = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 1),
            nn.Softplus(beta=1, threshold=20),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict, num_graphs):
        x_element = x_dict["element"]
        x_node = x_dict["node"]
        x_integration = x_dict["integration"]
        batch_element = batch_dict["element"]
        batch_node = batch_dict["node"]
        batch_integration = batch_dict["integration"]
        edge_index_en = edge_index_dict["element", "to", "node"]
        edge_index_ee = edge_index_dict["element", "to", "element"]
        edge_attr_ee = edge_attr_dict["element", "to", "element"]
        edge_index_nn = edge_index_dict["node", "to", "node"]
        edge_attr_nn = edge_attr_dict["node", "to", "node"]
        edge_index_ii = edge_index_dict["integration", "to", "integration"]
        edge_attr_ii = edge_attr_dict["integration", "to", "integration"]
        edge_index_ni = edge_index_dict["node", "to", "integration"]
        edge_attr_ni = edge_attr_dict["node", "to", "integration"]
        edge_index_in = edge_index_dict["integration", "to", "node"]
        edge_attr_in = edge_attr_dict["integration", "to", "node"]
        x_e = self.ve_lin(x_element)
        x_n = self.vn_lin(x_node)
        x_ip = self.vi_lin(x_integration)
        edge_attr_ee = self.eee_lin(edge_attr_ee)
        edge_attr_nn = self.enn_lin(edge_attr_nn)
        edge_attr_ii = self.eii_lin(edge_attr_ii)
        edge_attr_in = self.ein_lin(edge_attr_in)
        edge_attr_ni = self.eni_lin(edge_attr_ni)
        x_e_update = self.ee_mpnn(x_e, edge_index_ee, edge_attr_ee, batch_element, num_graphs)
        x_n = self.en_mpnn(x_e_update, x_n, edge_index_en, batch_node, num_graphs)
        d_Y, peeq_Y = self.nin(
            x_n, edge_index_nn, edge_attr_nn, x_ip, edge_index_ni, edge_attr_ni,
            edge_index_ii, edge_attr_ii, edge_index_in, edge_attr_in,
            batch_node, batch_integration, num_graphs,
        )

        return self.disp_out(d_Y), self.peeq_out(peeq_Y)


def loss_function(prediction, graph):
    pred_u, pred_peeq = prediction
    target_u = graph["node"].y
    target_peeq = graph["integration"].y
    loss_u = (pred_u - target_u).abs().mean()
    loss_peeq = (pred_peeq - target_peeq).abs().mean()
    loss = 0.5 * loss_u + 0.5 * loss_peeq
    return loss, loss_u, loss_peeq


def train_epoch(model, train_iter, optimizer, device):
    model.train()
    total_loss = total_u = total_peeq = 0.0
    for graph in train_iter:
        graph = graph.to(device)
        prediction = model(
            graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict,
            graph.batch_dict, graph.num_graphs,
        )
        loss, loss_u, loss_peeq = loss_function(prediction, graph)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        total_u += loss_u.detach().item()
        total_peeq += loss_peeq.detach().item()
    count = len(train_iter)
    return total_loss / count, total_u / count, total_peeq / count


def train():
    sample_path = Path(__file__).resolve().parent / "sample_P1_last_increment.pt"
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    graphs = [sample["graph"]]
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    model = Model(nn_num_layers=K, nin_num_layers=L, hidden_dim=D)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0.01 * LEARNING_RATE,
    )
    for epoch in range(1, EPOCHS + 1):
        train_loss, loss_u, loss_peeq = train_epoch(
            model, train_iter, optimizer, device,
        )
        scheduler.step()
        print(f"Epoch {epoch}, DD = {train_loss:.6g}, "
              f"DeltaU = {loss_u:.6g}, DeltaPEEQ = {loss_peeq:.6g}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": EPOCHS},
        OUTPUT_DIR / "model.pt",
    )


if __name__ == "__main__":
    train()
