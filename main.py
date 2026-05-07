import argparse
import json
import os
import os.path as osp
import sys
import time
from math import ceil

import matplotlib
import matplotlib.pyplot
import networkx as nx
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GPSE,
    DenseGCNConv,
    DenseGraphConv,
    GCNConv,
    dense_diff_pool,
    dense_mincut_pool,
)
from torch_geometric.nn.models import GCN, MLP
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    Compose,
    Constant,
)
from torch_geometric.utils import to_dense_adj, to_dense_batch

from libs.lib import AddGPSEPE, AddNode2VecPE
from libs.pooling_lib import (
    AddFlow,
    LabelsToInt,
    dmon_pool,
    jb_pool,
    mdl_pool,
)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=1, type=int)
parser.add_argument(
    "-p",
    "--pooler",
    default="none",
    type=str,
    choices=["none", "mincut", "diffpool", "dmon", "mdlpool", "jbpool"],
)
parser.add_argument(
    "--pe",
    default="none",
    type=str,
    choices=["none", "laplacian", "node2vec", "rw", "gpse"],
)
parser.add_argument("--pe-dim", default=6, type=int)
parser.add_argument("-c", "--hidden-channels", default=200, type=int)
parser.add_argument(
    "-d",
    "--dataset",
    default="PROTEINS",
    type=str,
    choices=[
        "PROTEINS",
        "MUTAG",
        "ENZYMES",
        "COLLAB",
        "IMDB-BINARY",
        "REDDIT-BINARY",
        "COLORS-3",
        "DD",
        "Mutagenicity",
        "NCI1",
    ],
)
parser.add_argument("--sel", default="mlp", type=str, choices=["mlp", "gcn"])
parser.add_argument("--num_gcn_layers", default=1, type=int)
parser.add_argument("--output_dir", default="results", type=str)
parser.add_argument("--draw_assignments", default=False, type=bool)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_dir = args.output_dir
os.makedirs(res_dir, exist_ok=True)

data_root = osp.join(osp.dirname(osp.realpath(__file__)), "data")

pe_dim = args.pe_dim
min_size = 8

match args.pe:
    case "none":
        transform = None
    case "laplacian":
        transform = AddLaplacianEigenvectorPE(k=pe_dim, attr_name="pe")
    case "node2vec":
        transform = AddNode2VecPE(
            embedding_dim=pe_dim,
            walks_per_node=10,
            n2v_epochs=400,
            attr_name="pe",
            device=device,
        )
    case "rw":
        transform = AddRandomWalkPE(walk_length=pe_dim, attr_name="pe")
    case "gpse":
        gpse_model = GPSE.from_pretrained("zinc", root=osp.join(data_root, "gpse"))
        transform = AddGPSEPE(model=gpse_model, proj_dim=pe_dim)

match args.pooler:
    case "none":
        pooler = None
    case "mincut":
        pooler = dense_mincut_pool
    case "diffpool":
        pooler = dense_diff_pool
    case "dmon":
        pooler = dmon_pool
    case "mdlpool":
        pooler = mdl_pool
    case "jbpool":
        pooler = jb_pool


transforms_list = []
use_node_attr = False
if transform is not None:
    transforms_list.append(transform)
if pooler == mdl_pool:
    transforms_list.append(AddFlow())
if args.dataset == "COLORS-3":
    transforms_list.append(LabelsToInt())
    use_node_attr = True
transforms_list.append(Constant())
transform = Compose(transforms_list)

path_suffix = (
    f"{args.dataset}-{args.pe}-{args.pooler}"
    if pooler == mdl_pool
    else f"{args.dataset}-{args.pe}"
)
path = osp.join(data_root, path_suffix)

dataset = TUDataset(
    path,
    name=args.dataset,
    use_node_attr=use_node_attr,
    pre_transform=transform,
    pre_filter=lambda data: data.num_nodes > min_size,
)

avg_num_nodes = int(sum([d.num_nodes for d in dataset]) / len(dataset))
torch.manual_seed(0)
dataset = dataset[torch.randperm(len(dataset))]
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n : 2 * n]
train_dataset = dataset[2 * n :]
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)


class DenseGCNPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(DenseGCNConv(in_channels, out_channels))
        else:
            self.convs.append(DenseGCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(DenseGCNConv(hidden_channels, hidden_channels))
            self.convs.append(DenseGCNConv(hidden_channels, out_channels))

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i != len(self.convs) - 1:
                x = x.relu()
        return x


def reduce(x, s, mask=None, temp=1.0):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    return torch.matmul(s.transpose(1, 2), x)


class Net(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels=32, pooler=None, use_pe=True
    ):
        super().__init__()

        self.pooler = pooler
        self.use_pe = use_pe

        self.conv1 = GCNConv(in_channels, hidden_channels)

        num_nodes = ceil(0.5 * avg_num_nodes)

        if use_pe:
            self.trim = Linear(hidden_channels, pe_dim)

        if args.sel == "mlp":
            self.pool = MLP(
                [hidden_channels, hidden_channels, num_nodes], norm=None, dropout=0.5
            )
        elif args.sel == "gcn":
            self.pool = DenseGCNPool(
                hidden_channels,
                hidden_channels,
                num_nodes,
                num_layers=args.num_gcn_layers,
            )

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, batch, pe=None, f=None, p=None):
        aux_loss = 0

        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, 0.5, self.training)

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = None
        if self.pooler != None:
            h = x
            if pe != None and self.use_pe:
                pe, _ = to_dense_batch(pe, batch)
                h = h.detach()
                h = self.trim(h).relu()
                h = F.dropout(h, 0.5, self.training)
                h = torch.cat((h, pe), dim=-1)
            if args.sel == "gcn":
                s = self.pool(h, adj)
            elif args.sel == "mlp":
                s = self.pool(h)
            if self.pooler == dmon_pool:
                x, adj, sp, _, c = self.pooler(x, adj, s, mask)
                aux_loss += sp + c
            elif self.pooler == mdl_pool:
                if f is None or p is None:
                    raise ValueError("mdlpool requires flow attributes (f, p). ")
                flow = to_dense_adj(edge_index, batch, edge_attr=f)
                p_dense, _ = to_dense_batch(p, batch)
                x, adj, loss = self.pooler(x, adj, s, flow, p_dense, mask)
                aux_loss += loss
            elif self.pooler == jb_pool:
                x, adj, loss = self.pooler(x, adj, s, mask)
                aux_loss += loss
            else:
                x, adj, mc1, o1 = self.pooler(x, adj, s, mask)
                aux_loss += mc1 + o1

        x = self.conv3(x, adj).relu()
        x = F.dropout(x, 0.5, self.training)

        x = x.max(dim=1)[0]
        x = self.lin1(x).relu()
        x = F.dropout(x, 0.5, self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), aux_loss, s


torch.manual_seed(args.seed)

model = Net(
    dataset.num_features,
    dataset.num_classes,
    pooler=pooler,
    use_pe=args.pe != "none",
    hidden_channels=args.hidden_channels,
).to(device)

optimizer = torch.optim.AdamW(model.parameters())


def train(epoch):
    model.train()
    loss_all = 0
    aux_loss_all = 0

    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, aux_loss, s = model(
            data.x,
            data.edge_index,
            data.batch,
            None if args.pe == "none" else data.pe,
            f=data.f if hasattr(data, "f") else None,
            p=data.p if hasattr(data, "p") else None,
        )

        loss = F.nll_loss(out, data.y.view(-1)) + aux_loss
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        aux_loss_all += float(aux_loss)
        optimizer.step()
    return loss_all / len(train_dataset), aux_loss_all / (i + 1)


graph_layout = {}


@torch.no_grad()
def test(loader, epoch, print=False):
    global graph_layout

    model.eval()
    loss_all = 0
    aux_loss_all = 0

    preds = []
    truths = []

    graph_idx = 0

    for i, data in enumerate(loader):
        data = data.to(device)
        pred, aux_loss, s = model(
            data.x,
            data.edge_index,
            data.batch,
            None if args.pe == "none" else data.pe,
            f=data.f if hasattr(data, "f") else None,
            p=data.p if hasattr(data, "p") else None,
        )
        if print:
            adj = to_dense_adj(data.edge_index, data.batch)
            batch_size = adj.size(0)
            for a in range(batch_size):
                g = nx.from_numpy_array(adj[a].cpu().numpy())
                palette = sns.color_palette(
                    "colorblind", n_colors=max(1, int(s[a].shape[-1]))
                )
                nx.set_node_attributes(
                    g,
                    {i: c for i, c in enumerate(s[a].argmax(dim=-1).cpu().numpy())},
                    "color",
                )
                g.remove_nodes_from(list(nx.isolates(g)))

                if graph_idx not in graph_layout:
                    graph_layout[graph_idx] = nx.kamada_kawai_layout(g)

                graph_dir = f"{res_dir}/graphs/{args.dataset}/{args.pooler}/{args.pe}/"
                os.makedirs(graph_dir, exist_ok=True)

                fig = matplotlib.pyplot.figure()
                nx.draw(
                    g,
                    ax=fig.add_subplot(),
                    node_color=[
                        palette[int(color) % len(palette)]
                        for color in nx.get_node_attributes(g, "color").values()
                    ],
                    pos=graph_layout[graph_idx],
                )
                matplotlib.use("Agg")
                fig.tight_layout()
                fig.savefig(
                    f"{graph_dir}/{graph_idx}.pdf",
                    transparent=True,
                    bbox_inches="tight",
                )
                matplotlib.pyplot.close(fig)

                graph_idx += 1

        loss = F.nll_loss(pred, data.y.view(-1)) + aux_loss
        loss_all += data.y.size(0) * float(loss)
        aux_loss_all += float(aux_loss)
        preds.append(pred.max(dim=1)[1].cpu())
        truths.append(data.y.view(-1).cpu())

    truths = torch.cat(truths, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    acc = accuracy_score(truths, preds)
    bacc = balanced_accuracy_score(truths, preds)
    return (
        loss_all / len(loader.dataset),
        aux_loss_all / (i + 1),
        acc,
        bacc,
    )


times = []
best_val_acc = test_acc = test_bacc = 0
best_val_loss = float("inf")
patience = start_patience = 50
best_epoch = -1
for epoch in range(1, 15001):
    start = time.time()
    train_loss, train_aux_loss = train(epoch)
    _, _, train_acc, train_bacc = test(train_loader, epoch)
    val_loss, _, val_acc, val_bacc = test(val_loader, epoch)
    if val_loss < best_val_loss:
        best_epoch = epoch
        test_loss, test_aux_loss, test_acc, test_bacc = test(
            test_loader, epoch, print=args.draw_assignments
        )
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1
        if patience == 0:
            break
    print(
        f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, "
        f"Train Acc: {train_acc:.3f}, Train BAcc: {train_bacc:.3f}, "
        f"Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.3f}, "
        f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, "
        f"Test BAcc: {test_bacc:.3f}, "
        f"Train Aux Loss: {train_aux_loss:.5f}, "
        f"Test Aux Loss: {test_aux_loss:.5f}"
    )
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


result = {
    "dataset": args.dataset,
    "pooler": args.pooler,
    "acc": test_acc,
    "bacc": test_bacc,
    "epoch": best_epoch,
    "time": torch.tensor(times).median().item(),
    "pe": args.pe,
    "seed": args.seed,
    "aux-loss": test_aux_loss,
    "sel": args.sel,
    "num-gcn-layers": args.num_gcn_layers,
}


with open(f"{res_dir}/{args.dataset}_result.json", "a") as json_file:
    json_file.write(json.dumps(result))
    json_file.write("\n")
