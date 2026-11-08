"""Shared UPFD code: dataset with per-node time/depth, official UPFD model port, train/eval.

Dataset: torch_geometric.datasets.UPFD (train+val+test concatenated, original graph ids
restored from raw/{split}_idx.npy). Each graph gets extra node attributes, computed on the
directed tree as stored (edges parent -> child, root = local node 0):
    t      tweet timestamp (UNIX s) from GNN-FakeNews data/{pol,gos}_id_time_mapping.pkl,
           root = -inf (README: "The news node doesn't contain timestamp")
    trank  position in time order, ties broken by node index; root = 0
    depth  hop distance from the root
Edges stay directed here; transforms add ToUndirected (+ self-loops) like the official code.

Model: port of safe-graph/GNN-FakeNews gnn_model/gnn.py `Model` (1 conv layer, ReLU,
global_max_pool, optional news concat via lin0/lin1, lin2, log_softmax).
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import add_self_loops, subgraph, to_undirected

from upfd_structure import load_meta

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"

# gnn_model/gnn.py argparse defaults + hyper_parameters.md (UPFD-SAGE, bert)
HPARAMS = {
    "politifact": {"epochs": 30, "lr": 0.01},
    "gossipcop": {"epochs": 80, "lr": 0.001},
}
COMMON = {"nhid": 128, "batch_size": 128, "weight_decay": 0.01}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graphs(name: str, feature: str = "bert") -> list[Data]:
    """All graphs in original id order, with t / trank / depth / gid attached. Cached."""
    cache = CACHE / f"upfd_{name}_{feature}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    raw = DATA / "upfd" / name / "raw"
    gid_of_node = np.load(raw / "node_graph_id.npy")
    start = np.searchsorted(gid_of_node, np.arange(gid_of_node.max() + 1))
    times, _ = load_meta(name)
    out: dict[int, Data] = {}
    for split in ("train", "val", "test"):
        ds = UPFD(str(DATA / "upfd"), name, feature, split)
        idx = np.load(raw / f"{split}_idx.npy")
        assert len(ds) == len(idx)
        for g, d in zip(idx.tolist(), ds):
            n = d.num_nodes
            s = start[g]
            assert times[s] == "" and all(times[s + i] != "" for i in range(1, n)), f"graph {g}: time map misaligned"
            t = torch.tensor([-np.inf] + [float(times[s + i]) for i in range(1, n)], dtype=torch.float64)
            order = np.lexsort((np.arange(n), t.numpy()))  # by time, then index; root (-inf) first
            trank = torch.empty(n, dtype=torch.long)
            trank[torch.from_numpy(order)] = torch.arange(n)
            ch = [[] for _ in range(n)]
            for a, b in d.edge_index.t().tolist():
                ch[a].append(b)
            depth = torch.full((n,), -1, dtype=torch.long)
            depth[0] = 0
            q = deque([0])
            while q:
                u = q.popleft()
                for v in ch[u]:
                    depth[v] = depth[u] + 1
                    q.append(v)
            assert (depth >= 0).all() and trank[0] == 0
            out[g] = Data(x=d.x, edge_index=d.edge_index, y=d.y, t=t, trank=trank, depth=depth, gid=g)
    graphs = [out[g] for g in sorted(out)]
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, cache)
    return graphs


class Truncate(BaseTransform):
    """Keep the root plus an early part of the (directed) propagation tree, re-indexed.

    mode='time',  k: the first k nodes in time order (root counts as 1; k=1 -> root only).
                     Connected by construction: child time >= parent time for 100% of
                     user-user edges and parent index < child index (checked in upfd_structure.py).
    mode='depth', k: nodes with depth <= k (k=0 -> root only).
    mode='full':     identity.
    """

    def __init__(self, mode: str, k: int | None = None):
        self.mode, self.k = mode, k

    def forward(self, data: Data) -> Data:
        if self.mode == "full":
            return data
        mask = data.trank < self.k if self.mode == "time" else data.depth <= self.k
        mask[0] = True
        ei, _ = subgraph(mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        out = Data(x=data.x[mask], edge_index=ei, y=data.y, t=data.t[mask],
                   trank=data.trank[mask], depth=data.depth[mask], gid=data.gid)
        assert ei.size(1) == int(mask.sum()) - 1, "truncated graph is not a tree"
        return out


class OfficialEdges(BaseTransform):
    """FNNDataset adds self-loops at load time, then gnn.py applies ToUndirected().

    Only edge_index is touched (the official data has no edge attributes). PyG's ToUndirected
    is NOT used: it infers edge attributes by size, so on a 2-node graph (1 edge) it treats the
    graph label y (size 1) as an edge attribute and duplicates it.
    """

    def forward(self, data: Data) -> Data:
        ei = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        data.edge_index, _ = add_self_loops(ei, num_nodes=data.num_nodes)
        return data


def official_transform() -> OfficialEdges:
    return OfficialEdges()


def apply(graphs: list[Data], *transforms) -> list[Data]:
    t = Compose(list(transforms))
    return [t(g.clone()) for g in graphs]


class UPFDModel(torch.nn.Module):
    """Port of GNN-FakeNews gnn_model/gnn.py Model (forward semantics unchanged)."""

    def __init__(self, in_dim: int, nhid: int, n_cls: int, conv: str = "sage", concat: bool = True):
        super().__init__()
        self.concat = concat
        self.conv1 = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}[conv](in_dim, nhid)
        if concat:
            self.lin0 = torch.nn.Linear(in_dim, nhid)
            self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, n_cls)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = gmp(x, data.batch)
        if self.concat:
            # official: data.x[(data.batch == idx).nonzero().squeeze()[0]] == first node of each graph
            news = F.relu(self.lin0(data.x[data.ptr[:-1]]))
            x = F.relu(self.lin1(torch.cat([x, news], dim=1)))
        return F.log_softmax(self.lin2(x), dim=-1)


class NewsMLP(torch.nn.Module):
    """News-text-only baseline: the concat branch alone (lin0 -> ReLU -> lin2) on the root embedding."""

    def __init__(self, in_dim: int, nhid: int, n_cls: int):
        super().__init__()
        self.lin0 = torch.nn.Linear(in_dim, nhid)
        self.lin2 = torch.nn.Linear(nhid, n_cls)

    def forward(self, data):
        return F.log_softmax(self.lin2(F.relu(self.lin0(data.x[data.ptr[:-1]]))), dim=-1)


def random_split_idx(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """20/10/70 like gnn.py (int() truncation, test gets the rest), seeded permutation."""
    n_tr, n_va = int(n * 0.2), int(n * 0.1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def fixed_split_idx(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = DATA / "upfd" / name / "raw"
    return tuple(np.load(raw / f"{s}_idx.npy") for s in ("train", "val", "test"))


def train(model: torch.nn.Module, train_graphs: list[Data], epochs: int, lr: float, seed: int) -> torch.nn.Module:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=COMMON["weight_decay"])
    loader = DataLoader(train_graphs, batch_size=COMMON["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(epochs):  # official: fixed epochs, last epoch is evaluated, no model selection
        for b in loader:
            opt.zero_grad()
            F.nll_loss(model(b), b.y).backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, graphs: list[Data]) -> dict:
    model.eval()
    ys, ps = [], []
    for b in DataLoader(graphs, batch_size=512):
        ps.append(model(b).argmax(1))
        ys.append(b.y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {"acc": float(accuracy_score(y, p)), "f1": float(f1_score(y, p, average="macro"))}


def make_model(kind: str, in_dim: int, seed: int) -> torch.nn.Module:
    set_seed(seed)
    if kind == "news_mlp":
        return NewsMLP(in_dim, COMMON["nhid"], 2)
    conv, concat = {"sage_concat": ("sage", True), "sage_noconcat": ("sage", False),
                    "gcn_concat": ("gcn", True)}[kind]
    return UPFDModel(in_dim, COMMON["nhid"], 2, conv, concat)
