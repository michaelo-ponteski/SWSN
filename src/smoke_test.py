"""Smoke test (v1): plain 2-layer GCN + global mean pool on UPFD, PyG fixed split, no tuning.

Usage:
    python src/smoke_test.py --upfd politifact --feature bert
(The Twitter15/16 variant lives in archive/twitter1516/src/smoke_test.py.)
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEED = 0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_cls: int):
        super().__init__()
        self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, n_cls)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.c2(x, edge_index))
        return self.lin(global_mean_pool(x, batch))


def train_eval(train_ds, test_ds, n_cls: int, epochs: int, lr: float = 0.01, seed: int = SEED) -> dict:
    set_seed(seed)
    model = GCN(train_ds[0].num_node_features, 64, n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=gen)
    for _ in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y).backward()
            opt.step()
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in DataLoader(test_ds, batch_size=256):
            ps.append(model(b.x, b.edge_index, b.batch).argmax(1))
            ys.append(b.y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {"acc": accuracy_score(y, p), "f1": f1_score(y, p, average="macro")}


def majority(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    p = np.full_like(y_test, np.bincount(y_train).argmax())
    return {"acc": accuracy_score(y_test, p), "f1": f1_score(y_test, p, average="macro")}


def summarize(rows: list[dict]) -> dict:
    return {k: f"{np.mean([r[k] for r in rows]):.3f} ± {np.std([r[k] for r in rows]):.3f}" for k in rows[0]}


def run_upfd(name: str, feature: str, epochs: int) -> dict:
    from torch_geometric.datasets import UPFD
    from torch_geometric.transforms import ToUndirected

    root = str(DATA / "upfd")
    tr, va, te = (UPFD(root, name, feature, s, transform=ToUndirected()) for s in ("train", "val", "test"))
    r = train_eval(tr, te, 2, epochs)
    return {"dataset": f"UPFD/{name}", "feature": feature, "sizes": [len(tr), len(va), len(te)],
            "epochs": epochs, "gcn": {k: round(v, 3) for k, v in r.items()},
            "majority": {k: round(v, 3) for k, v in majority(tr.y.numpy(), te.y.numpy()).items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upfd", default="politifact", choices=["politifact", "gossipcop"])
    ap.add_argument("--feature", default="bert", choices=["profile", "spacy", "bert", "content"])
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(4)
    res = run_upfd(a.upfd, a.feature, a.epochs)
    print(json.dumps(res, indent=1))
    out = ROOT / "artifacts" / "smoke_results.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
