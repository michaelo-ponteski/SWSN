"""Build data/processed/<dataset>/{graphs,nodes[,edges_raw]}.parquet from data/raw (see README, section Data).

Every cascade becomes a rooted tree with nodes in time order: node_idx 0 = root, parent_idx < node_idx and
t_parent <= t_child, so any prefix (first k nodes / t_min <= h) is a connected early-propagation tree.
"""
from __future__ import annotations

import ast
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RAW, OUT = DATA / "raw", DATA / "processed"
TW_LABELS = {"non-rumor": 0, "false": 1, "true": 2, "unverified": 3}
TW_BINARY = {"false": 1, "true": 0, "non-rumor": 0}  # 1 = fake; unverified -> <NA>


def depths(parent: np.ndarray) -> np.ndarray:
    d = np.zeros(len(parent), dtype=np.int32)
    for i in range(1, len(parent)):
        d[i] = d[parent[i]] + 1
    return d


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\burl\b|@\w+|#", " ", s.lower())).strip()


def check(graphs: pd.DataFrame, nodes: pd.DataFrame) -> None:
    p = nodes[nodes.parent_idx >= 0]
    parent = nodes.set_index(["graph_id", "node_idx"]).loc[list(zip(p.graph_id, p.parent_idx))]
    assert len(nodes) == graphs.n_nodes.sum() and (nodes.parent_idx == -1).sum() == len(graphs)
    assert (p.parent_idx.values < p.node_idx.values).all()
    assert (parent.t_min.values <= p.t_min.values).all() and (parent.depth.values + 1 == p.depth.values).all()


def build_twitter(name: str, other: str) -> None:
    d = RAW / "rumor_detection_acl2017" / name
    labels = dict(reversed(l.split(":")) for l in (d / "label.txt").read_text().split())
    texts = dict(l.split("\t", 1) for l in (d / "source_tweets.txt").read_text().splitlines() if l.strip())
    in_other = {p.stem for p in (d.parent / other / "tree").glob("*.txt")}
    graphs, nodes, edges = [], [], []
    for path in sorted((d / "tree").glob("*.txt")):
        gid = path.stem
        raw = [tuple(map(ast.literal_eval, l.split("->"))) for l in path.read_text().splitlines()]
        root = next(b for a, b in raw if a[0] == "ROOT")
        info = {}  # uid -> (tweet_id, raw delay); uid is unique per tree, the tweet_id field is not
        for a, b in raw:
            for n in (a, b):
                if n[0] != "ROOT":
                    info.setdefault(n[0], (n[1], float(n[2])))
        # pre-source posts (negative delay) are clamped to t = 0; ties keep first appearance order
        t = {u: 0.0 if u == root[0] else max(v[1], 0.0) for u, v in info.items()}
        order = [root[0]] + sorted((u for u in info if u != root[0]), key=t.get)
        pos = {u: i for i, u in enumerate(order)}
        cand = defaultdict(set)
        for a, b in raw:
            if a[0] == "ROOT":
                continue
            reason = "self_loop" if a[0] == b[0] else "into_root" if b[0] == root[0] else None
            if reason is None:
                cand[pos[b[0]]].add(pos[a[0]])
            edges.append((gid, pos[a[0]], pos[b[0]], float(a[2]), float(b[2]), reason))
        # several candidate parents: take the most recent earlier node (latest exposure); fallback root
        parent = np.array([-1] + [max((q for q in cand[i] if q < i), default=0) for i in range(1, len(order))])
        dep = depths(parent)
        for i, u in enumerate(order):
            nodes.append((gid, i, parent[i], dep[i], t[u], info[u][1], u, info[u][0], t[u] != info[u][1]))
        graphs.append(dict(graph_id=gid, label=TW_LABELS[labels[gid]], label_name=labels[gid],
                           label_binary=TW_BINARY.get(labels[gid]), text=texts[gid], text_norm=norm_text(texts[gid]),
                           also_in_other_twitter=gid in in_other, n_nodes=len(order), max_depth=int(dep.max()),
                           t_span_min=max(t.values()), n_multi_parent=sum(len(c) > 1 for c in cand.values())))
    graphs = pd.DataFrame(graphs).astype({"label_binary": "Int8"})
    nodes = pd.DataFrame(nodes, columns=["graph_id", "node_idx", "parent_idx", "depth", "t_min", "t_raw_min",
                                         "user_id", "tweet_id", "t_clamped"])
    edges = pd.DataFrame(edges, columns=["graph_id", "src_idx", "dst_idx", "src_t_raw", "dst_t_raw", "dropped_reason"])
    tree = set(zip(nodes.graph_id, nodes.parent_idx, nodes.node_idx))
    edges["in_tree"] = [e in tree for e in zip(edges.graph_id, edges.src_idx, edges.dst_idx)]
    edges.loc[edges.dropped_reason.isna() & ~edges.in_tree, "dropped_reason"] = "not_chosen_parent"
    assert edges.in_tree.sum() == len(nodes) - len(graphs)
    save(name, graphs, nodes, edges)


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):  # the mapping pickles are plain dicts
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def build_upfd(name: str, prefix: str) -> None:
    raw, meta = RAW / "upfd" / name, RAW / "upfd_meta"
    A = np.loadtxt(raw / "A.txt", delimiter=",", dtype=np.int64)
    graph_of = np.load(raw / "node_graph_id.npy")
    y = np.load(raw / "graph_labels.npy").astype(int)
    times = _NoGlobals(open(meta / f"{prefix}_id_time_mapping.pkl", "rb")).load()
    tw = _NoGlobals(open(meta / f"{prefix}_id_twitter_mapping.pkl", "rb")).load()
    starts = np.searchsorted(graph_of, np.arange(len(y) + 1))
    par = np.full(len(graph_of), -1)
    par[A[:, 1]] = A[:, 0]
    split = np.empty(len(y), dtype=object)
    for s in ("train", "val", "test"):
        split[np.load(raw / f"{s}_idx.npy")] = s
    graphs, nodes = [], []
    for g in range(len(y)):
        s, e = starts[g], starts[g + 1]
        tu = np.array([times[i] for i in range(s + 1, e)], dtype=np.int64)  # the news node has no timestamp
        t0 = tu.min()  # origin = first engagement
        order = np.r_[0, 1 + np.argsort(tu, kind="stable")]
        pos = np.empty_like(order)
        pos[order] = np.arange(len(order))
        parent = np.r_[-1, pos[par[s + order[1:]] - s]]
        dep = depths(parent)
        for i, o in enumerate(order):
            nodes.append((g, i, parent[i], dep[i], 0.0 if o == 0 else (tu[o - 1] - t0) / 60,
                          None if o == 0 else tu[o - 1], None if o == 0 else str(tw[s + o]), s + o))
        graphs.append(dict(graph_id=g, news_id=tw[s], label=y[g], label_name=["real", "fake"][y[g]],
                           label_binary=y[g], pyg_split=split[g], n_nodes=e - s, max_depth=int(dep.max()),
                           t_span_min=(tu.max() - t0) / 60, t0_unix=t0,
                           cascade=(tuple(tw[i] for i in range(s + 1, e)), tuple(tu))))
    graphs = pd.DataFrame(graphs)
    # different news items with an identical cascade (same users, same times) share a dup_group
    graphs["dup_group"] = graphs.groupby("cascade", sort=False).graph_id.transform("min")
    nodes = pd.DataFrame(nodes, columns=["graph_id", "node_idx", "parent_idx", "depth", "t_min", "t_unix",
                                         "user_id", "feat_row"]).astype({"t_unix": "Int64"})
    save(name, graphs.drop(columns="cascade"), nodes)


def save(name: str, graphs: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame | None = None) -> None:
    check(graphs, nodes)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    graphs.to_parquet(out / "graphs.parquet", index=False)
    nodes.to_parquet(out / "nodes.parquet", index=False)
    if edges is not None:
        edges.to_parquet(out / "edges_raw.parquet", index=False)
        print(name, "dropped edges:", edges.dropped_reason.value_counts().to_dict())
    print(f"{name}: {len(graphs)} graphs, {len(nodes)} nodes -> {out}")


if __name__ == "__main__":
    build_twitter("twitter15", "twitter16")
    build_twitter("twitter16", "twitter15")
    build_upfd("politifact", "pol")
    build_upfd("gossipcop", "gos")
