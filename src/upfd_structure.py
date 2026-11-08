"""Checks 1-2: node order vs time order, and graph structure, on the raw UPFD files.

Reads exactly what PyG's UPFD.process() reads (raw/A.txt, raw/node_graph_id.npy,
raw/graph_labels.npy, raw/{train,val,test}_idx.npy) plus the official
data/{pol,gos}_id_{time,twitter}_mapping.pkl from safe-graph/GNN-FakeNews.
Writes artifacts/upfd_structure.json and artifacts/upfd_graph_stats.csv.
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"
PREFIX = {"politifact": "pol", "gossipcop": "gos"}
KS = (5, 10, 20, 50)


class _NoGlobals(pickle.Unpickler):
    """The mapping pickles are plain dicts; refuse anything that needs an import."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_meta(name: str) -> tuple[dict, dict]:
    p = DATA / "upfd_meta"
    with open(p / f"{PREFIX[name]}_id_time_mapping.pkl", "rb") as f:
        times = _NoGlobals(f).load()
    with open(p / f"{PREFIX[name]}_id_twitter_mapping.pkl", "rb") as f:
        tw = _NoGlobals(f).load()
    return times, tw


def pct(s) -> dict:
    s = pd.Series(s, dtype=float)
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {"mean": round(s.mean(), 2), **{f"p{int(k * 100)}": float(v) for k, v in q.items()}, "max": float(s.max())}


def analyse(name: str) -> tuple[dict, pd.DataFrame]:
    raw = DATA / "upfd" / name / "raw"
    A = np.loadtxt(raw / "A.txt", delimiter=",", dtype=np.int64)
    gid = np.load(raw / "node_graph_id.npy")
    y_raw = np.load(raw / "graph_labels.npy")
    y = np.unique(y_raw, return_inverse=True)[1].reshape(-1)  # same mapping as PyG UPFD.process()
    times, tw = load_meta(name)
    n_nodes = len(gid)

    res: dict = {"n_nodes_total": int(n_nodes), "n_edges_total": int(len(A)), "n_graphs": int(len(y)),
                 "graph_label_raw_values": {str(k): int(v) for k, v in zip(*np.unique(y_raw, return_counts=True))},
                 "node_graph_id_sorted": bool(np.all(np.diff(gid) >= 0)),
                 "time_map_keys_eq_0..N-1": sorted(times) == list(range(n_nodes)),
                 "self_loops_in_A": int((A[:, 0] == A[:, 1]).sum()),
                 "duplicate_edges_in_A": int(len(A) - len(np.unique(A, axis=0))),
                 "edges_crossing_graphs": int((gid[A[:, 0]] != gid[A[:, 1]]).sum())}

    start = np.searchsorted(gid, np.arange(len(y)))
    end = np.searchsorted(gid, np.arange(len(y)), side="right")
    news_nodes = {i for i, v in tw.items() if v.startswith(("politifact", "gossipcop"))}
    res["news_nodes_total"] = len(news_nodes)
    res["news_node_is_first_node"] = int(sum(start[g] in news_nodes for g in range(len(y))))
    res["empty_time_nodes_are_news"] = int(sum(1 for i, t in times.items() if t == "" and i in news_nodes))
    res["empty_time_nodes_total"] = int(sum(1 for t in times.values() if t == ""))

    # edge direction as stored
    src_lt_dst = A[:, 0] < A[:, 1]
    res["edges_src_index_lt_dst"] = int(src_lt_dst.sum())
    res["edges_from_news_node"] = int(np.isin(A[:, 0], list(news_nodes)).sum())
    res["edges_into_news_node"] = int(np.isin(A[:, 1], list(news_nodes)).sum())

    # time checks on user->user edges and on node order
    t = np.array([np.nan if times[i] == "" else float(times[i]) for i in range(n_nodes)])
    uu = ~np.isin(A[:, 0], list(news_nodes))
    dt = t[A[uu, 1]] - t[A[uu, 0]]
    res["user_user_edges"] = int(uu.sum())
    res["user_user_child_time_ge_parent"] = int((dt >= 0).sum())

    rows, per_graph_sorted, adj_pairs, adj_ok, inv_frac = [], 0, 0, 0, []
    in_deg_bad = not_tree = disconnected = 0
    edges_by_graph = [[] for _ in range(len(y))]
    for s, d in A:
        edges_by_graph[gid[s]].append((s, d))
    for g in range(len(y)):
        a, b = start[g], end[g]
        n = b - a
        tu = t[a + 1:b]
        if n > 1:
            dif = np.diff(tu)
            adj_pairs += len(dif)
            adj_ok += int((dif >= 0).sum())
            per_graph_sorted += int(np.all(dif >= 0))
        else:
            per_graph_sorted += 1
        E = [(s - a, d - a) for s, d in edges_by_graph[g]]
        indeg = np.bincount([d for _, d in E], minlength=n)
        if indeg[0] != 0 or np.any(indeg[1:] != 1):
            in_deg_bad += 1
        if len(E) != n - 1:
            not_tree += 1
        und = [[] for _ in range(n)]
        for s, d in E:
            und[s].append(d)
            und[d].append(s)
        depth = np.full(n, -1)
        depth[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            for v in und[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    q.append(v)
        disconnected += int((depth < 0).any())
        root_deg = len(und[0])
        span_h = (np.nanmax(tu) - np.nanmin(tu)) / 3600 if n > 1 else 0.0
        rows.append({"graph": g, "label": int(y[g]), "n_nodes": n, "depth": int(depth.max()),
                     "root_degree": root_deg, "root_deg_frac": root_deg / max(n - 1, 1),
                     "parents_before_children": all(s < d for s, d in E), "time_span_h": span_h})
    df = pd.DataFrame(rows)
    res["graphs_users_time_sorted_by_index"] = int(per_graph_sorted)
    res["adjacent_user_pairs_time_nondecreasing"] = f"{adj_ok}/{adj_pairs}"
    res["graphs_all_edges_parent_index_lt_child"] = int(df.parents_before_children.sum())
    res["graphs_bad_in_degree(root!=0 or user!=1)"] = int(in_deg_bad)
    res["graphs_edges_ne_n_minus_1"] = int(not_tree)
    res["graphs_disconnected_from_root"] = int(disconnected)
    res["graphs_root_star(depth==1)"] = int((df.depth == 1).sum())
    res["single_node_graphs"] = int((df.n_nodes == 1).sum())

    for col in ("n_nodes", "depth", "root_degree", "root_deg_frac", "time_span_h"):
        res[f"{col}_all"] = pct(df[col])
        for lab in sorted(df.label.unique()):
            res[f"{col}_label{lab}"] = pct(df.loc[df.label == lab, col])
    res["graphs_with_n_nodes_le_k"] = {k: int((df.n_nodes <= k).sum()) for k in KS}
    res["graphs_with_n_nodes_le_k_by_label"] = {k: {int(l): int(((df.n_nodes <= k) & (df.label == l)).sum())
                                                    for l in sorted(df.label.unique())} for k in KS}
    res["graphs_with_depth_le_d"] = {d: int((df.depth <= d).sum()) for d in (1, 2, 3)}

    # size/depth-only classifier on the fixed split
    tr, te = np.load(raw / "train_idx.npy"), np.load(raw / "test_idx.npy")
    X = np.c_[np.log1p(df.n_nodes), df.depth, np.log1p(df.root_degree)]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y[tr])
    p = m.predict(X[te])
    maj = np.full_like(y[te], np.bincount(y[tr]).argmax())
    res["lr_size_depth_rootdeg_fixed_split"] = {
        "acc": round(accuracy_score(y[te], p), 4), "f1": round(f1_score(y[te], p, average="macro"), 4),
        "majority_acc": round(accuracy_score(y[te], maj), 4), "n_train": int(len(tr)), "n_test": int(len(te)),
        "coef(log_n, depth, log_rootdeg)": [round(c, 3) for c in m[-1].coef_[0]]}
    return res, df


def main() -> None:
    OUT.mkdir(exist_ok=True)
    allres, dfs = {}, []
    for name in PREFIX:
        r, df = analyse(name)
        allres[name] = r
        dfs.append(df.assign(dataset=name))
    pd.concat(dfs).to_csv(OUT / "upfd_graph_stats.csv", index=False)
    (OUT / "upfd_structure.json").write_text(json.dumps(allres, indent=1, default=str))
    print(json.dumps(allres, indent=1, default=str))


if __name__ == "__main__":
    main()
