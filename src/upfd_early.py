"""Check 4: early-detection smoke test. UPFD (SAGE + BERT + concat), official hparams, 3 seeds,
official-style random 20/10/70 split per seed.

Truncations (Truncate in upfd_common):  time-k (first k nodes by tweet time, root = 1),
depth-d (nodes within d hops of root), full.
Protocols:  (i)  train on full graphs, test on truncated
            (ii) train and test on truncated graphs (same setting)
Baselines:  news_mlp (root BERT only; independent of k)
            lr_struct (LogReg on [log n_nodes, max depth, log root degree] of the *truncated* graph, protocol ii)
Each test score is reported on all test graphs and on the 'affected' subset
(test graphs that the truncation actually changed, i.e. n_nodes > k for time-k).
Writes artifacts/upfd_early.json and artifacts/upfd_early_table.md.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from upfd_common import (HPARAMS, ROOT, Truncate, apply, evaluate, load_graphs, make_model,
                         official_transform, random_split_idx, train)

SEEDS = (0, 1, 2)
SETTINGS = [("full", None), ("time", 1), ("time", 5), ("time", 10), ("time", 20), ("time", 50),
            ("depth", 1), ("depth", 2), ("depth", 3)]


def struct_feats(g) -> list[float]:
    root_deg = int((g.edge_index[0] == 0).sum())
    return [np.log1p(g.num_nodes), float(g.depth.max()), np.log1p(root_deg)]


def sk_score(y, p) -> dict:
    return {"acc": float(accuracy_score(y, p)), "f1": float(f1_score(y, p, average="macro"))}


def main() -> None:
    torch.set_num_threads(8)
    res: dict = defaultdict(list)
    for name in ("politifact", "gossipcop"):
        base = load_graphs(name)
        n_full = np.array([g.num_nodes for g in base])
        hp = HPARAMS[name]
        full = apply(base, official_transform())
        trunc = {}
        for mode, k in SETTINGS:
            directed = apply(base, Truncate(mode, k))
            trunc[(mode, k)] = (directed, apply(directed, official_transform()))
        for seed in SEEDS:
            tr, _, te = random_split_idx(len(base), seed)
            t0 = time.time()
            m_full = train(make_model("sage_concat", 768, seed), [full[i] for i in tr], hp["epochs"], hp["lr"], seed)
            mlp = train(make_model("news_mlp", 768, seed), [full[i] for i in tr], hp["epochs"], hp["lr"], seed)
            for mode, k in SETTINGS:
                directed, g = trunc[(mode, k)]
                n_tr = np.array([d.num_nodes for d in directed])
                aff = te[n_tr[te] < n_full[te]] if mode != "full" else te
                subsets = {"all": te, "affected": aff}
                row = {"dataset": name, "seed": seed, "mode": mode, "k": k,
                       "n_test": int(len(te)), "n_affected": int(len(aff)),
                       "mean_nodes_test": float(n_tr[te].mean())}
                m_tr = m_full if mode == "full" else train(
                    make_model("sage_concat", 768, seed), [g[i] for i in tr], hp["epochs"], hp["lr"], seed)
                X = np.array([struct_feats(d) for d in directed])
                y = np.array([int(d.y) for d in directed])
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y[tr])
                for sname, idx in subsets.items():
                    if len(idx) == 0:
                        continue
                    tg = [g[i] for i in idx]
                    row[f"i_{sname}"] = evaluate(m_full, tg)
                    row[f"ii_{sname}"] = evaluate(m_tr, tg)
                    row[f"mlp_{sname}"] = evaluate(mlp, [full[i] for i in idx])
                    row[f"lrstruct_{sname}"] = sk_score(y[idx], lr.predict(X[idx]))
                res[name].append(row)
                print(name, seed, mode, k, {kk: round(v["acc"], 3) for kk, v in row.items() if isinstance(v, dict)}, flush=True)
            print(f"{name} seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    (ROOT / "artifacts" / "upfd_early.json").write_text(json.dumps(res, indent=1))
    write_table(res)


def write_table(res: dict) -> None:
    lines = []
    for name, rows in res.items():
        lines += [f"### {name}", "",
                  "| setting | mean nodes (test) | affected / test | (i) full→trunc acc | (ii) trunc→trunc acc | news-MLP acc | LR struct acc | (i) acc, affected | (ii) acc, affected | news-MLP acc, affected | LR struct acc, affected |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for mode, k in SETTINGS:
            rs = [r for r in rows if r["mode"] == mode and r["k"] == k]

            def ms(key):
                v = [r[key]["acc"] for r in rs if key in r]
                return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "—"
            label = "full" if mode == "full" else f"{mode} {'k' if mode == 'time' else 'd'}={k}"
            lines.append(f"| {label} | {np.mean([r['mean_nodes_test'] for r in rs]):.1f} | "
                         f"{np.mean([r['n_affected'] for r in rs]):.0f} / {rs[0]['n_test']} | "
                         f"{ms('i_all')} | {ms('ii_all')} | {ms('mlp_all')} | {ms('lrstruct_all')} | "
                         f"{ms('i_affected')} | {ms('ii_affected')} | {ms('mlp_affected')} | {ms('lrstruct_affected')} |")
        lines.append("")
    (ROOT / "artifacts" / "upfd_early_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
