"""Check 3: reproduce UPFD (GraphSAGE + BERT + news concat) on PolitiFact and GossipCop, 5 seeds.

Rows:
  sage_concat   / random  official protocol: random 20/10/70 split per seed (gnn.py uses random_split)
  sage_concat   / fixed   PyG's fixed train/val/test_idx.npy split, only init/shuffle vary
  sage_noconcat / random  ablation: no news concat (explains part of the v1 smoke-test gap)
  news_mlp      / random  news-text-only baseline (root BERT embedding -> MLP)
Writes artifacts/upfd_repro.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from upfd_common import (HPARAMS, ROOT, apply, evaluate, fixed_split_idx, load_graphs, make_model,
                         official_transform, random_split_idx, train)

SEEDS = range(5)
RUNS = [("sage_concat", "random"), ("sage_concat", "fixed"), ("sage_noconcat", "random"), ("news_mlp", "random")]


def main() -> None:
    torch.set_num_threads(8)
    results = {}
    for name in ("politifact", "gossipcop"):
        graphs = apply(load_graphs(name), official_transform())
        hp = HPARAMS[name]
        for kind, split in RUNS:
            rows = []
            for seed in SEEDS:
                tr, _, te = random_split_idx(len(graphs), seed) if split == "random" else fixed_split_idx(name)
                t0 = time.time()
                m = train(make_model(kind, graphs[0].num_features, seed), [graphs[i] for i in tr],
                          hp["epochs"], hp["lr"], seed)
                r = evaluate(m, [graphs[i] for i in te])
                rows.append(r)
                print(f"{name} {kind} {split} seed={seed} acc={r['acc']:.4f} f1={r['f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            results[f"{name}/{kind}/{split}"] = {
                k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in ("acc", "f1")
            } | {"runs": rows, "epochs": hp["epochs"], "lr": hp["lr"]}
    (ROOT / "artifacts" / "upfd_repro.json").write_text(json.dumps(results, indent=1))
    for k, v in results.items():
        print(f"{k:40s} acc {v['acc'][0]:.4f} ± {v['acc'][1]:.4f}   f1 {v['f1'][0]:.4f} ± {v['f1'][1]:.4f}")


if __name__ == "__main__":
    main()
