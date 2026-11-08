"""Plot accuracy vs k (time-ordered truncation) from artifacts/upfd_early.json -> artifacts/upfd_early_acc_vs_k.png."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from upfd_common import ROOT

# dataviz reference palette (light): categorical slots 1-3, text tokens, surface
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"ii_all": ("(ii) train & test truncated", "#2a78d6"),
          "i_all": ("(i) train full, test truncated", "#eb6834"),
          "lrstruct_all": ("LR on size / depth / root degree", "#1baf7a")}
KS = [1, 5, 10, 20, 50]


def main() -> None:
    res = json.loads((ROOT / "artifacts" / "upfd_early.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, name in zip(axes, ("politifact", "gossipcop")):
        rows = res[name]
        xs = KS + ["full"]
        xpos = np.arange(len(xs))

        def stat(key, mode, k):
            v = [r[key]["acc"] for r in rows if r["mode"] == mode and r["k"] == k and key in r]
            return np.mean(v), np.std(v)

        for key, (lab, col) in SERIES.items():
            pts = [stat(key, "time", k) for k in KS] + [stat(key, "full", None)]
            m, s = np.array(pts).T
            ax.fill_between(xpos, m - s, m + s, color=col, alpha=0.12, linewidth=0)
            ax.plot(xpos, m, color=col, linewidth=2, marker="o", markersize=5, label=lab)
        mlp = np.mean([r["mlp_all"]["acc"] for r in rows if r["mode"] == "full"])
        ax.axhline(mlp, color=INK2, linewidth=1.5, linestyle="--", label="news-only MLP (root BERT)")
        ax.axhline(0.5, color=GRID, linewidth=1, zorder=0)
        ax.set_xticks(xpos, [str(x) for x in xs])
        ax.set_xlabel("k = first k nodes by tweet time (root = 1)", color=INK2)
        ax.set_title({"politifact": "PolitiFact", "gossipcop": "GossipCop"}[name], color=INK, loc="left", fontsize=12)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0].set_ylabel("test accuracy (mean ± std, 3 seeds)", color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Early detection: UPFD (GraphSAGE + BERT + news concat) vs number of observed nodes",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = ROOT / "artifacts" / "upfd_early_acc_vs_k.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
