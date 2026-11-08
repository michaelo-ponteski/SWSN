# Early Fake News Detection

Time-aware graph vs. time-series representations of news propagation (Semantic Web and Social Networks, PUT).

## Setup

```bash
uv sync                          # Python 3.11 + dependencies from uv.lock
./download_data.sh               # ~1.6 GB, checksummed, resumable -> data/archives, data/raw (read-only)
uv run python src/build_data.py  # data/raw -> data/processed (~2 min)
```

## Data

| dataset | source | cascades | label |
|---|---|---|---|
| politifact, gossipcop | UPFD (Dou et al., SIGIR 2021), same files as `torch_geometric.datasets.UPFD`; per-node timestamps from [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) | 314, 5464 | real / fake |
| twitter15, twitter16 | Ma et al., ACL 2017 (`rumdetect2017.zip`) | 1490, 818 | non-rumor / false / true / unverified |

`data/processed/<dataset>/` (parquet):

- **nodes**: `graph_id, node_idx, parent_idx, depth, t_min, user_id` + raw fields
  (UPFD: `t_unix`, `feat_row` = row in `data/raw/upfd/<name>/new_*_feature.npz`; Twitter: `t_raw_min`, `tweet_id`, `t_clamped`).
  `node_idx` is time order (root = 0), `parent_idx < node_idx` and parent time <= child time, so the first *k* nodes
  or all nodes with `t_min <= h` always form a connected early-propagation tree.
- **graphs**: `graph_id, label, label_name, label_binary, n_nodes, max_depth, t_span_min` + per-dataset fields
  (UPFD: `news_id, pyg_split, t0_unix, dup_group`; Twitter: `text, text_norm, also_in_other_twitter, n_multi_parent`).
- **edges_raw** (Twitter): every raw edge, `in_tree`, `dropped_reason` - to try other parent rules without re-parsing.

Preprocessing decisions:

1. **Time origin**: Twitter delays are relative to the source tweet; UPFD news nodes have no timestamp, so t = 0 is the first engagement.
2. **Node identity**: UPFD node = one post (a user may repeat); Twitter node = one user (uid is unique per tree; the raw `tweet_id` field is not a node id).
3. **Twitter parents**: ~1.3 % of nodes have several candidate parents, which explains 98 % of time-inverted edges.
   We take the most recent earlier node (latest exposure); self-loops and edges into the root are dropped. Every tree ends up time-consistent.
4. **Posts before the source tweet** (736 nodes in 27 trees): `t_min` clamped to 0, raw value kept.
5. **Binary labels**: fake = `false`, real = `true` + `non-rumor`, `unverified` = NA (veracity unknown). UPFD: 1 = fake.
6. **Leakage**: 169 cascades are in both Twitter15 and Twitter16 (`also_in_other_twitter`, drop when pooling);
   repeated source texts share `text_norm`; UPFD has 4 + 35 groups of different news with identical cascades (`dup_group`).
   Use these as groups when splitting.

## Milestone 1 scripts

`src/upfd_*.py`, `src/smoke_test.py`, `src/plot_early.py` (outputs in `artifacts/`) read UPFD through PyG;
`download_data.sh` links `data/upfd/<name>/raw` to the raw files so nothing is downloaded twice.
