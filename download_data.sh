#!/usr/bin/env bash
# Download the raw datasets into data/archives (checksummed) and unpack them to data/raw (read-only).
# Idempotent and resumable: re-run after an interrupted download.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/archives
META=https://raw.githubusercontent.com/safe-graph/GNN-FakeNews/798c290334b56195bfe686e76a3acc57899dc01e/data
GD=https://drive.usercontent.google.com/download  # same Google Drive files as torch_geometric.datasets.UPFD

fetch() {  # url file sha256
  local f=data/archives/$2
  if [ -f "$f" ] && echo "$3  $f" | sha256sum -c --status; then return; fi
  curl -fL -C - --retry 20 --retry-delay 5 -o "$f" "$1"
  echo "$3  $f" | sha256sum -c
}
fetch "https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1" rumdetect2017.zip 57059e5a0317280b84d60a189c8d8aa5a0beaf811679e2cbd1c97dee5d265156
fetch "$GD?id=1toou2GO0agoY_OS54LaCWEECQfe93nuq&export=download&confirm=t" politifact.zip eb6c348ce1fa6211b48b57c2b6798d800e21cf9e9dd43ab5b21a5eb099c53070
fetch "$GD?id=1DkMAzC7XUUciAxsSujRJt3sq1MqaVI3g&export=download&confirm=t" gossipcop.zip 886e0f00e3db483ff704f4d1fead88b042d390f15affa561e5cc6abd5c209aa1
# per-node timestamps and Twitter ids for UPFD (not part of the PyG download)
fetch "$META/pol_id_time_mapping.pkl"    pol_id_time_mapping.pkl    099ebae02456dca56c030ef9124eadb2e3771cb30edd144dff05902d329def5f
fetch "$META/pol_id_twitter_mapping.pkl" pol_id_twitter_mapping.pkl eaa91ddc97cf2f46941e59695696c8554fffcca3c307a5da6b99d72c48b14f03
fetch "$META/pol_news_list.txt"          pol_news_list.txt          d0f734ef6594f7cb2ba1560b4d16902bb8a4c6169ef4da5476eba7a91e2e2688
fetch "$META/gos_id_time_mapping.pkl"    gos_id_time_mapping.pkl    eba5b00ae2c5c5d8ec795f7bc5b72d08b66096036b22fb5bf34364be09d64c65
fetch "$META/gos_id_twitter_mapping.pkl" gos_id_twitter_mapping.pkl 984a4252a877bb2cf9591613d51871501d0c30196da19cce5327293f88e0fe0f
fetch "$META/gos_news_list.txt"          gos_news_list.txt          0a073de717b269659680c6a0d5369fa816ff61659d974c9dadde33347e4d40f2

if [ ! -d data/raw ]; then
  rm -rf data/raw.tmp && mkdir -p data/raw.tmp/upfd_meta data/raw.tmp/upfd
  unzip -q data/archives/rumdetect2017.zip -d data/raw.tmp
  unzip -q data/archives/politifact.zip -d data/raw.tmp/upfd/politifact
  unzip -q data/archives/gossipcop.zip -d data/raw.tmp/upfd/gossipcop
  cp data/archives/*_mapping.pkl data/archives/*_news_list.txt data/raw.tmp/upfd_meta/
  chmod -R a-w data/raw.tmp/*
  mv data/raw.tmp data/raw
fi
# layout expected by torch_geometric.datasets.UPFD and src/upfd_*.py (PyG writes its cache next to raw/)
for n in politifact gossipcop; do mkdir -p data/upfd/$n && ln -sfn ../../raw/upfd/$n data/upfd/$n/raw; done
ln -sfn raw/upfd_meta data/upfd_meta
echo "data/raw ready"
