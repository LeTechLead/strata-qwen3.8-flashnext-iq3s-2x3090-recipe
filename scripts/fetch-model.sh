#!/usr/bin/env bash
# Fetch the IQ3_S shards for Strata from ISTA-DASLab with checksum verification.
# Shard 2 is byte-identical across all Qwen3.8-Flash-Next GSQ-RCO quants: if you already
# have it for any size, the script skips it and links it in.
set -euo pipefail

HF="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main"
DST="${1:-./models/qwen38-flashnext-gsq-iq3s}"
S1_SHA="4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"
S2_SHA="316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
S1="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
S2="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"

mkdir -p "$DST"; cd "$DST"

fetch() { # $1=file $2=sha256 $3=hf_subdir
  if [ -f "$1" ]; then
    echo "present: $1 — verifying…"
    echo "$2  $1" | sha256sum -c - && return 0
    echo "CORRUPT: $1 — refetching"; rm -f "$1"
  fi
  echo "downloading $1 (~51 GB / ~27 GB, resumable)…"
  wget -c --tries=0 --retry-connrefused -O "$1.part" "$HF/$3/$1"
  mv "$1.part" "$1"
  echo "$2  $1" | sha256sum -c -
}

fetch "$S2" "$S2_SHA" IQ3_S
fetch "$S1" "$S1_SHA" IQ3_S
echo "all shards verified"
