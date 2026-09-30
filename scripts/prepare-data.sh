#!/usr/bin/env bash
# Turn fetched GGUF shards into the two things Strata's engine needs at runtime:
#   packs/qwen3.8-flash-next/   (dense.bin + index + native_experts + tokenizer)
#   mtp/rt/                     (MTP draft layer, ~5 GB range-fetched from the ORIGINAL Qwen checkpoint,
#                                packed to q2_0, then converted to the engine's runtime format)
# Every step runs inside the strata image with --network none except the MTP range-fetch, which only
# needs read access to huggingface.co. Nothing writes outside the data dir you pass.
#
# usage: ./scripts/prepare-data.sh <models-dir> <data-dir>
#   models-dir holds the two verified IQ3_XXS shards (see fetch-model.sh)
set -euo pipefail
MODELS="$(realpath "${1:?models dir}")"
DATA="$(realpath "${2:?data dir}")"
IMAGE="${IMAGE:-strata:0.1.30}"
S1="$MODELS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
mkdir -p "$DATA/packs" "$DATA/mtp"

echo '== pack (experts read from GGUF at start; dense.bin + index + tokenizer) =='
docker run --rm --network none \
  -v "$MODELS":/models:ro -v "$DATA/packs":/packs --entrypoint sh "$IMAGE" -c '
PYTHONPATH=/app/gguf-py python3 /app/strata/tools/iq_pack.py --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out /packs/qwen3.8-flash-next &&
PYTHONPATH=/app/gguf-py python3 /app/strata/tools/strata_tokenizer.py --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out /packs/qwen3.8-flash-next'

echo '== MTP draft layer: range-fetch from the original BF16 checkpoint (the only step needing HF) =='
docker run --rm \
  -v "$DATA/mtp":/mtp --entrypoint sh "$IMAGE" -c \
  'python3 /app/strata/tools/mtp_fetch.py fetch --out /mtp'

echo '== MTP: pack to q2_0 + runtime convert + draft vocab subset =='
docker run --rm --network none \
  -v "$DATA/mtp":/mtp --entrypoint sh "$IMAGE" -c '
export STRATA_GGUF_PY=/app/gguf-py
python3 /app/strata/tools/mtp_pack.py --src /mtp --experts q2_0 --out /mtp/mtp-q2_0.gguf &&
python3 /app/strata/tools/mtp_rt.py --gguf /mtp/mtp-q2_0.gguf --out /mtp/rt &&
cp /app/strata/data/draft_vocab.bin /mtp/rt/draft_vocab.bin'

echo "data ready: $DATA/packs/qwen3.8-flash-next and $DATA/mtp/rt"
