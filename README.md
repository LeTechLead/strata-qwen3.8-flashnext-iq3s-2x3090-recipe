# Strata running Qwen3.8-Flash-Next IQ3_XXS on 2x RTX 3090 — a containerized recipe

Runs [Strata](https://github.com/Niko1221/Strata) (v0.1.30, MIT) on the 125B MoE
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` GGUFs from [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
without ever executing Strata's own installer on your host: the engine is compiled inside Docker from
the checksum-pinned upstream tag, the model files are fetched with sha256 verification, and the runtime
container has **no network access at all** — which also means the engine can never self-update under you.

Why bother: Strata's installer (`setup.py`) is honest software — no hidden endpoints, no persistence, no
telemetry (I read it line by line before building this) — but it compiles on your host with `sudo`,
pulls a CUDA toolkit onto your system, re-downloads its binary from GitHub releases at every start,
and keeps its own venv and state outside your service manager. On a box that already serves other models,
that's pollution. This recipe keeps only what the installer actually does that matters — compile the
source, pack the GGUFs, launch the same two processes — and gives it an image and a compose file.

## What you get
- OpenAI-compatible `POST /v1/chat/completions` and Anthropic-compatible `POST /v1/messages`,
  plus `GET /health`, `GET /metrics` (per-request reuse and hit-rate counters).
- The engine's own flags work as upstream documents them: prompt-checkpoint reuse, MTP draft layer
  (speculative decode), two-card layer split with a VRAM expert cache, 131,072-token context.

## Hardware floor (measured, this exact config)
- 2x RTX 3090 (48 GB VRAM): engine loads dense weights + KV + a 21,793-slot expert cache ≈ 35 GiB across the pair; VRAM runs near-full (~0.5 GB free — keep your other GPU services off).
- ~62 GB system RAM: the 512-expert MoE arena (~40 GB) is pinned; the 28.8 GB n-gram table streams from SSD. With a serving stack resident, budget nothing else: it fits only when the box is mostly free (Strata's own docs call IQ3_XXS a "64 GB PC" size).
- ~80 GB free disk for shards + pack + MTP (shard 2 is shared across all GSQ-RCO sizes — link it, don't refetch).
- NVIDIA driver ≥ 580 (CUDA 13 runtime).

## Steps
```bash
# 0. build the engine image (fetches llama.cpp 3cf03257, verifies sha256)
./scripts/fetch-strata.sh                       # repo @ tag v0.1.30 -> ./strata-src/
docker build -t strata:0.1.30 .                 # ~15 min on a 12-core box

# 1. model weights (resumable, sha256-verified against HuggingFace LFS oids)
./scripts/fetch-model.sh ./models

# 2. pack + MTP draft layer — runs entirely inside the image
./scripts/prepare-data.sh ./models ./data

# 3. serve
cp config/iq3xxs.json.example config/iq3xxs.json   # set api_key, port as you like
docker compose up -d                               # healthcheck on /health
```

Clients (any OpenAI SDK): base URL `http://<host>:8080/v1`, model name `qwen3.8-flash-next-iq3xxs`.
Anthropic-style clients: `http://<host>:8080/v1/messages`.

## Benchmark 1: context ladder (`benchmark/context-ladder.png`)

llama-benchy 0.3.5 · pp 4,096 / tg 512 · 3 runs per depth · unique requests (`--no-cache`) ·
depths 8,192 / 32,768 / 65,536 / 126,208 (near-max: 131,072 − 4,096 − 512 − margin) ·
tokenizer `Qwen/Qwen3.8-Flash-Next` · prefill rate = (depth + 4,096) ÷ full-prefill wall time.

| depth | prefill t/s | decode t/s | peak decode | TTFT | E2E (est.) |
|---|---|---|---|---|---|
| 8,192 | 1,084 ± 13 | 97.7 ± 6.7 | 133.7 | 11.34 s | 16.58 s |
| 32,768 | 1,643 ± 10 | 96.9 ± 2.5 | 118.3 | 22.44 s | 27.72 s |
| 65,536 | 1,909 ± 5 | 92.0 ± 2.2 | 113.7 | 36.47 s | 42.04 s |
| 126,208 | 1,996 ± 6 | 86.1 ± 4.0 | 108.0 | 65.28 s | 71.23 s |

Raw benchy tables for both engine versions measured on this box:
`context-ladder-0.1.30.md` (above) and `context-ladder-0.1.27.md` — identical protocol, same pack.
0.1.27 was flat ~660–678 t/s at every depth; 0.1.30's 1024-token expert-streaming prefill made deep
prefill 2.4–3x faster (TTFT at 126k: 192 s → 65 s) and decode +2–10%. If you are upgrading from a
0.1.27 pack: the pack format (`native experts v3`), the serve config keys, and the engine flags below
are unchanged — rebuild the image, reuse `data/`.

## Benchmark 2: agent-shape results (`benchmark/`)

![Strata IQ3_XXS agent-shape benchmark — TTFT cliff, decode by phase, full results table](benchmark/strata-agentbench.png)

`agent-shape-card.html` is the editable source of the PNG above; raw per-request JSON is the artifact behind it.
Measured on v0.1.30 with server-side engine timings (not client estimates), the numbers that matter for coding agents:

| scenario | result |
|---|---|
| cold 25,078-token prompt prefill | 16.5 s (~1,521 t/s) |
| same prompt replayed (checkpoint warm) | **0.13 s TTFT** — only 5 new tokens recompute |
| warm decode | 94–99 t/s; sustained prose up to 110.7 t/s |
| MTP accept | prose 80.8% / code 73.8% (drafts up to 4) |
| 10-turn growing conversation | every turn reused all-but-5 tokens; TTFT 1.35→1.65 s |
| session-wide | 88% of prompt tokens reused; expert-cache hit 99.8% |

The cliff is still real but much shallower than v0.1.27: first turn on a new 25k context costs ~17 s
(was ~40 s); every turn that keeps its prefix costs ~0.13 s. Agents that reuse sessions feel this
engine; agents that re-dump context pay for it every time.

## Honest caveats (from reading the source, not the marketing)
- **Self-update by design:** with network access, `setup.py` re-downloads the release binary at every
  start if its version gate wants newer. The runtime container here has no egress, so the binary can
  never change after build. Keep it that way unless you trust the upstream release stream.
- **No checksums on model downloads** in their installer; `fetch-model.sh` adds verification against HF's
  published LFS sha256s.
- **Single sequence at a time** (FIFO). Parallel sub-agents queue, including against chat use.
- **Pinned arena = one big load-time stall:** first load locks ~40 GB into RAM (~17 s at 3.9 GiB/s;
  theirs, not ours — the runs survived it fine).
- IQ3_XXS at 262k context is **not** supported by this config (their own installer caps < 90 GB boxes at 128k;
  deep-context prefill gets slower still and the KV is int8 here).
- The engine is one week old as of writing and essentially one maintainer plus PRs. The MIT license and the
  audited code are what make a pinned fork-in-a-container sane; none of it makes it a drop-in for vLLM.

## Layout
```
Dockerfile                     two-stage build (devel->compile -> runtime)
scripts/fetch-strata.sh        pinned repo tarball (tag v0.1.30) + sha256
scripts/fetch-model.sh         IQ3_XXS shards + sha256 (shard 2 dedupe documented inline)
scripts/prepare-data.sh        pack + tokenizer + MTP range-fetch/pack/rt (inside the image)
config/iq3xxs.json.example     server.py config (exe/args/gpu list = layer split)
docker-compose.yml             service definition, no egress at runtime
benchmark/                     both cards (PNG + HTML source) + raw artifacts
NOTICE / LICENSE               provenance
```
`strata-src/`, `models/`, `data/`, `config/iq3xxs.json` are gitignored — everything is reproducible via the scripts.

## Provenance / licenses
- Strata engine + server: © Niko1221 and contributors, MIT — https://github.com/Niko1221/Strata @ `v0.1.30` (commit `30ec18e`).
- llama.cpp (ggml/gguf-py/mtmd) pinned at `3cf03257f219`, MIT (unchanged by Strata v0.1.28–v0.1.30).
- Model weights: Qwen team + GSQ-RCO quants by ISTA-DASLab — see the HF repo's terms; you fetch them yourself.
- Scripts/Dockerfile/README in this repo: MIT.

Recipe and benchmark: @LeTechLead
