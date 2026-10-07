# Strata running Qwen3.8-Flash-Next IQ3_S on 2x RTX 3090 — a containerized recipe

Runs [Strata](https://github.com/Niko1221/Strata) (v0.1.40.1, MIT) on the 125B MoE
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` GGUFs from [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
without ever executing Strata's own installer on your host: the engine is compiled inside Docker from
the checksum-pinned upstream tag, the model files are fetched with sha256 verification, and the runtime
container has **no network access at all** — which also means the engine can never self-update under you.

Why bother: Strata's installer (`setup.py`) is honest software — no hidden endpoints, no persistence, no
telemetry (I read it line by line before building this) — but it compiles on your host with `sudo`,
pulls a CUDA toolkit onto your system, re-downloads its binary from GitHub releases at every start,
and keeps its own venv and state outside your service manager. On a box that already serves other models,
that's pollution. This recipe keeps only what the installer actually does that matters — compile the
source, pack the GGUF, launch the same two processes — and gives it an image and a compose file.

## What you get
- OpenAI-compatible `POST /v1/chat/completions` and Anthropic-compatible `POST /v1/messages`,
  plus `GET /health`, `GET /metrics` (per-request reuse and hit-rate counters).
- The engine's own flags work as upstream documents them: prompt-checkpoint reuse, MTP draft layer
  (speculative decode), two-card layer split with a VRAM expert cache, **262,144-token context** —
  the IQ3_S shards' native trained window (`qwen4exp.context_length` in the GGUF metadata).

## Hardware floor (measured, this exact config)
- 2x RTX 3090 (48 GB VRAM): engine loads dense weights + KV + a 17,077-slot expert cache (33,085 MiB
  across the pair); VRAM runs near-full (~460 MiB free — keep your other GPU services off).
- ~62 GB system RAM: the 512-expert MoE arena (~48 GB, measured 47,962 MiB at load) is pinned; the rest
  streams from SSD. With a serving stack resident, budget nothing else: it fits only when the box is
  mostly free.
- ~85 GB free disk for shards + pack + MTP (shard 2 is byte-identical across all GSQ-RCO sizes — link
  it, don't refetch).
- NVIDIA driver ≥ 580 (CUDA 13 runtime).
- **Context ceiling:** 262,144 is hard. We probed above native: YaRN extrapolation steals VRAM from the
  GPU expert cache — slots collapse (~17k → 3,812 at 2M), decode craters (~80 → 40 t/s), and past ~3M
  the session allocation fails outright. Don't raise `--max-context`; the native window is the recipe.

## Steps
```bash
# 0. build the engine image (fetches llama.cpp 3cf03257, verifies sha256)
./scripts/fetch-strata.sh                       # repo @ tag v0.1.40.1 -> ./strata-src/
docker build -t strata:0.1.40.1 .               # ~15 min on a 12-core box

# 1. model weights (resumable, sha256-verified against HuggingFace LFS oids)
./scripts/fetch-model.sh ./models

# 2. pack + MTP draft layer — runs entirely inside the image
./scripts/prepare-data.sh ./models ./data

# 3. serve
cp config/iq3s.json.example config/iq3s.json    # set api_key, port as you like
docker compose up -d                            # healthcheck on /health
```

Clients (any OpenAI SDK): base URL `http://<host>:8080/v1`, model name `qwen3.8-flashnext-iq3s`.
Anthropic-style clients: `http://<host>:8080/v1/messages`.

## Benchmark 1: context ladder (`benchmark/context-ladder.png`)

![Strata IQ3_S context ladder — prefill, decode and TTFT across 8k to 257k depth](benchmark/context-ladder.png)

llama-benchy 0.4.0 · pp 4,096 / tg 512 · 3 runs per depth · unique requests (`--no-cache`) ·
depths 8,192 → 257,280 (the top rung is 98% of the 262,144 window — the rungs above 126k only exist
because IQ3_S serves native 262k) · tokenizer `Qwen/Qwen3.8-Flash-Next` ·
prefill rate = (depth + 4,096) ÷ full-prefill wall time · engine restarted into a clean session.

| depth | prefill t/s | decode t/s | peak decode | TTFT | E2E (est.) |
|---|---|---|---|---|---|
| 8,192 | 869 | 113.1 ± 3.6 | 113.7 | 14.1 s | 18.7 s |
| 32,768 | 1,310 | 110.8 ± 4.3 | 111.3 | 28.1 s | 32.8 s |
| 65,536 | 1,504 | 107.3 ± 2.4 | 108.0 | 46.3 s | 51.1 s |
| 126,208 | 1,551 | 102.7 ± 2.3 | 103.3 | 84.0 s | 89.0 s |
| 196,608 | 1,675 | 98.1 ± 4.5 | 98.7 | 119.8 s | 125.1 s |
| 257,280 | 1,671 | 98.8 ± 4.5 | 99.3 | 156.5 s | 161.6 s |

Raw benchy table: `context-ladder-0.1.40.1.md`. The first rungs' prefill is cold-window (CUDA graph
capture); steady prefill is ~1,300–1,675 t/s. The headline is the deep half of the table: decode holds
98–103 t/s from 126k to 257k — at 98% of native window the engine is still barely off its short-prompt
speed.

Against the IQ3_XXS predecessor (0.1.39, same protocol, 131k window): IQ3_S decode matches XXS within
0–6% at every shared depth, but IQ3_S pays the prefill tax on this P2P-less pair (~3.3 GB/s PCIe expert
stream): ~1,274–1,675 pp t/s vs XXS's 1,692–2,376, so E2E is +27–49% at matched depths. What the
trade buys: twice the window (XXS caps at 131k on this box), unchanged deep decode, and a smarter
3-bit point. Warm-agent metrics actually *improve* (replay TTFT 0.166 → 0.121 s, sustained prose
97.1 → 105.6 t/s) — see `benchmark/` for both sides' raw artifacts.

If you are upgrading from an older pinned build of this recipe (v0.1.30/v0.1.36/v0.1.39): the pack
format (`native experts v3`), the serve config keys, and the engine flags are unchanged — rebuild the
image, reuse `data/`. Only `--max-context` moves if you bump the quant.

## Benchmark 2: agent-shape results (`benchmark/`)

![Strata IQ3_S agent-shape benchmark — TTFT cliff, decode by phase, full results table](benchmark/strata-agentbench.png)

`agent-shape-card.html` is the editable source of the PNG above; `agent-bench-results.json` is the raw
per-request artifact behind it. Measured on v0.1.40.1 with server-side engine timings (not client
estimates), engine restarted into a clean session, D at the published temperatures (0.6 prose /
0.2 code), the numbers that matter for coding agents:

| scenario | result |
|---|---|
| cold 22,284-token prompt prefill | 19.0 s (1,274 t/s) |
| same prompt replayed (checkpoint warm) | **0.121 s TTFT** — only 5 new tokens recompute (45.8 ms prompt) |
| warm decode | 110–126 t/s (B replay mean ~118) |
| sustained 2,048-token decode | prose 105.6 t/s / code 107.5 t/s (temp-matched) |
| MTP accept | prose 69% / code 66% on sustained runs; session-wide 67.4% |
| 10-turn growing conversation | cold turns TTFT 2.0→2.7 s; replay pass 0.063→0.077 s/turn, decode ~106.6 t/s |
| session-wide | 80.2% of prompt tokens reused (110,734 of 138,030); 0 aborts |

The cliff is the same physics as every quant: first turn on a new 22k context costs ~19 s; every turn
that keeps its prefix costs ~0.12 s. Agents that reuse sessions feel this engine; agents that re-dump
context pay the IQ3_S byte tax on every one.

Also measured, not carded: a custom agentic coding suite (8 shell-tool tasks × 3 passes, native
function calling) at **22/24 = 91.7%**, and a Terminal-Bench 2.1 subset (8 expert-level tasks via the
official Harbor harness) at **0/8** — reported honestly: the agent loop runs to the full turn budget
without producing the required artifacts; zero infra errors, so it reads as task difficulty, not
plumbing. Raw: `benchmark/cd-matched-results.json`.

## Layout
```
Dockerfile                     two-stage build (devel->compile -> runtime)
scripts/fetch-strata.sh        pinned repo tarball (tag v0.1.40.1) + sha256
scripts/fetch-model.sh         IQ3_S shards + sha256 (shard 2 dedupe documented inline)
scripts/prepare-data.sh        pack + tokenizer + MTP range-fetch/pack/rt (inside the image)
config/iq3s.json.example       server.py config (exe/args/gpu list = layer split)
docker-compose.yml             service definition, no egress at runtime
benchmark/                     both cards (PNG + HTML source) + raw artifacts
NOTICE / LICENSE               provenance
```
`strata-src/`, `models/`, `data/`, `config/iq3s.json` are gitignored — everything is reproducible via the scripts.

## Provenance / licenses
- Strata engine + server: © Niko1221 and the Strata contributors, MIT — https://github.com/Niko1221/Strata @ `v0.1.40.1` (commit `82f46a8c`, tarball sha256 `45e6ec0f41d96c77…`).
- llama.cpp (ggml/gguf-py/mtmd) pinned at `3cf03257f219`, MIT (unchanged by Strata through v0.1.40.1; re-verified in v0.1.40.1's setup.py 2026-10-07).
- Model weights: Qwen team + GSQ-RCO quants by ISTA-DASLab — see the HF repo's terms; you fetch them yourself. IQ3_S shard1 `4c1eb2ceb4915e11…` (54,817,524,224 B), shard2 `316b46f3a2dbd68c…` (28,800,138,432 B) — both match the HF LFS oids.
- Scripts/Dockerfile/README in this repo: MIT.

Recipe and benchmark: @LeTechLead
