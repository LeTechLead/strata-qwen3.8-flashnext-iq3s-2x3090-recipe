# Strata engine 0.1.27 (https://github.com/Niko1221/Strata, MIT — see NOTICE), source-built, no installer.
# Strata's own setup.py never ships a prebuilt Linux engine — it compiles the checked-in source against
# llama.cpp pinned at 3cf03257f219 (for ggml, gguf-py and mtmd). This file does the same, but with
# checksum-verified downloads at build time and NO network access in the runtime image: the engine can
# never self-update after build.
#
# Prerequisite (fetch once before building):
#   ./scripts/fetch-strata.sh          # repo @ tag v0.1.27 into ./strata-src/
#
# Build:
#   docker build -t strata:0.1.27 .
# (set --build-arg CUDA_ARCHS=120 etc. for other cards; 86 = RTX 3090)

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS builder
ARG LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
# sha256 of https://github.com/ggml-org/llama.cpp/archive/<LLAMA_COMMIT>.zip
ARG LLAMA_ZIP_SHA=cbe23c594282ead2937abb3f008e51fcec4609d9256652fff42c7cc1c21ea47b
ARG CUDA_ARCHS=86
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build python3 unzip ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*
COPY strata-src /src/strata
RUN wget -q -O /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.zip" \
    && echo "${LLAMA_ZIP_SHA}  /tmp/llama.zip" | sha256sum -c - \
    && unzip -q /tmp/llama.zip -d /tmp/ll && mv /tmp/ll/llama.cpp-* /src/llama.cpp && rm /tmp/llama.zip
WORKDIR /src/strata
# Same flags Strata's setup.py passes on its Linux CUDA path (STRATA_PREFILL_MMQ is HIP-only: do not set it)
RUN cmake -B build -G Ninja \
      -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DSTRATA_GGML_DIR=/src/llama.cpp \
    && cmake --build build --target strata -j"$(nproc)"
# optional image encoder (tools/vision); skip these two lines if you don't need vision input
RUN cmake -B build-vision /src/strata/tools/vision -G Ninja \
      -DLLAMA_DIR=/src/llama.cpp \
      -DSTRATA_VISION_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build-vision --target strata-vision -j"$(nproc)"

FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages regex jinja2 pillow psutil numpy pyyaml
WORKDIR /app
COPY strata-src /app/strata
COPY --from=builder /src/strata/build/strata /app/engine/strata
COPY --from=builder /src/strata/build-vision/bin/strata-vision* /app/engine/
# gguf-py (pure python) so the pack tools can run inside this image
COPY --from=builder /src/llama.cpp/gguf-py /app/gguf-py
# stamp the engine as source-built (the format Strata's own setup writes; server.py reads it for version)
RUN printf '{\n "source": "local",\n "version": "0.1.27",\n "archs": [86],\n "vision": "gpu"\n}\n' > /app/engine/BUILD.json
EXPOSE 8080
