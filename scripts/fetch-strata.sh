#!/usr/bin/env bash
# Pin the exact Strata source (tag v0.1.40.1) into ./strata-src for the Docker build.
# Fetched from the upstream repo and verified against the sha256 of this recipe's build date.
# Upstream: https://github.com/Niko1221/Strata — MIT licensed; see NOTICE.
set -euo pipefail
URL="https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.40.1.tar.gz"
SHA="45e6ec0f41d96c77fd43d10ad07998cd68b7515c39d5f8e4b5da013a209e0602"
[ -d strata-src ] && echo "strata-src exists — nothing to do" && exit 0
wget -q -O /tmp/strata.tar.gz "$URL"
echo "$SHA  /tmp/strata.tar.gz" | sha256sum -c -
mkdir -p strata-src && tar -xzf /tmp/strata.tar.gz -C strata-src --strip-components=1
echo "Strata v0.1.40.1 (upstream commit 82f46a8c) at ./strata-src"
