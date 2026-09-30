#!/usr/bin/env bash
# Pin the exact Strata source (tag v0.1.30) into ./strata-src for the Docker build.
# Fetched from the upstream repo and verified against the sha256 of this recipe's build date.
# Upstream: https://github.com/Niko1221/Strata — MIT licensed; see NOTICE.
set -euo pipefail
URL="https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.30.tar.gz"
SHA="6e78f4618416362d2a47ebb77461ddaabd20cd4696e2aa4ef3446c5b6c0dfce6"
[ -d strata-src ] && echo "strata-src exists — nothing to do" && exit 0
wget -q -O /tmp/strata.tar.gz "$URL"
echo "$SHA  /tmp/strata.tar.gz" | sha256sum -c -
mkdir -p strata-src && tar -xzf /tmp/strata.tar.gz -C strata-src --strip-components=1
echo "Strata v0.1.30 (upstream commit 30ec18e) at ./strata-src"
