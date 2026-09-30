#!/usr/bin/env bash
# Pin the exact Strata source (tag v0.1.27) into ./strata-src for the Docker build.
# Fetched from the upstream repo and verified against the sha256 of this recipe's build date.
# Upstream: https://github.com/Niko1221/Strata — MIT licensed; see NOTICE.
set -euo pipefail
URL="https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.27.tar.gz"
SHA="313e4c0e8888868672a8b3b693a0afd1673d5e6e0e1056193c6d9d251588dbf4"
[ -d strata-src ] && echo "strata-src exists — nothing to do" && exit 0
wget -q -O /tmp/strata.tar.gz "$URL"
echo "$SHA  /tmp/strata.tar.gz" | sha256sum -c -
mkdir -p strata-src && tar -xzf /tmp/strata.tar.gz -C strata-src --strip-components=1
echo "Strata v0.1.27 (upstream commit a790805) at ./strata-src"
