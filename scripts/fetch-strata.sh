#!/usr/bin/env bash
# Pin the exact Strata source (tag v0.1.40.2) into ./strata-src for the Docker build.
# Fetched from the upstream repo and verified against the sha256 of this recipe's build date.
# Upstream: https://github.com/Niko1221/Strata — MIT licensed; see NOTICE.
set -euo pipefail
URL="https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.40.2.tar.gz"
SHA="80f32a37852401a5f2246a91047786d12f925f4c216e1c52d58c07fb038248aa"
[ -d strata-src ] && echo "strata-src exists — nothing to do" && exit 0
wget -q -O /tmp/strata.tar.gz "$URL"
echo "$SHA  /tmp/strata.tar.gz" | sha256sum -c -
mkdir -p strata-src && tar -xzf /tmp/strata.tar.gz -C strata-src --strip-components=1
echo "Strata v0.1.40.2 (upstream commit e8ca9afd) at ./strata-src"
