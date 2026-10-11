#!/usr/bin/env bash
# Pin the exact Strata source (tag v0.1.43) into ./strata-src for the Docker build.
# Fetched from the upstream repo and verified against the sha256 of this recipe's build date.
# Upstream: https://github.com/Niko1221/Strata — MIT licensed; see NOTICE.
set -euo pipefail
URL="https://github.com/Niko1221/Strata/archive/refs/tags/v0.1.43.tar.gz"
SHA="16ecfb44dbd322b94c3ff013c6cb54f23e1cd5bea63f318a7fcdad6a1c14b286"
[ -d strata-src ] && echo "strata-src exists — nothing to do" && exit 0
wget -q -O /tmp/strata.tar.gz "$URL"
echo "$SHA  /tmp/strata.tar.gz" | sha256sum -c -
mkdir -p strata-src && tar -xzf /tmp/strata.tar.gz -C strata-src --strip-components=1
echo "Strata v0.1.43 (upstream commit 16374b3e) at ./strata-src"
