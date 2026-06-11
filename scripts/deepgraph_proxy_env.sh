#!/usr/bin/env bash

# Best-effort local proxy defaults for external research fetches.
# Keep localhost traffic direct so the processor can still poll the web API.
DEEPGRAPH_LOCAL_PROXY_URL="${DEEPGRAPH_LOCAL_PROXY_URL:-http://127.0.0.1:7890}"

if [[ "${DEEPGRAPH_DISABLE_LOCAL_PROXY_AUTODETECT:-0}" != "1" ]]; then
  export http_proxy="${http_proxy:-$DEEPGRAPH_LOCAL_PROXY_URL}"
  export https_proxy="${https_proxy:-$DEEPGRAPH_LOCAL_PROXY_URL}"
  export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
  export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
  export no_proxy="${no_proxy:-127.0.0.1,localhost,::1}"
  export NO_PROXY="${NO_PROXY:-$no_proxy}"
fi
