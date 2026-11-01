#!/usr/bin/env bash
set -euo pipefail

# First call downloads the public Hugging Face model weights into Docker's model_cache volume.
curl -fsS --max-time 3600 http://127.0.0.1:8081/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["South Asia GDP growth forecast"],"is_query":true}' \
  | jq -e '.vectors[0] | length == 1024' >/dev/null

curl -fsS --max-time 3600 http://127.0.0.1:8081/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"South Asia GDP growth","texts":["South Asia growth is 6.3 percent in 2026.","This document discusses weather."]}' \
  | jq -e '.scores | length == 2' >/dev/null

echo 'Embedding and reranker models are loaded.'
