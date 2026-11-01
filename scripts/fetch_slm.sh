#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/models"
cd "$repo_root"
uvx --from huggingface-hub hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir ./models
test -s models/Qwen3-4B-Q4_K_M.gguf
echo "Local SLM model ready: models/Qwen3-4B-Q4_K_M.gguf"
