#!/bin/bash
# Pull the real-model comparison set. Network only, no GPU.
set -uo pipefail
export HF_HOME=/data/huggingface
REPO=/admin/home/ryan.kim/registers
for m in kuleshov-group/mdlm-owt GSAI-ML/LLaDA-8B-Base Qwen/Qwen2.5-7B Dream-org/Dream-v0-Base-7B; do
  echo "=== fetching $m ==="
  "$REPO/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
import sys
try:
    p = snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.py'])
    print('OK $m ->', p)
except Exception as e:
    print('FAILED $m:', type(e).__name__, e)
" || echo "FETCH_FAILED $m"
done
echo FETCH_ALL_DONE
