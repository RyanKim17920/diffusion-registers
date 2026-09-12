#!/bin/bash
# Pull the real-model comparison set. Network only, no GPU.
set -uo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
for m in GSAI-ML/LLaDA-8B-Base kuleshov-group/mdlm-owt inclusionAI/LLaDA2.0-mini dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 Efficient-Large-Model/Fast_dLLM_v2_1.5B Dream-org/Dream-v0-Base-7B Qwen/Qwen2.5-7B; do
  echo "=== fetching $m ==="
  "${REG_PYTHON:-python}" -c "
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
