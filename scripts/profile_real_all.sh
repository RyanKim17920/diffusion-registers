#!/bin/bash
# Profile released models with the same statistics used on our own runs.
# Model set is CURRENT as of 2026-09: the LLaDA2.x line superseded LLaDA-8B,
# which is kept only because the concurrent dLLM-registers work builds on it.
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
export HF_HOME=/data/huggingface
export PYTHONPATH="$REPO/src"
P="$REPO/.venv/bin/python $REPO/src/profile_real_models.py --n_seq 16 --batch 2 --seq_len 1024"

echo "### current diffusion LMs (masked input, as seen at inference)"
$P --model inclusionAI/LLaDA2.0-mini              --diffusion || echo "FAILED LLaDA2.0-mini"
$P --model inclusionAI/LLaDA2.1-mini              --diffusion || echo "FAILED LLaDA2.1-mini"
$P --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 --diffusion || echo "FAILED qwen3-mdlm"
$P --model Efficient-Large-Model/Fast_dLLM_v2_1.5B --diffusion || echo "FAILED fastdllm"
echo "### older dLLM, base of the concurrent registers work"
$P --model GSAI-ML/LLaDA-8B-Base                  --diffusion || echo "FAILED LLaDA-8B"
echo "### autoregressive control (clean input) + dLLM on clean input"
$P --model Qwen/Qwen2.5-7B                                    || echo "FAILED Qwen2.5"
$P --model GSAI-ML/LLaDA-8B-Base                              || echo "FAILED LLaDA-8B-clean"
echo PROFILE_REAL_DONE
