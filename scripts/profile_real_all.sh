#!/bin/bash
# Profile released models with the same statistics used on our own runs.
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
export HF_HOME=/data/huggingface
export PYTHONPATH="$REPO/src"
P="$REPO/.venv/bin/python $REPO/src/profile_real_models.py --n_seq 16 --batch 2 --seq_len 1024"
echo "### diffusion LMs (masked input, as they see at inference)"
$P --model kuleshov-group/mdlm-owt      --diffusion || echo "FAILED mdlm-owt"
$P --model GSAI-ML/LLaDA-8B-Base        --diffusion || echo "FAILED LLaDA"
echo "### autoregressive control (clean input)"
$P --model Qwen/Qwen2.5-7B                          || echo "FAILED Qwen"
echo "### LLaDA on clean input too, to separate the mask tokens from the model"
$P --model GSAI-ML/LLaDA-8B-Base                    || echo "FAILED LLaDA-clean"
echo PROFILE_REAL_DONE
