#!/bin/bash
# Profile released models with the same statistics used on our own runs.
#
# Released models CANNOT test whether registers reduce outliers -- that needs
# retraining, and lives in our from-scratch ladder. What they can establish is
# (a) that the pathology exists in deployed dLLMs at all, and (b) whether it
# looks different by TRAINING LINEAGE, which is the interesting axis:
#
#   from-scratch bidirectional diffusion : no position is privileged, so
#       outliers should be content-dependent (high position entropy)
#   adapted from an AR checkpoint        : may INHERIT the autoregressive
#       attention sink at position 0 (low entropy, mass on pos 0)
#
# If that split holds, the AR sink fix ("keep token 0") transfers only to the
# adapted models, and from-scratch dLLMs need something else.
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
export HF_HOME=/data/huggingface
export PYTHONPATH="$REPO/src"
P="$REPO/.venv/bin/python $REPO/src/profile_real_models.py --n_seq 16 --batch 2 --seq_len 1024"

echo "### LINEAGE A: from-scratch bidirectional diffusion (closest to ours)"
$P --model kuleshov-group/mdlm-owt                 --diffusion || echo "FAILED mdlm-owt"
$P --model GSAI-ML/LLaDA-8B-Base                   --diffusion || echo "FAILED LLaDA-8B"

echo "### LINEAGE B: adapted from an autoregressive checkpoint"
$P --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 --diffusion || echo "FAILED qwen3-mdlm"
$P --model Efficient-Large-Model/Fast_dLLM_v2_1.5B --diffusion || echo "FAILED fastdllm"
$P --model Dream-org/Dream-v0-Base-7B              --diffusion || echo "FAILED dream"

echo "### LINEAGE C: current LLaDA2.x line (MoE; recipe differs from all of the above)"
$P --model inclusionAI/LLaDA2.0-mini               --diffusion || echo "FAILED LLaDA2.0-mini"

echo "### AR reference point (clean input), and a dLLM on clean input"
echo "### -- separates the mask tokens from the model itself"
$P --model Qwen/Qwen2.5-7B                                     || echo "FAILED Qwen2.5"
$P --model GSAI-ML/LLaDA-8B-Base                               || echo "FAILED LLaDA-8B-clean"
echo PROFILE_REAL_DONE
