#!/bin/bash
#SBATCH --job-name=onset
#SBATCH --partition=cpu        # cluster-specific; edit or pass --partition to sbatch
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=3:00:00
#SBATCH --output=runs/onset_%j.out
set -uo pipefail
REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" PYTHONPATH="$REPO/src" OMP_NUM_THREADS=64
P="$REPO/.venv/bin/python $REPO/src/profile_real_models.py --channels --n_seq 8 --batch 2 --seq_len 512 --dtype float32"
# Is our 4.13x channel max/median about SIZE or about TRAINING TOKENS?
# GPT-2 is the known-positive control: the massive-activations literature
# reports them in GPT-2 at 124M, which is the size of our largest rungs.
echo "### AR, well-trained, 124M -- known to show massive activations"
$P --model gpt2                                       || echo "FAILED gpt2"
echo "### AR, well-trained, 355M"
$P --model gpt2-medium                                || echo "FAILED gpt2-medium"
echo "### diffusion objective, well-trained lineage, 0.6B"
$P --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 --diffusion \
                                                      || echo "FAILED qwen3-diff"
echo ONSET_DIAG_DONE
