#!/bin/bash
#SBATCH --job-name=onset
#SBATCH --partition=c
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/ryan.kim/registers_runs/onset_%j.out
set -uo pipefail
REPO=/admin/home/ryan.kim/registers
export HF_HOME=/data/huggingface PYTHONPATH="$REPO/src" OMP_NUM_THREADS=64
# Is our lack of massive outliers about SIZE or about TRAINING TOKENS?
# mdlm-owt is ~170M (comparable to our rungs) but properly trained on OWT.
"$REPO/.venv/bin/python" "$REPO/src/profile_real_models.py" \
  --model kuleshov-group/mdlm-owt --diffusion --channels \
  --n_seq 8 --batch 2 --seq_len 512 --dtype float32 || echo "FAILED mdlm-owt"
echo ONSET_DIAG_DONE
