#!/bin/bash
#SBATCH --job-name=gen_text
#SBATCH --partition=cpu        # cluster-specific; edit or pass --partition to sbatch
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=runs/gen_text_%j.out
set -euo pipefail
REPO="${REG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
exec "$REPO/.venv/bin/python" "$REPO/src/text_data.py" \
  --out ${REG_TEXT_DATA:-$REPO/data/text} --num_proc 32
