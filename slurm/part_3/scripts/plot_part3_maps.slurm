#!/bin/bash
#SBATCH --job-name=cs137a3_part3_plot
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/part_3/logs/%x_%j.out
#SBATCH --error=slurm/part_3/logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/part_3/logs

PYTHON="$HOME/.conda/envs/cs137-cnn/bin/python"

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Using Python: $PYTHON"

$PYTHON --version

# Edit ATTN_NPZ to the Part 3 output you want to plot.
ATTN_NPZ="outputs/part_3/part1_first_improved_3seed_s40_val_20260427_153022/attention_maps.npz"

$PYTHON code/part_3/plot_maps.py \
  --attention-npz "$ATTN_NPZ" \
  --future-hours 0,6,12,18 \
  --zones all

echo "Finished at $(date)"
