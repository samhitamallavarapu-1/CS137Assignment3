#!/bin/bash
#SBATCH --job-name=cs137a3_part3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
nvidia-smi

$PYTHON -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

# Edit PART1_RUN to whichever Part 1 experiment you want to diagnose.
PART1_RUN="outputs/part_1/part1_first_improved_3seed_s40"

$PYTHON code/part_3/run.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --part1-run-dir "$PART1_RUN" \
  --output-dir outputs/part_3 \
  --checkpoint best \
  --split val \
  --max-samples 256 \
  --batch-size 8 \
  --num-workers 2 \
  --compute-zone-grad-maps

echo "Finished at $(date)"
