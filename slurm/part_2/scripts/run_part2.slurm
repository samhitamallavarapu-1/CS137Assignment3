#!/bin/bash
#SBATCH --job-name=cs137a3_part2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment3/slurm/part_2/logs/%x_%j.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment3/slurm/part_2/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment3"
cd "$PROJECT_ROOT"

# Logs directory already exists in repository structure.

PYTHON="$HOME/.conda/envs/cs137-cnn/bin/python"
TRAIN_PY="$PROJECT_ROOT/code/part_2/train.py"

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Using Python: $PYTHON"
echo "Project root: $PROJECT_ROOT"

$PYTHON --version
nvidia-smi

$PYTHON -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

COMMON_ARGS=(
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data
  --output-dir outputs/part_2
  --epochs 80
  --batch-size 64
  --history-len 168
  --future-len 24
  --split-mode blocked
  --blocked-val-position tail
  --crop-mode new_england
  --downsample-h 96
  --downsample-w 96
  --scheduler cosine
)

# Variation 1: remove CNN layer (direct pooled weather tokens)
$PYTHON "$TRAIN_PY" \
  "${COMMON_ARGS[@]}" \
  --run-name part2_no_cnn_${SLURM_JOB_ID} \
  --arch-variant no_cnn \
  --num-layers 4 \
  --decoder-layers 3 \
  --spatial-layers 2 \
  --d-model 256 \
  --num-heads 8 \
  --ff-dim 1024

# Variation 2: residual CNN backbone before hierarchical attention
$PYTHON "$TRAIN_PY" \
  "${COMMON_ARGS[@]}" \
  --run-name part2_residual_cnn_${SLURM_JOB_ID} \
  --arch-variant residual_cnn \
  --residual-blocks 3 \
  --num-layers 4 \
  --decoder-layers 3 \
  --spatial-layers 2 \
  --d-model 256 \
  --num-heads 8 \
  --ff-dim 1024

echo "Finished at $(date)"
