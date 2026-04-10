#!/bin/bash
#SBATCH --job-name=cs137a3_part1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/part_1/logs/%x_%j.out
#SBATCH --error=slurm/part_1/logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/part_1/logs outputs/part_1

PYTHON="$HOME/.conda/envs/cs137-cnn/bin/python"

# Optional overrides at submit-time, e.g.:
#   sbatch --export=ALL,RUN_NAME=part1_lr1e4,LR=1e-4,BATCH_SIZE=1 slurm/part_1/scripts/run_part1.slurm
RUN_NAME="${RUN_NAME:-part1_${SLURM_JOB_ID}}"
YEARS="${YEARS:-2019-2023}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS="${LOSS:-mse}"
D_MODEL="${D_MODEL:-256}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
FF_DIM="${FF_DIM:-1024}"
DROPOUT="${DROPOUT:-0.1}"
CNN_HIDDEN_DIM="${CNN_HIDDEN_DIM:-64}"
PATCH_GRID_H="${PATCH_GRID_H:-10}"
PATCH_GRID_W="${PATCH_GRID_W:-10}"
HISTORY_LEN="${HISTORY_LEN:-168}"
FUTURE_LEN="${FUTURE_LEN:-24}"
VAL_RATIO="${VAL_RATIO:-0.15}"
DOWNSAMPLE_H="${DOWNSAMPLE_H:-96}"
DOWNSAMPLE_W="${DOWNSAMPLE_W:-96}"
CROP_MODE="${CROP_MODE:-new_england}"
NORM_MAX_TIMESTAMPS="${NORM_MAX_TIMESTAMPS:-2000}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"


echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Using Python: $PYTHON"
echo "Run name: $RUN_NAME"

$PYTHON --version
nvidia-smi
$PYTHON -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

$PYTHON code/part_1/train.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --output-dir outputs/part_1 \
  --run-name "$RUN_NAME" \
  --years "$YEARS" \
  --history-len "$HISTORY_LEN" \
  --future-len "$FUTURE_LEN" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --crop-mode "$CROP_MODE" \
  --downsample-h "$DOWNSAMPLE_H" \
  --downsample-w "$DOWNSAMPLE_W" \
  --norm-max-timestamps "$NORM_MAX_TIMESTAMPS" \
  --d-model "$D_MODEL" \
  --num-heads "$NUM_HEADS" \
  --num-layers "$NUM_LAYERS" \
  --ff-dim "$FF_DIM" \
  --dropout "$DROPOUT" \
  --cnn-hidden-dim "$CNN_HIDDEN_DIM" \
  --patch-grid-h "$PATCH_GRID_H" \
  --patch-grid-w "$PATCH_GRID_W" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --loss "$LOSS" \
  --num-workers "$NUM_WORKERS" \
  --device cuda

echo "Finished at $(date)"
