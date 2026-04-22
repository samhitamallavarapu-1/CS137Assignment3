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
# sbatch --export=ALL,RUN_NAME_PREFIX=part2_first_improved_3seed,EPOCHS=20,LOSS=mae,SPLIT_MODE=blocked,BLOCKED_VAL_POSITION=tail,PURGE_GAP=-1,SCHEDULER=cosine,WARMUP_EPOCHS=2,MIN_LR=1e-6,AMP=1,RESUME_IF_EXISTS=1 slurm/part_2/scripts/run_part2.slurm

PYTHON="$HOME/.conda/envs/cs137-cnn/bin/python"
TRAIN_PY="$PROJECT_ROOT/code/part_2/train.py"

# Environment variable overrides with defaults
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-part2}"
YEARS="${YEARS:-2019-2023}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
SCHEDULER="${SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS="${LOSS:-mse}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
D_MODEL="${D_MODEL:-256}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DECODER_LAYERS="${DECODER_LAYERS:-3}"
SPATIAL_LAYERS="${SPATIAL_LAYERS:-2}"
FF_DIM="${FF_DIM:-1024}"
DROPOUT="${DROPOUT:-0.1}"
CNN_HIDDEN_DIM="${CNN_HIDDEN_DIM:-64}"
PATCH_GRID_H="${PATCH_GRID_H:-10}"
PATCH_GRID_W="${PATCH_GRID_W:-10}"
ARCH_VARIANT="${ARCH_VARIANT:-no_cnn}"
RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-3}"
HISTORY_LEN="${HISTORY_LEN:-168}"
FUTURE_LEN="${FUTURE_LEN:-24}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SPLIT_MODE="${SPLIT_MODE:-blocked}"
BLOCKED_VAL_POSITION="${BLOCKED_VAL_POSITION:-tail}"
VAL_YEARS="${VAL_YEARS:-}"
PURGE_GAP="${PURGE_GAP:--1}"
DOWNSAMPLE_H="${DOWNSAMPLE_H:-96}"
DOWNSAMPLE_W="${DOWNSAMPLE_W:-96}"
CROP_MODE="${CROP_MODE:-new_england}"
NORM_MAX_TIMESTAMPS="${NORM_MAX_TIMESTAMPS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
AMP="${AMP:-1}"
RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-1}"

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
  --years "$YEARS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --history-len "$HISTORY_LEN"
  --future-len "$FUTURE_LEN"
  --val-ratio "$VAL_RATIO"
  --split-mode "$SPLIT_MODE"
  --blocked-val-position "$BLOCKED_VAL_POSITION"
  --purge-gap "$PURGE_GAP"
  --crop-mode "$CROP_MODE"
  --downsample-h "$DOWNSAMPLE_H"
  --downsample-w "$DOWNSAMPLE_W"
  --norm-max-timestamps "$NORM_MAX_TIMESTAMPS"
  --scheduler "$SCHEDULER"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --warmup-epochs "$WARMUP_EPOCHS"
  --weight-decay "$WEIGHT_DECAY"
  --loss "$LOSS"
  --huber-delta "$HUBER_DELTA"
  --d-model "$D_MODEL"
  --num-heads "$NUM_HEADS"
  --num-layers "$NUM_LAYERS"
  --decoder-layers "$DECODER_LAYERS"
  --spatial-layers "$SPATIAL_LAYERS"
  --ff-dim "$FF_DIM"
  --dropout "$DROPOUT"
  --cnn-hidden-dim "$CNN_HIDDEN_DIM"
  --patch-grid-h "$PATCH_GRID_H"
  --patch-grid-w "$PATCH_GRID_W"
  --residual-blocks "$RESIDUAL_BLOCKS"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --device cuda
)

if [[ -n "$VAL_YEARS" ]]; then
  COMMON_ARGS+=(--val-years "$VAL_YEARS")
fi

if [[ "$AMP" == "1" ]]; then
  COMMON_ARGS+=(--amp)
else
  COMMON_ARGS+=(--no-amp)
fi

if [[ "$RESUME_IF_EXISTS" == "1" ]]; then
  COMMON_ARGS+=(--resume-from auto)
fi

# Variation 1: remove CNN layer (direct pooled weather tokens)
$PYTHON "$TRAIN_PY" \
  "${COMMON_ARGS[@]}" \
  --run-name ${RUN_NAME_PREFIX}_no_cnn \
  --arch-variant no_cnn

# Variation 2: residual CNN backbone before hierarchical attention
$PYTHON "$TRAIN_PY" \
  "${COMMON_ARGS[@]}" \
  --run-name ${RUN_NAME_PREFIX}_residual_cnn \
  --arch-variant residual_cnn

echo "Finished at $(date)"
