#!/bin/bash
#SBATCH --job-name=cs137a3_p3_fi_only
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/part_3/logs/cs137a3_p3_fi_only_%j.out
#SBATCH --error=slurm/part_3/logs/cs137a3_p3_fi_only_%j.err

set -euo pipefail

RUN_DIR="outputs/part_1/part1_first_improved_3seed_s40"
PART1_MODEL_PATH="code/part_1/model.py"
CKPT="best.pt"
EVAL_YEAR="all"
N_DAYS=0
DEVICE="cuda"
ZONES="ME,NH,VT,CT,RI,SEMA,WCMA,NEMA_BOST"
BATCH_ANCHORS=16
OUTPUT_ROOT="outputs/part_3"

PROJECT_ROOT="/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment3"
cd "$PROJECT_ROOT"

# sbatch slurm/part_3/scripts/run_feature_importance_only.slurm

set +u
source ~/.bashrc
set -u
conda activate cs137-cnn

mkdir -p slurm/part_3/logs

echo "[$(date)] Starting feature importance only analysis"
echo "RUN_DIR=${RUN_DIR} CKPT=${CKPT} EVAL_YEAR=${EVAL_YEAR} N_DAYS=${N_DAYS}"
echo "DEVICE=${DEVICE} BATCH_ANCHORS=${BATCH_ANCHORS} OUTPUT_ROOT=${OUTPUT_ROOT}"

python code/part_3/feature_importance_only.py \
  --run-dir "${RUN_DIR}" \
  --part1-model-path "${PART1_MODEL_PATH}" \
  --ckpt "${CKPT}" \
  --eval-year "${EVAL_YEAR}" \
  --n-days "${N_DAYS}" \
  --device "${DEVICE}" \
  --zones "${ZONES}" \
  --batch-anchors "${BATCH_ANCHORS}" \
  --output-root "${OUTPUT_ROOT}"

echo "[$(date)] Finished feature importance only analysis"
