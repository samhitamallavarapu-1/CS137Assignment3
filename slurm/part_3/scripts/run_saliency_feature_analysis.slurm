#!/bin/bash
#SBATCH --job-name=cs137a3_p3_sal
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --output=slurm/part_3/logs/cs137a3_p3_sal_%j.out
#SBATCH --error=slurm/part_3/logs/cs137a3_p3_sal_%j.err

set -euo pipefail

RUN_DIR="outputs/part_1/part1_first_improved_3seed_s40"
PART1_MODEL_PATH="code/part_1/model.py"
CKPT="best.pt"
EVAL_YEAR=2022
N_DAYS=10
DEVICE="cpu"
ZONES="ME,NH,VT,CT,RI,SEMA,WCMA,NEMA_BOST"
CLEAN_OUTPUT=1

PROJECT_ROOT="/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment3"
cd "$PROJECT_ROOT"

set +u
source ~/.bashrc
set -u
conda activate cs137-cnn

mkdir -p slurm/part_3/logs

echo "[$(date)] Starting saliency + feature importance analysis"
echo "RUN_DIR=${RUN_DIR} CKPT=${CKPT} EVAL_YEAR=${EVAL_YEAR} N_DAYS=${N_DAYS} DEVICE=${DEVICE}"
echo "ZONES=${ZONES} CLEAN_OUTPUT=${CLEAN_OUTPUT}"

CMD=(
  python code/part_3/saliency_feature_analysis.py
  --run-dir "${RUN_DIR}"
  --part1-model-path "${PART1_MODEL_PATH}"
  --ckpt "${CKPT}"
  --eval-year "${EVAL_YEAR}"
  --n-days "${N_DAYS}"
  --device "${DEVICE}"
  --zones "${ZONES}"
)

if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
  CMD+=(--clean-output)
fi

"${CMD[@]}"

echo "[$(date)] Finished saliency + feature importance analysis"
