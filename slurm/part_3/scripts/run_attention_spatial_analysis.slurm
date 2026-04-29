#!/bin/bash
#SBATCH --job-name=cs137a3_p3_attn
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
#SBATCH --output=slurm/part_3/logs/cs137a3_p3_attn_%j.out
#SBATCH --error=slurm/part_3/logs/cs137a3_p3_attn_%j.err

set -euo pipefail

# =========================
# User-configurable inputs
# =========================
RUN_DIR="outputs/part_1/part1_first_improved_3seed_s40"
PART1_MODEL_PATH="code/part_1/model.py"
CKPT="best.pt"              # best.pt or last.pt
EVAL_YEAR=2022
N_DAYS=30
DEVICE="cpu"                # cpu or cuda
TOPK=8
ZONES="ME,NH,VT,CT,RI,SEMA,WCMA,NEMA_BOST"
CLEAN_OUTPUT=1

# =========================
# Environment setup
# =========================
PROJECT_ROOT="/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment3"
cd "$PROJECT_ROOT"

# Optional: activate your environment if needed
set +u
source ~/.bashrc
set -u
conda activate cs137-cnn
# Submit from project root:
# sbatch slurm/part_3/scripts/run_attention_spatial_analysis.slurm

mkdir -p slurm/part_3/logs

echo "[$(date)] Starting attention spatial analysis"
echo "RUN_DIR=${RUN_DIR}"
echo "CKPT=${CKPT} EVAL_YEAR=${EVAL_YEAR} N_DAYS=${N_DAYS} DEVICE=${DEVICE}"
echo "ZONES=${ZONES} CLEAN_OUTPUT=${CLEAN_OUTPUT}"

CMD=(
  python code/part_3/attention_spatial_analysis.py
  --run-dir "${RUN_DIR}" \
  --part1-model-path "${PART1_MODEL_PATH}" \
  --ckpt "${CKPT}" \
  --eval-year "${EVAL_YEAR}" \
  --n-days "${N_DAYS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}" \
  --zones "${ZONES}"
)

if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
  CMD+=(--clean-output)
fi

"${CMD[@]}"

echo "[$(date)] Finished attention spatial analysis"
