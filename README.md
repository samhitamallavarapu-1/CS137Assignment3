# CS137Assignment3

This repository is organized for a 3-part assignment with separate locations for:
- code per part
- generated outputs per part
- SLURM scripts and SLURM logs per part (for HPC runs)

## Assignment part mapping

- `part_1`: Baseline CNN-Transformer Patch Architecture
- `part_2`: Architecture Search & Beating the Baseline
- `part_3`: Model Diagnosis or Independent Study

## Directory layout

```text
CS137Assignment3/
├── code/
│   ├── part_1/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── crop_sweep.py
│   ├── part_2/
│   └── part_3/
├── outputs/
│   ├── part_1/
│   ├── part_2/
│   └── part_3/
├── slurm/
│   ├── part_1/
│   │   ├── scripts/
│   │   │   └── run_part1.slurm
│   │   └── logs/
│   ├── part_2/
│   │   ├── scripts/
│   │   └── logs/
│   └── part_3/
│       ├── scripts/
│       └── logs/
└── main.pdf
```

## Data paths

Training data for this assignment lives at:
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data`
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data`

The `evaluation/` folder is intentionally not used in current training code.

## Part 1 implementation status

Implemented files:
- `code/part_1/model.py`: baseline CNN-Transformer patch architecture with `get_model(...)`
- `code/part_1/train.py`: configurable training pipeline (CLI-driven, HPC-friendly)
- `code/part_1/crop_sweep.py`: visual utility for selecting weather crop bounds

Current `new_england` crop defaults in `train.py` correspond to approximately:
- raw map bounds: `y:105:386, x:180:374` for `450x449` weather tensors
- ratio form (used in code): `y0=0.233h, y1=0.858h, x0=0.401w, x1=0.833w`

## Part 1 training behavior

`train.py` includes:
- year-scoped training via `--years` (example: `2019-2023`)
- random train/validation split over valid anchor windows
- per-channel weather normalization (train split only)
- per-zone energy normalization (train split only)
- crop + downsample preprocessing before model input
- checkpointing to `best.pt` and `last.pt`
- metrics logging (`metrics.csv`) and saved run config (`config.json`)

Normalization artifacts are saved in each run folder:
- `normalization.json` (weather mean/std, energy mean/std, crop settings, downsample size)

## Example Part 1 run

```bash
python code/part_1/train.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --output-dir outputs/part_1 \
  --run-name exp_baseline \
  --years 2019-2023 \
  --epochs 20 \
  --batch-size 2
```

## Running Part 1 on SLURM

Use the provided script:
- `slurm/part_1/scripts/run_part1.slurm`

Submit:

```bash
sbatch slurm/part_1/scripts/run_part1.slurm
```

The script uses the same environment as Assignment 2:
- Python executable: `$HOME/.conda/envs/cs137-cnn/bin/python`

Logs go to:
- `slurm/part_1/logs/%x_%j.out`
- `slurm/part_1/logs/%x_%j.err`

## Crop sweep utility

To generate candidate crop overlays and downsample previews:

```bash
python code/part_1/crop_sweep.py \
  --weather-pt /cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data/2019/X_2019010100.pt \
  --output-dir outputs/part_1/crop_sweep
```

This writes:
- candidate overlay image
- downsampled candidate comparison image
- CSV mapping oriented coordinates to raw `train.py` coordinates
