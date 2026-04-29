# CS137Assignment3

Code and experiment outputs for CS137 Assignment 3:
1. Part 1 baseline forecasting model
2. Part 2 architecture variants
3. Part 3 post-hoc analysis (attention, saliency, feature importance)

## Repository layout

```text
CS137Assignment3/
├── code/
│   ├── part_1/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── multi_seed.py
│   │   ├── crop_sweep.py
│   │   ├── evaluate.py
│   │   └── analyze_attention.py
│   ├── part_2/
│   │   ├── model.py
│   │   └── train.py
│   └── part_3/
│       ├── attention_spatial_analysis.py
│       ├── saliency_feature_analysis.py
│       └── feature_importance_only.py
├── slurm/
│   ├── part_1/scripts/run_part1.slurm
│   ├── part_2/scripts/run_part2.slurm
│   └── part_3/scripts/
│       ├── run_attention_spatial_analysis.slurm
│       ├── run_saliency_feature_analysis.slurm
│       └── run_feature_importance_only.slurm
├── outputs/
├── main.pdf
└── README.md
```

## Data location

Training and analysis scripts expect:
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data`
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data`

Use parent directory:
- `--data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data`

## Environment

SLURM scripts use:
- `$HOME/.conda/envs/cs137-cnn/bin/python` (Part 1/2)
- `conda activate cs137-cnn` (Part 3 scripts)

## Part 1

Entry point:
- `code/part_1/train.py`

Typical local run:

```bash
python code/part_1/train.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --output-dir outputs/part_1 \
  --run-name part1_exp \
  --years 2019-2023 \
  --epochs 20 \
  --batch-size 16
```

SLURM run:

```bash
sbatch slurm/part_1/scripts/run_part1.slurm
```

Utilities:
- `code/part_1/multi_seed.py`: launches multiple seeds and aggregates
- `code/part_1/crop_sweep.py`: crop/downsample diagnostics
- `code/part_1/evaluate.py`: evaluation helper
- `code/part_1/analyze_attention.py`: additional analysis helper

## Part 2

Entry point:
- `code/part_2/train.py`

Architecture options:
- `--arch-variant no_cnn`
- `--arch-variant residual_cnn`

Typical local run:

```bash
python code/part_2/train.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --output-dir outputs/part_2 \
  --run-name part2_exp_no_cnn \
  --arch-variant no_cnn \
  --years 2019-2023 \
  --epochs 20 \
  --batch-size 16
```

SLURM run (executes both variants):

```bash
sbatch slurm/part_2/scripts/run_part2.slurm
```

## Part 3

Part 3 analyzes trained Part 1 checkpoints from `outputs/part_1/<run_name>`.

### 1) Attention spatial analysis

Script:
- `code/part_3/attention_spatial_analysis.py`

SLURM:

```bash
sbatch slurm/part_3/scripts/run_attention_spatial_analysis.slurm
```

### 2) Saliency + feature importance

Script:
- `code/part_3/saliency_feature_analysis.py`

SLURM:

```bash
sbatch slurm/part_3/scripts/run_saliency_feature_analysis.slurm
```

### 3) Feature importance only (batched, faster)

Script:
- `code/part_3/feature_importance_only.py`

SLURM:

```bash
sbatch slurm/part_3/scripts/run_feature_importance_only.slurm
```

## Output artifacts

Each Part 1/Part 2 training run directory includes:
- `config.json`
- `normalization.json`
- `metrics.csv`
- `train_indices.csv`
- `val_indices.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`

Part 3 scripts write analysis outputs under the selected Part 1 run directory and/or `outputs/part_3`, depending on the script.
