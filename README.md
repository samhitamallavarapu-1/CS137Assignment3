# CS137 Assignment 3 Codebase

This repository contains all code, SLURM launch scripts, and saved outputs for:
1. Part 1 baseline CNN-Transformer demand forecasting
2. Part 2 architecture variants (`no_cnn`, `residual_cnn`)
3. Part 3 post-hoc interpretability analysis (attention, saliency, feature importance)

## Top-level structure

```text
CS137Assignment3/
├── code/
│   ├── part_1/                      # baseline model training + utilities
│   ├── part_2/                      # architecture-variant model training
│   └── part_3/                      # analysis scripts on trained part_1 runs
├── slurm/
│   ├── part_1/scripts/              # run_part1.slurm
│   ├── part_2/scripts/              # run_part2.slurm
│   └── part_3/scripts/              # three analysis launchers
├── outputs/
│   ├── part_1/                      # training runs + analysis under runs
│   ├── part_2/                      # part_2 training runs
│   └── part_3/                      # separate analysis outputs (fi-only)
├── main.pdf
└── README.md
```

## Data and environment

All train/analysis scripts expect assignment data under:
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data`
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data`

Use:
- `--data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data`

SLURM scripts are set up for:
- Part 1/2: `$HOME/.conda/envs/cs137-cnn/bin/python`
- Part 3: `conda activate cs137-cnn`

## Code map

### `code/part_1`

- `train.py`: main Part 1 training entrypoint (data loading, split logic, training loop, checkpoints, metrics).
- `model.py`: baseline CNN + patch-token + Transformer architecture.
- `multi_seed.py`: launches repeated Part 1 runs across seeds and writes aggregate summaries.
- `crop_sweep.py`: visual helper to tune crop/downsample settings for weather maps.
- `analyze_attention.py`: standalone attention extraction/visualization from trained Part 1 checkpoints.
- `evaluate.py`: legacy standalone evaluation helper script.

### `code/part_2`

- `train.py`: Part 2 training entrypoint (same training pipeline style as Part 1).
- `model.py`: hierarchical encoder-decoder with two variants:
  - `no_cnn`
  - `residual_cnn`

### `code/part_3`

- `attention_spatial_analysis.py`: zone/hour-conditioned patch-level attention attribution from Part 1 runs.
- `saliency_feature_analysis.py`: gradient saliency maps + channel importance + visual artifacts (GIF/plots).
- `feature_importance_only.py`: faster batched feature-importance-only workflow.

## Running experiments

Run from `CS137Assignment3/`.

### Part 1 training

Local:

```bash
python code/part_1/train.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --output-dir outputs/part_1 \
  --run-name part1_exp \
  --years 2019-2023 \
  --epochs 20 \
  --batch-size 16
```

SLURM:

```bash
sbatch slurm/part_1/scripts/run_part1.slurm
```

Multi-seed mode is supported via `SEEDS` in the SLURM script (`multi_seed.py` is called automatically when set).

### Part 2 training

Local single variant:

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

SLURM (runs both `no_cnn` and `residual_cnn` sequentially):

```bash
sbatch slurm/part_2/scripts/run_part2.slurm
```

### Part 3 analysis

Part 3 scripts read a trained Part 1 run directory (for example `outputs/part_1/part1_first_improved_3seed_s40`).

1. Attention spatial analysis
```bash
sbatch slurm/part_3/scripts/run_attention_spatial_analysis.slurm
```

2. Saliency + feature importance
```bash
sbatch slurm/part_3/scripts/run_saliency_feature_analysis.slurm
```

3. Feature importance only (batched, faster)
```bash
sbatch slurm/part_3/scripts/run_feature_importance_only.slurm
```

Edit `RUN_DIR`, `CKPT`, `EVAL_YEAR`, and related variables directly in each Part 3 SLURM script before submitting.

## Key CLI options

Common Part 1/2 training knobs include:
- data/splits: `--years`, `--split-mode`, `--blocked-val-position`, `--val-years`, `--purge-gap`
- spatial preprocessing: `--crop-mode`, `--crop-y0/--crop-y1/--crop-x0/--crop-x1`, `--downsample-h`, `--downsample-w`
- model size: `--d-model`, `--num-heads`, `--num-layers`, `--ff-dim`, `--patch-grid-h`, `--patch-grid-w`
- optimization: `--epochs`, `--batch-size`, `--lr`, `--scheduler`, `--warmup-epochs`, `--min-lr`, `--loss`
- runtime: `--device`, `--amp/--no-amp`, `--resume-from auto`

Part 2-specific knobs:
- `--arch-variant {no_cnn,residual_cnn}`
- `--decoder-layers`, `--spatial-layers`, `--residual-blocks`, `--use-weather-stats`

Part 3 knobs:
- checkpoint/run selection: `--run-dir`, `--part1-model-path`, `--ckpt`
- evaluation scope: `--eval-year` (`YYYY` or `all` in saliency/fi-only), `--n-days`
- zone selection: `--zones`
- perf/output: `--batch-anchors`, `--output-root` (feature-importance-only)

## Output artifacts

Each Part 1/2 run directory (under `outputs/part_1/<run_name>` or `outputs/part_2/<run_name>`) includes:
- `config.json`
- `normalization.json`
- `metrics.csv`
- `train_indices.csv`
- `val_indices.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`

Part 3 output locations:
- `attention_spatial_analysis.py` writes under `<run_dir>/attention_analysis_part3/`
- `saliency_feature_analysis.py` writes under `<run_dir>/saliency_analysis_part3/`
- `feature_importance_only.py` writes under `outputs/part_3/<run_name>/feature_importance_only/` (or custom `--output-root`)

## Logs

SLURM logs are written to:
- `slurm/part_1/logs/`
- `slurm/part_2/logs/`
- `slurm/part_3/logs/`
