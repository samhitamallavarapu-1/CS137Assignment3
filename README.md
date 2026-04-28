# CS137Assignment3

Code and experiment outputs for CS137 Assignment 3:
1. Part 1 baseline forecasting model
2. Part 2 architecture search / improved model
3. Part 3 model diagnostics and attention analysis

## Repository layout

```text
CS137Assignment3/
├── code/
│   ├── part_1/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── multi_seed.py
│   │   ├── crop_sweep.py
│   │   └── evaluate.py
│   ├── part_2/
│   │   ├── model.py
│   │   └── train.py
│   └── part_3/
│       ├── README.md
│       ├── run.py
│       ├── attention_tools.py
│       ├── diagnostic_saliency.py
│       ├── diagnostic_layer_ablation.py
│       └── plot_maps.py
├── outputs/
│   ├── part_1/
│   ├── part_2/
│   └── part_3/
├── slurm/
│   ├── part_1/scripts/run_part1.slurm
│   ├── part_2/scripts/run_part2.slurm
│   └── part_3/scripts/run_part3.slurm
└── main.pdf
```

## Data location

All training/diagnostic scripts expect:
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data`
- `/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data`

Pass the parent directory as `--data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data`.

## Environment

Provided Slurm scripts use:
- `$HOME/.conda/envs/cs137-cnn/bin/python`

## Part 1

Main training entrypoint:
- `code/part_1/train.py`

Typical run:

```bash
python code/part_1/train.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --output-dir outputs/part_1 \
  --run-name part1_exp \
  --years 2019-2023 \
  --epochs 20 \
  --batch-size 16
```

Part 1 Slurm:

```bash
sbatch slurm/part_1/scripts/run_part1.slurm
```

Useful utilities:
- `code/part_1/multi_seed.py`: multi-seed launcher + summary
- `code/part_1/crop_sweep.py`: crop/downsample visual diagnostics
- `code/part_1/evaluate.py`: legacy evaluator-style script (path assumptions differ from current repo layout)

## Part 2

Main training entrypoint:
- `code/part_2/train.py`

Supports both architecture variants:
- `no_cnn`
- `residual_cnn`

Typical run:

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

Part 2 Slurm (runs both variants):

```bash
sbatch slurm/part_2/scripts/run_part2.slurm
```

## Part 3

Part 3 extracts and analyzes attribution maps from a trained Part 1 run.

Primary script:
- `code/part_3/run.py`

Typical run:

```bash
python code/part_3/run.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --part1-run-dir outputs/part_1/<run_name> \
  --output-dir outputs/part_3 \
  --checkpoint best \
  --split val \
  --max-samples 256 \
  --batch-size 8
```

Additional diagnostics:
- `code/part_3/diagnostic_saliency.py`
- `code/part_3/diagnostic_layer_ablation.py`
- `code/part_3/plot_maps.py`

Part 3 Slurm:

```bash
sbatch slurm/part_3/scripts/run_part3.slurm
```

See `code/part_3/README.md` for artifact formats and plotting details.

## Output artifacts

Each training run folder stores:
- `config.json`
- `normalization.json`
- `metrics.csv`
- `train_indices.csv`
- `val_indices.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
