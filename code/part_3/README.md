# Part 3: Attention-Based Geographic Diagnosis

This module extracts spatial attribution maps from the Part 2 model by combining:

- decoder cross-attention (`future token -> history hour token`)
- hourly summarizer attention (`history hour token -> spatial patch token`)

and composing them into:

- `future token -> spatial patch token` maps

It also supports an optional zone-conditioned diagnostic map using gradient sensitivity for each output zone.

## What gets saved

Running `run.py` writes a new folder under `outputs/part_3/<part2_run>_<split>_<timestamp>/` with:

- `attention_maps.npz`
  - `future_to_history`: `[N, Tf, Th]`
  - `history_to_spatial`: `[N, Th, patch_grid_h, patch_grid_w]`
  - `future_to_spatial`: `[N, Tf, patch_grid_h, patch_grid_w]`
  - `zone_grad_spatial` (optional): `[N, Tf, Z, patch_grid_h, patch_grid_w]`
  - `predictions`, `anchor_index`, `anchor_timestamp_utc`, and metadata arrays
- `mean_future_to_spatial.csv`: mean map per forecast hour
- `top_patches_by_zone.csv`: top-K cells by zone-conditioned score (if enabled)
- `metadata.json`

## Example command

```bash
python code/part_3/run.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --part2-run-dir outputs/part_2/part2_first_improved_3seed_no_cnn \
  --output-dir outputs/part_3 \
  --checkpoint best \
  --split val \
  --max-samples 256 \
  --batch-size 8
```

## Suggested next diagnostics

1. Saliency / gradient maps on weather input:
   Compare against attention maps to verify whether high-attention regions are truly prediction-sensitive.
2. Integrated Gradients (IG):
   More stable than raw gradients for nonlinear heads and better for per-zone attribution claims.
3. Occlusion sensitivity:
   Zero out one spatial patch (or local window) at a time and measure per-zone prediction delta.
4. Attention-layer ablation:
   Compare maps from early vs late decoder layers to see if spatial focus sharpens with depth.
5. Temporal robustness:
   Check whether zone maps are stable across seasons and major weather regimes.

## Diagnostic 1: Saliency Maps

Compute gradient saliency maps on weather input for each forecast hour and zone:

```bash
python code/part_3/diagnostic_saliency.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --part2-run-dir outputs/part_2/part2_first_improved_3seed_no_cnn \
  --output-dir outputs/part_3 \
  --checkpoint best \
  --split val \
  --max-samples 64 \
  --batch-size 4
```

Outputs include:

- `saliency_maps.npz`
- `saliency_summary.csv`
- `metadata.json`

## Diagnostic 4: Attention-Layer Ablation

Compare early vs late decoder-layer maps and quantify prediction shifts when dropping/isolating decoder cross-attention layers:

```bash
python code/part_3/diagnostic_layer_ablation.py \
  --data-dir /cluster/tufts/c26sp1cs0137/data/assignment3_data \
  --part2-run-dir outputs/part_2/part2_first_improved_3seed_no_cnn \
  --output-dir outputs/part_3 \
  --checkpoint best \
  --split val \
  --max-samples 128 \
  --batch-size 8
```

Outputs include:

- `layer_ablation_maps.npz`
- `ablation_prediction_deltas.csv`
- `layer_map_drift.csv`
- `metadata.json`

## Plotting script

Use `plot_maps.py` to generate publication-ready heatmaps from `attention_maps.npz`.

```bash
python code/part_3/plot_maps.py \
  --attention-npz outputs/part_3/<run_folder>/attention_maps.npz \
  --future-hours 0,6,12,18 \
  --zones all
```

Outputs are written to:

- `outputs/part_3/<run_folder>/figures/` (default)

Key options:

- `--future-hours`: `all`, range (`0-23`), or list (`0,6,12,18`)
- `--zones`: `all`, range, or list of zone indices (only used when `zone_grad_spatial` exists)
- `--cmap`, `--vmin`, `--vmax`: colormap and fixed color scaling controls for consistent comparisons
