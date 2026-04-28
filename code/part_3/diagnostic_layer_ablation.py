#!/usr/bin/env python3
"""Diagnostic 4: attention-layer ablation and early-vs-late map comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from attention_tools import (
    extract_layerwise_future_to_spatial_batch,
    predict_with_layer_mask,
    reshape_patch_maps,
)
from run import _build_dataset, _build_model, _make_output_dir, _resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transformer-layer attention maps and run layer-mask ablations")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--part1-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_3"))
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--weather-cache-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    part1_run_dir = args.part1_run_dir.resolve()
    ckpt_path = _resolve_checkpoint(part1_run_dir, args.checkpoint)

    ds, _, zone_cols, cfg, norm = _build_dataset(
        data_dir=args.data_dir,
        run_dir=part1_run_dir,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        weather_cache_size=args.weather_cache_size,
    )
    if len(ds) == 0:
        raise RuntimeError("No samples selected. Check --split / --max-samples.")

    model = _build_model(cfg=cfg, norm=norm, checkpoint_path=ckpt_path, device=device)
    model.eval()
    num_layers = len(model.transformer.layers)
    patch_h = int(cfg["patch_grid_h"])
    patch_w = int(cfg["patch_grid_w"])

    loader_kwargs: Dict[str, int | bool] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

    loader = DataLoader(ds, batch_size=max(1, args.batch_size), shuffle=False, **loader_kwargs)

    layer_maps_all: List[torch.Tensor] = []  # each [L,B,Tf,P]
    baseline_pred_all: List[torch.Tensor] = []

    # We keep mean absolute prediction shift vs baseline for each ablation mode.
    delta_drop = torch.zeros(num_layers, dtype=torch.float64)
    delta_only = torch.zeros(num_layers, dtype=torch.float64)
    n_batches = 0

    print("=" * 80)
    print("Part 3 Diagnostic 4: layer ablation")
    print(f"part1_run_dir: {part1_run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"split={args.split} samples={len(ds)} batch_size={args.batch_size} device={device}")
    print("=" * 80)

    with torch.no_grad():
        for batch in loader:
            hist_weather = batch["hist_weather"].to(device, non_blocking=True)
            hist_demand = batch["hist_demand"].to(device, non_blocking=True)
            hist_calendar = batch["hist_calendar"].to(device, non_blocking=True)
            fut_weather = batch["fut_weather"].to(device, non_blocking=True)
            fut_calendar = batch["fut_calendar"].to(device, non_blocking=True)

            layer_maps, _, _ = extract_layerwise_future_to_spatial_batch(
                model=model,
                hist_weather=hist_weather,
                hist_demand=hist_demand,
                hist_calendar=hist_calendar,
                fut_weather=fut_weather,
                fut_calendar=fut_calendar,
            )  # [L,B,Tf,P]
            layer_maps_all.append(layer_maps.detach().cpu())

            pred_base = predict_with_layer_mask(
                model=model,
                hist_weather=hist_weather,
                hist_demand=hist_demand,
                hist_calendar=hist_calendar,
                fut_weather=fut_weather,
                fut_calendar=fut_calendar,
                active_layers=None,
            )  # [B,Tf,Z]
            baseline_pred_all.append(pred_base.detach().cpu())

            for li in range(num_layers):
                active_drop: Set[int] = set(range(num_layers))
                active_drop.discard(li)
                pred_drop = predict_with_layer_mask(
                    model=model,
                    hist_weather=hist_weather,
                    hist_demand=hist_demand,
                    hist_calendar=hist_calendar,
                    fut_weather=fut_weather,
                    fut_calendar=fut_calendar,
                    active_layers=active_drop,
                )
                pred_only = predict_with_layer_mask(
                    model=model,
                    hist_weather=hist_weather,
                    hist_demand=hist_demand,
                    hist_calendar=hist_calendar,
                    fut_weather=fut_weather,
                    fut_calendar=fut_calendar,
                    active_layers={li},
                )
                delta_drop[li] += (pred_drop - pred_base).abs().mean().item()
                delta_only[li] += (pred_only - pred_base).abs().mean().item()

            n_batches += 1

    layer_maps_cat = torch.cat(layer_maps_all, dim=1)  # [L,N,Tf,P]
    layer_maps_2d = reshape_patch_maps(layer_maps_cat, patch_h, patch_w).numpy()  # [L,N,Tf,Ph,Pw]
    baseline_preds = torch.cat(baseline_pred_all, dim=0).numpy()  # [N,Tf,Z]

    mean_layer_map = layer_maps_2d.mean(axis=1)  # [L,Tf,Ph,Pw]
    early_late_delta = mean_layer_map[-1] - mean_layer_map[0]  # [Tf,Ph,Pw]

    # Layer drift metric: L1 distance between adjacent layer maps.
    layer_l1_drift = []
    for li in range(num_layers - 1):
        d = np.abs(mean_layer_map[li + 1] - mean_layer_map[li]).mean()
        layer_l1_drift.append(float(d))

    run_name = f"{part1_run_dir.name}_{args.split}_diag4_layer_ablation"
    out_dir = _make_output_dir(args.output_dir, run_name)

    np.savez_compressed(
        out_dir / "layer_ablation_maps.npz",
        zone_names=np.asarray(zone_cols, dtype=object),
        layer_future_to_spatial=layer_maps_2d,
        early_vs_late_delta=early_late_delta,
        baseline_predictions=baseline_preds,
        patch_grid_h=np.array([patch_h], dtype=np.int64),
        patch_grid_w=np.array([patch_w], dtype=np.int64),
    )

    rows = []
    for li in range(num_layers):
        rows.append(
            {
                "layer_idx": li,
                "drop_layer_mean_abs_pred_delta": float(delta_drop[li].item() / max(1, n_batches)),
                "only_layer_mean_abs_pred_delta": float(delta_only[li].item() / max(1, n_batches)),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "ablation_prediction_deltas.csv", index=False)

    drift_rows = [{"from_layer": i, "to_layer": i + 1, "mean_l1_map_drift": layer_l1_drift[i]} for i in range(num_layers - 1)]
    pd.DataFrame(drift_rows).to_csv(out_dir / "layer_map_drift.csv", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_local": datetime.now().isoformat(timespec="seconds"),
                "diagnostic": "4_attention_layer_ablation",
                "part1_run_dir": str(part1_run_dir),
                "checkpoint": str(ckpt_path),
                "split": args.split,
                "num_samples": int(layer_maps_2d.shape[1]),
                "num_transformer_layers": num_layers,
                "patch_grid_h": patch_h,
                "patch_grid_w": patch_w,
                "notes": [
                    "drop_layer_mean_abs_pred_delta: remove one transformer layer at a time.",
                    "only_layer_mean_abs_pred_delta: keep exactly one transformer layer active.",
                    "early_vs_late_delta: last-layer map minus first-layer map.",
                ],
            },
            f,
            indent=2,
        )

    print(f"Saved layer-ablation diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
