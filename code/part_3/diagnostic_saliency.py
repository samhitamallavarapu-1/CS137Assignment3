#!/usr/bin/env python3
"""Diagnostic 1: saliency maps on weather input for Part 1 model outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run import _build_dataset, _build_model, _make_output_dir, _resolve_checkpoint


def _pool_to_patch_grid(x: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """x: [B, H, W] -> [B, patch_h, patch_w]."""
    return F.adaptive_avg_pool2d(x.unsqueeze(1), output_size=(patch_h, patch_w)).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute weather-input saliency maps for each forecast hour and zone")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--part1-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_3"))
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
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

    loader_kwargs: Dict[str, int | bool] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

    loader = DataLoader(ds, batch_size=max(1, args.batch_size), shuffle=False, **loader_kwargs)

    patch_h = int(cfg["patch_grid_h"])
    patch_w = int(cfg["patch_grid_w"])
    tf = int(cfg["future_len"])
    num_zones = len(zone_cols)

    hist_saliency_all: List[torch.Tensor] = []
    fut_saliency_all: List[torch.Tensor] = []

    print("=" * 80)
    print("Part 3 Diagnostic 1: weather saliency maps")
    print(f"part1_run_dir: {part1_run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"split={args.split} samples={len(ds)} batch_size={args.batch_size} device={device}")
    print("=" * 80)

    for batch in loader:
        hist_weather = batch["hist_weather"].to(device, non_blocking=True).detach().requires_grad_(True)
        hist_demand = batch["hist_demand"].to(device, non_blocking=True)
        hist_calendar = batch["hist_calendar"].to(device, non_blocking=True)
        fut_weather = batch["fut_weather"].to(device, non_blocking=True).detach().requires_grad_(True)
        fut_calendar = batch["fut_calendar"].to(device, non_blocking=True)

        preds = model(hist_weather, hist_demand, hist_calendar, fut_weather, fut_calendar)  # [B,Tf,Z]
        bsz = preds.shape[0]

        hist_maps = torch.zeros(bsz, tf, num_zones, patch_h, patch_w, device=device)
        fut_maps = torch.zeros(bsz, tf, num_zones, patch_h, patch_w, device=device)

        for fut_idx in range(tf):
            for zone_idx in range(num_zones):
                score = preds[:, fut_idx, zone_idx].sum()
                grad_hist, grad_fut = torch.autograd.grad(
                    score,
                    [hist_weather, fut_weather],
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )
                # Collapse channels and time, then pool to patch grid.
                h_map = grad_hist.abs().mean(dim=2).mean(dim=1)  # [B, H, W]
                f_map = grad_fut.abs().mean(dim=2).mean(dim=1)  # [B, H, W]
                hist_maps[:, fut_idx, zone_idx] = _pool_to_patch_grid(h_map, patch_h, patch_w)
                fut_maps[:, fut_idx, zone_idx] = _pool_to_patch_grid(f_map, patch_h, patch_w)

        hist_saliency_all.append(hist_maps.detach().cpu())
        fut_saliency_all.append(fut_maps.detach().cpu())

    hist_saliency = torch.cat(hist_saliency_all, dim=0).numpy()  # [N,Tf,Z,Ph,Pw]
    fut_saliency = torch.cat(fut_saliency_all, dim=0).numpy()  # [N,Tf,Z,Ph,Pw]

    run_name = f"{part1_run_dir.name}_{args.split}_diag1_saliency"
    out_dir = _make_output_dir(args.output_dir, run_name)

    np.savez_compressed(
        out_dir / "saliency_maps.npz",
        zone_names=np.asarray(zone_cols, dtype=object),
        hist_input_saliency=hist_saliency,
        fut_input_saliency=fut_saliency,
        patch_grid_h=np.array([patch_h], dtype=np.int64),
        patch_grid_w=np.array([patch_w], dtype=np.int64),
    )

    rows = []
    mean_hist = hist_saliency.mean(axis=0)  # [Tf,Z,Ph,Pw]
    mean_fut = fut_saliency.mean(axis=0)
    for t in range(tf):
        for z in range(num_zones):
            rows.append(
                {
                    "future_hour_idx": t,
                    "zone": zone_cols[z],
                    "hist_saliency_mean": float(mean_hist[t, z].mean()),
                    "fut_saliency_mean": float(mean_fut[t, z].mean()),
                    "hist_saliency_max": float(mean_hist[t, z].max()),
                    "fut_saliency_max": float(mean_fut[t, z].max()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "saliency_summary.csv", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_local": datetime.now().isoformat(timespec="seconds"),
                "diagnostic": "1_saliency_maps",
                "part1_run_dir": str(part1_run_dir),
                "checkpoint": str(ckpt_path),
                "split": args.split,
                "num_samples": int(hist_saliency.shape[0]),
                "future_len": tf,
                "num_zones": num_zones,
                "patch_grid_h": patch_h,
                "patch_grid_w": patch_w,
            },
            f,
            indent=2,
        )

    print(f"Saved saliency diagnostics to: {out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
