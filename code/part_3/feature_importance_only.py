#!/usr/bin/env python3
"""Fast feature-importance-only analysis for Part 3.

Outputs:
- feature_importance_by_zone.csv
- feature_importance_overall.csv
- summary_importance_only.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


VAR_NAMES = [
    "TMP@2m_above_ground",
    "RH@2m_above_ground",
    "GUST@surface",
    "UGRD@10m_above_ground",
    "VGRD@10m_above_ground",
    "APCP_1h_accumulation",
    "DSWRF@surface",
]


def load_part1_model_module(part1_model_path: Path):
    spec = importlib.util.spec_from_file_location("part1_model", str(part1_model_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {part1_model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_energy_df(data_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    dfs = []
    for p in sorted((data_dir / "energy_demand_data").glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(p, parse_dates=["timestamp_utc"]))
    if not dfs:
        raise FileNotFoundError("No energy CSVs found in energy_demand_data")
    energy_df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    zone_cols = [c for c in energy_df.columns if c != "timestamp_utc"]
    return energy_df, zone_cols


def build_eval_anchors(
    energy_df: pd.DataFrame,
    eval_year: int | None,
    history_len: int,
    future_len: int,
    n_days: int,
) -> np.ndarray:
    if eval_year is None:
        midnight_mask = energy_df["timestamp_utc"].dt.hour == 0
    else:
        midnight_mask = (
            (energy_df["timestamp_utc"].dt.year == eval_year)
            & (energy_df["timestamp_utc"].dt.hour == 0)
        )
    anchors = np.where(midnight_mask)[0]
    anchors = anchors[(anchors >= history_len) & (anchors + future_len <= len(energy_df))]
    if len(anchors) == 0:
        scope = "all years" if eval_year is None else f"year {eval_year}"
        raise RuntimeError(f"No valid anchors found for {scope}")
    if n_days <= 0 or n_days >= len(anchors):
        return anchors
    return anchors[-n_days:]


def load_weather_tensor(data_dir: Path, hour_int64: int, cache: Dict[int, torch.Tensor]) -> torch.Tensor:
    if hour_int64 in cache:
        return cache[hour_int64]
    dt = pd.Timestamp(int(hour_int64), unit="h")
    path = data_dir / "weather_data" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    t = torch.load(path, weights_only=True).float()
    cache[hour_int64] = t
    if len(cache) > 256:
        cache.pop(next(iter(cache)))
    return t


def main():
    parser = argparse.ArgumentParser(description="Feature-importance-only analysis")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--part1-model-path", type=Path, default=Path("code/part_1/model.py"))
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--eval-year", type=str, default="all", help="Year (e.g. 2022) or 'all' for all years")
    parser.add_argument("--n-days", type=int, default=0, help="Number of trailing anchor days; <=0 means all anchors")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--zones", type=str, default="ME,NH,VT,CT,RI,SEMA,WCMA,NEMA_BOST")
    parser.add_argument("--batch-anchors", type=int, default=8, help="Number of anchors per gradient batch")
    parser.add_argument(
        "--min-batch-anchors",
        type=int,
        default=1,
        help="Minimum anchor batch size when shrinking after CUDA OOM",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/part_3"),
        help="Base output directory for Part 3 artifacts",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_dir = Path(cfg["data_dir"]).resolve()
    history_len = int(cfg["history_len"])
    future_len = int(cfg["future_len"])

    module = load_part1_model_module(args.part1_model_path.resolve())
    get_model = module.get_model

    energy_df, zone_cols_all = load_energy_df(data_dir)
    requested_zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    missing = [z for z in requested_zones if z not in zone_cols_all]
    if missing:
        raise ValueError(f"Requested zones not found: {missing}. Available: {zone_cols_all}")

    zone_cols = requested_zones
    num_zones = len(zone_cols)

    energy_vals = energy_df[zone_cols].values.astype(np.float32)
    all_hours = energy_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)

    model = get_model(
        weather_channels=7,
        num_zones=num_zones,
        calendar_dim=7,
        history_len=history_len,
        future_steps=future_len,
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        cnn_hidden_dim=int(cfg["cnn_hidden_dim"]),
        patch_grid_h=int(cfg["patch_grid_h"]),
        patch_grid_w=int(cfg["patch_grid_w"]),
        crop_mode=str(cfg["crop_mode"]),
        crop_y0=int(cfg["crop_y0"]),
        crop_y1=int(cfg["crop_y1"]),
        crop_x0=int(cfg["crop_x0"]),
        crop_x1=int(cfg["crop_x1"]),
        downsample_h=int(cfg["downsample_h"]),
        downsample_w=int(cfg["downsample_w"]),
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    if "weather_mean" in checkpoint and "weather_std" in checkpoint:
        model.weather_mean = torch.as_tensor(checkpoint["weather_mean"]).float().view(-1, 1, 1)
        model.weather_std = torch.as_tensor(checkpoint["weather_std"]).float().view(-1, 1, 1)
    if "energy_mean" in checkpoint and "energy_std" in checkpoint:
        model.energy_mean = torch.as_tensor(checkpoint["energy_mean"]).float().view(1, 1, -1)
        model.energy_std = torch.as_tensor(checkpoint["energy_std"]).float().view(1, 1, -1)

    device = torch.device(args.device)
    model.to(device).eval()

    eval_year = None if str(args.eval_year).strip().lower() == "all" else int(args.eval_year)
    anchors = build_eval_anchors(energy_df, eval_year, history_len, future_len, args.n_days)

    out_dir = args.output_root.resolve() / run_dir.name / "feature_importance_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using zones ({num_zones}): {zone_cols}")
    print(f"Anchors: {len(anchors)}")

    channel_hist_importance = np.zeros((num_zones, 7), dtype=np.float64)
    channel_fut_importance = np.zeros((num_zones, 7), dtype=np.float64)

    weather_cache: Dict[int, torch.Tensor] = {}

    batch_anchors = max(1, int(args.batch_anchors))
    min_batch_anchors = max(1, int(args.min_batch_anchors))
    total = len(anchors)
    start = 0
    while start < total:
        current_bs = min(batch_anchors, total - start)
        chunk = anchors[start:start + current_bs]
        bsz = len(chunk)

        try:
            hist_weather_list = []
            fut_weather_list = []
            hist_energy_list = []
            fut_time_list = []
            for t_idx in chunk:
                hist_slice = slice(t_idx - history_len, t_idx)
                fut_slice = slice(t_idx, t_idx + future_len)
                hist_hours = all_hours[hist_slice]
                fut_hours = all_hours[fut_slice]
                hist_weather_list.append(torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in hist_hours]))
                fut_weather_list.append(torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in fut_hours]))
                hist_energy_list.append(torch.from_numpy(energy_vals[hist_slice]))
                fut_time_list.append(torch.from_numpy(fut_hours).to(torch.int64))

            hist_weather_raw = torch.stack(hist_weather_list, dim=0).to(device)
            fut_weather_raw = torch.stack(fut_weather_list, dim=0).to(device)
            hist_energy = torch.stack(hist_energy_list, dim=0).to(device)
            fut_time = torch.stack(fut_time_list, dim=0).to(device)

            hist_weather, hist_demand, hist_cal, fut_weather, fut_cal = model.adapt_inputs(
                hist_weather_raw, hist_energy, fut_weather_raw, fut_time
            )
            hist_weather = hist_weather.detach().requires_grad_(True)
            fut_weather = fut_weather.detach().requires_grad_(True)

            preds = model(hist_weather, hist_demand, hist_cal, fut_weather, fut_cal)

            # Zone-specific gradients with batched anchors for throughput.
            for zi in range(num_zones):
                model.zero_grad(set_to_none=True)
                if hist_weather.grad is not None:
                    hist_weather.grad = None
                if fut_weather.grad is not None:
                    fut_weather.grad = None

                scalar = preds[:, :, zi].sum()
                scalar.backward(retain_graph=(zi < num_zones - 1))

                g_hist = hist_weather.grad.detach().abs()
                g_fut = fut_weather.grad.detach().abs()
                channel_hist_importance[zi] += g_hist.mean(dim=(0, 1, 3, 4)).cpu().numpy()
                channel_fut_importance[zi] += g_fut.mean(dim=(0, 1, 3, 4)).cpu().numpy()

            done = start + bsz
            if (done % 64 == 0) or (done == total):
                print(f"[{done:>4}/{total}] processed through {energy_df['timestamp_utc'].iloc[int(chunk[-1])].date()} (batch={bsz})")

            del hist_weather_list, fut_weather_list, hist_energy_list, fut_time_list
            del hist_weather_raw, fut_weather_raw, hist_energy, fut_time
            del hist_weather, hist_demand, hist_cal, fut_weather, fut_cal, preds
            if device.type == "cuda":
                torch.cuda.empty_cache()

            start = done
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            if current_bs <= min_batch_anchors:
                raise RuntimeError(
                    f"CUDA OOM at minimum batch size {current_bs}. "
                    "Try reducing model size or running on a larger GPU."
                ) from None
            next_bs = max(min_batch_anchors, current_bs // 2)
            batch_anchors = next_bs
            print(f"OOM at anchor batch {current_bs}; retrying from index {start} with batch {next_bs}")

    channel_hist_importance /= float(len(anchors))
    channel_fut_importance /= float(len(anchors))

    rows = []
    for zi, zone in enumerate(zone_cols):
        hist_vals = channel_hist_importance[zi]
        fut_vals = channel_fut_importance[zi]
        total = hist_vals + fut_vals
        order = np.argsort(total)[::-1]
        for rank, ci in enumerate(order, start=1):
            rows.append(
                {
                    "zone": zone,
                    "rank": rank,
                    "feature": VAR_NAMES[ci],
                    "hist_importance": float(hist_vals[ci]),
                    "future_importance": float(fut_vals[ci]),
                    "total_importance": float(total[ci]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "feature_importance_by_zone.csv", index=False)

    overall_hist = channel_hist_importance.mean(axis=0)
    overall_fut = channel_fut_importance.mean(axis=0)
    overall_total = overall_hist + overall_fut
    overall_order = np.argsort(overall_total)[::-1]
    overall_rows = []
    for rank, ci in enumerate(overall_order, start=1):
        overall_rows.append(
            {
                "rank": rank,
                "feature": VAR_NAMES[ci],
                "hist_importance": float(overall_hist[ci]),
                "future_importance": float(overall_fut[ci]),
                "total_importance": float(overall_total[ci]),
            }
        )
    pd.DataFrame(overall_rows).to_csv(out_dir / "feature_importance_overall.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "eval_year": "all" if eval_year is None else int(eval_year),
        "n_days": int(len(anchors)),
        "zones": zone_cols,
        "method": "importance_only_sum_backward",
        "notes": [
            "Uses one backward pass per zone while processing anchors in mini-batches.",
            "Automatically shrinks anchor batch size and retries when CUDA OOM is encountered.",
        ],
    }
    with open(out_dir / "summary_importance_only.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved feature-importance CSVs to: {out_dir}")


if __name__ == "__main__":
    main()
