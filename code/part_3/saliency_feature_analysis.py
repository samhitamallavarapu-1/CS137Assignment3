#!/usr/bin/env python3
"""Saliency + feature-importance analysis for Part 3.

Outputs:
- Gradient saliency maps per zone/hour (future weather input)
- Heatmaps and GIFs per zone across forecast horizon
- Channel-importance rankings for CNN weather features
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


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


def save_zone_saliency_plots(out_dir: Path, zone_names: List[str], sal: np.ndarray):
    # sal: [Z,Tf,H,W]
    if not HAS_MPL:
        return
    selected = [0, 5, 11, 17, 23]

    for zi, zone in enumerate(zone_names):
        # fixed horizons
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
        vmin = float(sal[zi].min())
        vmax = float(sal[zi].max())
        for k, h in enumerate(selected):
            ax = axes[k]
            im = ax.imshow(sal[zi, h], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"{zone} +{h+1}h")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        fig.savefig(out_dir / f"saliency_{zone}_hours_1_6_12_18_24.png", dpi=180)
        plt.close(fig)

        # sequential panel
        fig, axes = plt.subplots(4, 6, figsize=(13, 8), constrained_layout=True)
        vmin = float(sal[zi].min())
        vmax = float(sal[zi].max())
        for h in range(sal.shape[1]):
            r, c = divmod(h, 6)
            ax = axes[r, c]
            im = ax.imshow(sal[zi, h], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"+{h+1}h", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65)
        fig.suptitle(f"{zone} saliency shift over +1h...+24h", fontsize=12)
        fig.savefig(out_dir / f"saliency_{zone}_sequential_panel.png", dpi=180)
        plt.close(fig)

        if HAS_IMAGEIO:
            try:
                frames = []
                for h in range(sal.shape[1]):
                    fig, ax = plt.subplots(figsize=(4.5, 4.0))
                    ax.imshow(sal[zi, h], cmap="magma", vmin=vmin, vmax=vmax)
                    ax.set_title(f"{zone} saliency +{h+1}h")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.canvas.draw()
                    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    rgb = rgba[:, :, :3].copy()
                    frames.append(rgb)
                    plt.close(fig)
                imageio.mimsave(out_dir / f"saliency_{zone}.gif", frames, fps=2)
            except Exception as e:
                print(f"Warning: GIF generation skipped for zone={zone}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Gradient saliency and feature-importance analysis")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--part1-model-path", type=Path, default=Path("code/part_1/model.py"))
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--eval-year", type=str, default="2022", help="Year (e.g. 2022) or 'all' for all years")
    parser.add_argument("--n-days", type=int, default=10, help="Number of trailing anchor days; <=0 means all anchors")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--zones", type=str, default="ME,NH,VT,CT,RI,SEMA,WCMA,NEMA_BOST")
    parser.add_argument("--clean-output", action="store_true")
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

    out_dir = run_dir / "saliency_analysis_part3"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using zones ({num_zones}): {zone_cols}")

    # Aggregate saliency at model weather resolution [Z,Tf,H,W]
    h_ds = int(cfg["downsample_h"])
    w_ds = int(cfg["downsample_w"])
    zone_hour_sal = np.zeros((num_zones, future_len, h_ds, w_ds), dtype=np.float64)

    # Per-feature importance accumulators (history and future weather channels)
    channel_hist_importance = np.zeros((num_zones, 7), dtype=np.float64)
    channel_fut_importance = np.zeros((num_zones, 7), dtype=np.float64)

    weather_cache: Dict[int, torch.Tensor] = {}

    for i, t_idx in enumerate(anchors):
        hist_slice = slice(t_idx - history_len, t_idx)
        fut_slice = slice(t_idx, t_idx + future_len)

        hist_hours = all_hours[hist_slice]
        fut_hours = all_hours[fut_slice]

        hist_weather_raw = torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in hist_hours]).unsqueeze(0).to(device)
        fut_weather_raw = torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in fut_hours]).unsqueeze(0).to(device)
        hist_energy = torch.from_numpy(energy_vals[hist_slice]).unsqueeze(0).to(device)
        fut_time = torch.from_numpy(fut_hours).to(torch.int64).unsqueeze(0).to(device)

        # Build grad-enabled adapted inputs
        hist_weather, hist_demand, hist_cal, fut_weather, fut_cal = model.adapt_inputs(
            hist_weather_raw, hist_energy, fut_weather_raw, fut_time
        )
        hist_weather = hist_weather.detach().requires_grad_(True)
        fut_weather = fut_weather.detach().requires_grad_(True)

        preds = model(hist_weather, hist_demand, hist_cal, fut_weather, fut_cal)  # [1,Tf,Z]

        for zi in range(num_zones):
            for h in range(future_len):
                model.zero_grad(set_to_none=True)
                if hist_weather.grad is not None:
                    hist_weather.grad.zero_()
                if fut_weather.grad is not None:
                    fut_weather.grad.zero_()

                scalar = preds[0, h, zi]
                scalar.backward(retain_graph=True)

                g_hist = hist_weather.grad.detach().abs()[0]  # [Th,C,H,W]
                g_fut = fut_weather.grad.detach().abs()[0]    # [Tf,C,H,W]

                # Saliency map for this zone/hour from same forecast hour in future weather.
                zone_hour_sal[zi, h] += g_fut[h].mean(dim=0).cpu().numpy()

                # Channel importances
                channel_hist_importance[zi] += g_hist.mean(dim=(0, 2, 3)).cpu().numpy()
                channel_fut_importance[zi] += g_fut.mean(dim=(0, 2, 3)).cpu().numpy()

        print(f"[{i+1:>3}/{len(anchors)}] processed {energy_df['timestamp_utc'].iloc[t_idx].date()}")

    zone_hour_sal /= float(len(anchors))
    channel_hist_importance /= float(len(anchors) * future_len)
    channel_fut_importance /= float(len(anchors) * future_len)

    np.save(out_dir / "zone_hour_saliency.npy", zone_hour_sal.astype(np.float32))

    # Save patch-level saliency via adaptive pooling
    gh = int(cfg["patch_grid_h"])
    gw = int(cfg["patch_grid_w"])
    patch_sal = np.zeros((num_zones, future_len, gh, gw), dtype=np.float32)
    for zi in range(num_zones):
        for h in range(future_len):
            t = torch.from_numpy(zone_hour_sal[zi, h]).float().unsqueeze(0).unsqueeze(0)
            p = torch.nn.functional.adaptive_avg_pool2d(t, (gh, gw)).squeeze().numpy()
            patch_sal[zi, h] = p
    np.save(out_dir / "zone_hour_saliency_patch.npy", patch_sal)

    # Feature-importance ratings
    var_names = [
        "TMP@2m_above_ground",
        "RH@2m_above_ground",
        "GUST@surface",
        "UGRD@10m_above_ground",
        "VGRD@10m_above_ground",
        "APCP_1h_accumulation",
        "DSWRF@surface",
    ]

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
                    "feature": var_names[ci],
                    "hist_importance": float(hist_vals[ci]),
                    "future_importance": float(fut_vals[ci]),
                    "total_importance": float(total[ci]),
                }
            )
    fi_df = pd.DataFrame(rows)
    fi_df.to_csv(out_dir / "feature_importance_by_zone.csv", index=False)

    overall_hist = channel_hist_importance.mean(axis=0)
    overall_fut = channel_fut_importance.mean(axis=0)
    overall_total = overall_hist + overall_fut
    overall_order = np.argsort(overall_total)[::-1]
    overall_rows = []
    for rank, ci in enumerate(overall_order, start=1):
        overall_rows.append(
            {
                "rank": rank,
                "feature": var_names[ci],
                "hist_importance": float(overall_hist[ci]),
                "future_importance": float(overall_fut[ci]),
                "total_importance": float(overall_total[ci]),
            }
        )
    pd.DataFrame(overall_rows).to_csv(out_dir / "feature_importance_overall.csv", index=False)

    save_zone_saliency_plots(out_dir, zone_cols, zone_hour_sal.astype(np.float32))

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "eval_year": "all" if eval_year is None else int(eval_year),
        "n_days": int(len(anchors)),
        "zones": zone_cols,
        "saliency_shape": list(zone_hour_sal.shape),
        "patch_saliency_shape": list(patch_sal.shape),
        "has_matplotlib": HAS_MPL,
        "has_imageio": HAS_IMAGEIO,
        "notes": [
            "Saliency is absolute gradient of zone/hour output w.r.t. normalized weather inputs.",
            "Feature-importance ratings are gradient-magnitude based (higher means more influence).",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved saliency analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
