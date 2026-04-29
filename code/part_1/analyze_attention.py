#!/usr/bin/env python3
"""Attention analysis for Part 1 CNN-Transformer patch model.

Extracts attention from future tabular tokens (prediction-hour query tokens)
to all spatial patch tokens, reshapes attention over patches back to 2D, and
writes summary visualizations/statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import get_model


def load_energy_df(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    for p in sorted((data_dir / "energy_demand_data").glob("target_energy_zonal_*.csv")):
        dfs.append(pd.read_csv(p, parse_dates=["timestamp_utc"]))
    if not dfs:
        raise FileNotFoundError("No energy CSV files found")
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    zone_cols = [c for c in df.columns if c != "timestamp_utc"]
    return df, zone_cols


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


def build_eval_indices(energy_df: pd.DataFrame, eval_year: int, n_days: int, history_len: int, future_len: int) -> np.ndarray:
    midnight_mask = (
        (energy_df["timestamp_utc"].dt.year == eval_year) &
        (energy_df["timestamp_utc"].dt.hour == 0)
    )
    anchors = np.where(midnight_mask)[0]
    anchors = anchors[(anchors >= history_len) & (anchors + future_len <= len(energy_df))]
    if len(anchors) == 0:
        raise RuntimeError(f"No valid midnight anchors found for eval year {eval_year}")
    return anchors[-n_days:]


def build_model_seq(
    model: torch.nn.Module,
    hist_weather_raw: torch.Tensor,
    hist_energy: torch.Tensor,
    fut_weather_raw: torch.Tensor,
    fut_hours: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    with torch.no_grad():
        hist_weather, hist_demand, hist_cal, fut_weather, fut_cal = model.adapt_inputs(
            hist_weather_raw, hist_energy, fut_weather_raw, fut_hours
        )
        bsz, hist_steps = hist_weather.shape[:2]
        fut_steps = fut_weather.shape[1]
        num_patches = model.num_patches

        hist_spatial = model.patch_tokenizer(hist_weather) + model.spatial_pos_embed
        fut_spatial = model.patch_tokenizer(fut_weather) + model.spatial_pos_embed

        hist_tab = model.hist_tabular_embed(torch.cat([hist_demand, hist_cal], dim=-1)).unsqueeze(2)
        masked_future_demand = model.future_demand_mask.view(1, 1, -1).expand(bsz, fut_steps, -1)
        fut_tab = model.fut_tabular_embed(torch.cat([masked_future_demand, fut_cal], dim=-1)).unsqueeze(2)

        hist_group = torch.cat([hist_spatial, hist_tab], dim=2)
        fut_group = torch.cat([fut_spatial, fut_tab], dim=2)
        all_group = torch.cat([hist_group, fut_group], dim=1)
        all_group = all_group + model._temporal_encoding(hist_steps + fut_steps, all_group.device)

        tokens_per_step = num_patches + 1
        seq = all_group.reshape(bsz, (hist_steps + fut_steps) * tokens_per_step, model.config.d_model)
        seq = model.dropout(seq)
        return seq, hist_steps, fut_steps, tokens_per_step


def run_encoder_and_collect_attention(model: torch.nn.Module, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = seq
    layer_attn = []
    for layer in model.transformer.layers:
        if layer.norm_first:
            xn = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                xn, xn, xn, need_weights=True, average_attn_weights=False
            )
            x = x + layer.dropout1(attn_out)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            attn_out, attn_w = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            x = layer.norm2(x + layer._ff_block(x))
        layer_attn.append(attn_w.detach())

    if model.transformer.norm is not None:
        x = model.transformer.norm(x)
    return x, torch.stack(layer_attn, dim=0)


def topk_patch_table(zone_names: list[str], zone_patch_maps: np.ndarray, topk: int) -> pd.DataFrame:
    # zone_patch_maps: [Z, Gh, Gw]
    rows = []
    z_count, gh, gw = zone_patch_maps.shape
    for z in range(z_count):
        flat = zone_patch_maps[z].reshape(-1)
        idx = np.argsort(flat)[::-1][:topk]
        for rank, i in enumerate(idx, start=1):
            py = int(i // gw)
            px = int(i % gw)
            rows.append(
                {
                    "zone": zone_names[z],
                    "rank": rank,
                    "patch_y": py,
                    "patch_x": px,
                    "weight": float(flat[i]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/part_1/<run_name>")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--eval-year", type=int, default=2022)
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topk", type=int, default=5)
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

    energy_df, zone_cols = load_energy_df(data_dir)
    all_hours = energy_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)
    energy_vals = energy_df[zone_cols].values.astype(np.float32)

    model = get_model(
        weather_channels=7,
        num_zones=len(zone_cols),
        calendar_dim=7,
        history_len=int(cfg["history_len"]),
        future_steps=int(cfg["future_len"]),
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
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if "weather_mean" in checkpoint and "weather_std" in checkpoint:
        model.weather_mean = torch.as_tensor(checkpoint["weather_mean"]).float().view(-1, 1, 1)
        model.weather_std = torch.as_tensor(checkpoint["weather_std"]).float().view(-1, 1, 1)
    if "energy_mean" in checkpoint and "energy_std" in checkpoint:
        model.energy_mean = torch.as_tensor(checkpoint["energy_mean"]).float().view(1, 1, -1)
        model.energy_std = torch.as_tensor(checkpoint["energy_std"]).float().view(1, 1, -1)
    model.eval().to(args.device)

    anchors = build_eval_indices(energy_df, args.eval_year, args.n_days, history_len, future_len)
    out_dir = run_dir / "attention_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_zones = len(zone_cols)
    gh, gw = model.config.patch_grid_h, model.config.patch_grid_w
    fut_len = future_len

    # We aggregate across: samples, layers, heads, and source timesteps.
    zone_hour_patch_sum = np.zeros((num_zones, fut_len, gh, gw), dtype=np.float64)
    weather_cache: Dict[int, torch.Tensor] = {}

    for step_i, t_idx in enumerate(anchors):
        hist_slice = slice(t_idx - history_len, t_idx)
        fut_slice = slice(t_idx, t_idx + future_len)
        hist_hours = all_hours[hist_slice]
        fut_hours = all_hours[fut_slice]

        hist_weather = torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in hist_hours]).unsqueeze(0)
        fut_weather = torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in fut_hours]).unsqueeze(0)
        hist_energy = torch.from_numpy(energy_vals[hist_slice]).unsqueeze(0)
        fut_time = torch.from_numpy(fut_hours).to(torch.int64).unsqueeze(0)

        hist_weather = hist_weather.to(args.device)
        fut_weather = fut_weather.to(args.device)
        hist_energy = hist_energy.to(args.device)
        fut_time = fut_time.to(args.device)

        seq, hist_steps, fut_steps, tps = build_model_seq(model, hist_weather, hist_energy, fut_weather, fut_time)
        _, attn = run_encoder_and_collect_attention(model, seq)
        # attn shape: [L, B, H, S, S]
        attn_mean = attn.mean(dim=(0, 2))[0]  # [S, S], averaged over layers/heads, batch=1

        # Query positions: future tab tokens.
        fut_t_idx = torch.arange(hist_steps, hist_steps + fut_steps, device=attn_mean.device)
        q_pos = fut_t_idx * tps + model.num_patches  # [Tf]

        # Spatial key positions and patch ids.
        seq_idx = torch.arange(attn_mean.shape[1], device=attn_mean.device)
        key_is_spatial = (seq_idx % tps) < model.num_patches
        k_pos = seq_idx[key_is_spatial]
        patch_ids = (k_pos % tps).to(torch.long)  # [N_spatial_keys]

        # For each future hour, aggregate attention onto patch ids across all source timesteps.
        for h in range(fut_len):
            q = q_pos[h]
            row = attn_mean[q, k_pos]  # [N_spatial_keys]
            patch_sum = torch.zeros(model.num_patches, device=row.device)
            patch_sum.scatter_add_(0, patch_ids, row)
            patch_map = patch_sum.reshape(gh, gw).detach().cpu().numpy()
            for z in range(num_zones):
                zone_hour_patch_sum[z, h] += patch_map

        print(f"[{step_i + 1:>3}/{len(anchors)}] processed {energy_df['timestamp_utc'].iloc[t_idx].date()}")

    zone_hour_patch_avg = zone_hour_patch_sum / float(len(anchors))

    np.save(out_dir / "zone_hour_patch_attention.npy", zone_hour_patch_avg)

    # Zone-level average over all forecast hours.
    zone_patch_avg = zone_hour_patch_avg.mean(axis=1)
    top_df = topk_patch_table(zone_cols, zone_patch_avg, topk=args.topk)
    top_df.to_csv(out_dir / "zone_top_patches.csv", index=False)

    # Heatmaps: per-zone mean.
    for zi, zone in enumerate(zone_cols):
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        im = ax.imshow(zone_patch_avg[zi], cmap="viridis")
        ax.set_title(f"{zone}: mean future->spatial attention")
        ax.set_xlabel("Patch X")
        ax.set_ylabel("Patch Y")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"zone_mean_{zone}.png", dpi=180)
        plt.close(fig)

    # Hour-wise panels for each zone.
    for zi, zone in enumerate(zone_cols):
        fig, axes = plt.subplots(4, 6, figsize=(12, 8), constrained_layout=True)
        vmin = float(zone_hour_patch_avg[zi].min())
        vmax = float(zone_hour_patch_avg[zi].max())
        for h in range(fut_len):
            r, c = divmod(h, 6)
            ax = axes[r, c]
            im = ax.imshow(zone_hour_patch_avg[zi, h], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"+{h+1}h", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65)
        fig.suptitle(f"{zone}: future-hour attention maps", fontsize=12)
        fig.savefig(out_dir / f"zone_hours_{zone}.png", dpi=180)
        plt.close(fig)

    # Cross-zone difference map: each patch's dominant zone.
    dominant_zone = np.argmax(zone_patch_avg, axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(dominant_zone, cmap="tab20")
    plt.title("Dominant zone by patch (mean attention)")
    plt.xlabel("Patch X")
    plt.ylabel("Patch Y")
    cbar = plt.colorbar(ticks=np.arange(num_zones))
    cbar.ax.set_yticklabels(zone_cols)
    plt.tight_layout()
    plt.savefig(out_dir / "dominant_zone_by_patch.png", dpi=180)
    plt.close()

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "eval_year": args.eval_year,
        "n_days": int(len(anchors)),
        "zones": zone_cols,
        "patch_grid": [gh, gw],
        "notes": (
            "Attention is aggregated across transformer layers, heads, and all source timesteps' "
            "spatial tokens. In this architecture, the same future tab token drives all zone outputs; "
            "zone-specific maps are therefore identical and are repeated for convenient reporting."
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved attention analysis to: {out_dir}")


if __name__ == "__main__":
    main()
