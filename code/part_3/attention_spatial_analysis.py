#!/usr/bin/env python3
"""Spatial attention extraction and analysis for Assignment 3 Part 3.

Exports future-token -> spatial-token attention as:
  [sample, zone, forecast_hour, patch_y, patch_x]

Also saves patch coordinates, corresponding weather fields for same samples,
summary metrics, and zone-wise plots over forecast horizon.
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
import torch.nn.functional as F

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

try:
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False


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


def build_eval_anchors(energy_df: pd.DataFrame, eval_year: int, history_len: int, future_len: int, n_days: int) -> np.ndarray:
    midnight_mask = (
        (energy_df["timestamp_utc"].dt.year == eval_year)
        & (energy_df["timestamp_utc"].dt.hour == 0)
    )
    anchors = np.where(midnight_mask)[0]
    anchors = anchors[(anchors >= history_len) & (anchors + future_len <= len(energy_df))]
    if len(anchors) == 0:
        raise RuntimeError(f"No valid anchors found for year {eval_year}")
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


def build_seq_from_adapted(model, hist_weather, hist_demand, hist_cal, fut_weather, fut_cal):
    bsz, hist_steps = hist_weather.shape[:2]
    fut_steps = fut_weather.shape[1]

    hist_spatial = model.patch_tokenizer(hist_weather) + model.spatial_pos_embed
    fut_spatial = model.patch_tokenizer(fut_weather) + model.spatial_pos_embed

    hist_tab = model.hist_tabular_embed(torch.cat([hist_demand, hist_cal], dim=-1)).unsqueeze(2)
    masked_future_demand = model.future_demand_mask.view(1, 1, -1).expand(bsz, fut_steps, -1)
    fut_tab = model.fut_tabular_embed(torch.cat([masked_future_demand, fut_cal], dim=-1)).unsqueeze(2)

    hist_group = torch.cat([hist_spatial, hist_tab], dim=2)
    fut_group = torch.cat([fut_spatial, fut_tab], dim=2)
    all_group = torch.cat([hist_group, fut_group], dim=1)

    total_steps = hist_steps + fut_steps
    all_group = all_group + model._temporal_encoding(total_steps, all_group.device)

    tokens_per_step = model.num_patches + 1
    seq = all_group.reshape(bsz, total_steps * tokens_per_step, model.config.d_model)
    seq = model.dropout(seq)
    return seq, hist_steps, fut_steps, tokens_per_step


def extract_layer0_attention_rows(model, seq, q_pos):
    """Return attention rows [Q,S] averaged over heads for selected queries.

    Uses only layer 0 for memory-safe extraction.
    """
    layer0 = model.transformer.layers[0]
    x = seq
    x_in = layer0.norm1(x) if layer0.norm_first else x

    mha = layer0.self_attn
    bsz, seq_len, d_model = x_in.shape
    if bsz != 1:
        raise ValueError("Expected batch size 1 in analysis loop")

    num_heads = mha.num_heads
    head_dim = d_model // num_heads

    qkv = F.linear(x_in, mha.in_proj_weight, mha.in_proj_bias)
    q_all, k_all, _ = qkv.split(d_model, dim=-1)

    q_sel = q_all[:, q_pos, :].view(1, len(q_pos), num_heads, head_dim).transpose(1, 2)
    k_all = k_all.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

    scale = float(head_dim) ** -0.5
    logits = torch.matmul(q_sel, k_all.transpose(-2, -1)) * scale
    attn = torch.softmax(logits, dim=-1).mean(dim=1).squeeze(0)  # [Q,S]
    return attn


def parse_projection(projection_str: str):
    # Example: "LambertConformal(central_lon=262.5, central_lat=38.5)"
    if not projection_str.startswith("LambertConformal"):
        return None
    inside = projection_str.split("(", 1)[1].rsplit(")", 1)[0]
    parts = [p.strip() for p in inside.split(",") if p.strip()]
    vals = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            vals[k.strip()] = float(v.strip())
    if "central_lon" not in vals or "central_lat" not in vals:
        return None
    central_lon = vals["central_lon"]
    if central_lon > 180.0:
        central_lon = central_lon - 360.0
    central_lat = vals["central_lat"]
    return central_lon, central_lat


def build_patch_coords(metadata_path: Path, crop_box: Tuple[int, int, int, int], gh: int, gw: int) -> pd.DataFrame:
    m = torch.load(metadata_path, map_location="cpu", weights_only=False)
    grid_x = np.asarray(m["grid_x"], dtype=np.float64)  # W
    grid_y = np.asarray(m["grid_y"], dtype=np.float64)  # H
    proj_str = str(m.get("projection", ""))

    y0, y1, x0, x1 = crop_box
    gx = grid_x[x0:x1]
    gy = grid_y[y0:y1]

    px_edges = np.linspace(0, len(gx), gw + 1)
    py_edges = np.linspace(0, len(gy), gh + 1)

    rows = []

    transformer = None
    if HAS_PYPROJ:
        parsed = parse_projection(proj_str)
        if parsed is not None:
            lon0, lat0 = parsed
            try:
                src = CRS.from_proj4(
                    f"+proj=lcc +lat_0={lat0} +lon_0={lon0} +lat_1={lat0} +lat_2={lat0} +datum=WGS84 +units=m +no_defs"
                )
                dst = CRS.from_epsg(4326)
                transformer = Transformer.from_crs(src, dst, always_xy=True)
            except Exception:
                transformer = None

    for py in range(gh):
        ys = int(np.floor(py_edges[py]))
        ye = int(np.floor(py_edges[py + 1]))
        if ye <= ys:
            ye = min(ys + 1, len(gy))
        y_center_idx = (ys + ye - 1) / 2.0
        y_center_raw = y0 + y_center_idx
        y_center_proj = float(np.mean(gy[ys:ye]))

        for px in range(gw):
            xs = int(np.floor(px_edges[px]))
            xe = int(np.floor(px_edges[px + 1]))
            if xe <= xs:
                xe = min(xs + 1, len(gx))
            x_center_idx = (xs + xe - 1) / 2.0
            x_center_raw = x0 + x_center_idx
            x_center_proj = float(np.mean(gx[xs:xe]))

            lat = np.nan
            lon = np.nan
            if transformer is not None:
                try:
                    lon_v, lat_v = transformer.transform(x_center_proj, y_center_proj)
                    lat = float(lat_v)
                    lon = float(lon_v)
                except Exception:
                    pass

            rows.append(
                {
                    "patch_y": py,
                    "patch_x": px,
                    "raw_y_center": float(y_center_raw),
                    "raw_x_center": float(x_center_raw),
                    "proj_y_center": float(y_center_proj),
                    "proj_x_center": float(x_center_proj),
                    "lat": lat,
                    "lon": lon,
                }
            )

    return pd.DataFrame(rows)


def attention_entropy(map2d: np.ndarray) -> float:
    p = map2d.reshape(-1).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def pairwise_zone_cosine(zone_maps: np.ndarray, zone_names: List[str]) -> pd.DataFrame:
    # zone_maps: [Z,H,W]
    z = zone_maps.shape[0]
    flat = zone_maps.reshape(z, -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    flat_n = flat / norms
    sim = flat_n @ flat_n.T

    rows = []
    for i in range(z):
        for j in range(z):
            rows.append({"zone_i": zone_names[i], "zone_j": zone_names[j], "cosine_similarity": float(sim[i, j])})
    return pd.DataFrame(rows)


def save_zone_hour_plots(out_dir: Path, zone_names: List[str], zone_hour_map: np.ndarray):
    if not HAS_MPL:
        return

    # zone_hour_map: [Z,Tf,H,W]
    selected = [0, 5, 11, 17, 23]  # +1,+6,+12,+18,+24

    for zi, zone in enumerate(zone_names):
        # five fixed horizons
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
        vmin = float(zone_hour_map[zi].min())
        vmax = float(zone_hour_map[zi].max())
        for k, h in enumerate(selected):
            ax = axes[k]
            im = ax.imshow(zone_hour_map[zi, h], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{zone} +{h+1}h")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        fig.savefig(out_dir / f"zone_{zone}_hours_1_6_12_18_24.png", dpi=180)
        plt.close(fig)

        # sequential panel over full horizon
        fig, axes = plt.subplots(4, 6, figsize=(13, 8), constrained_layout=True)
        vmin = float(zone_hour_map[zi].min())
        vmax = float(zone_hour_map[zi].max())
        for h in range(zone_hour_map.shape[1]):
            r, c = divmod(h, 6)
            ax = axes[r, c]
            im = ax.imshow(zone_hour_map[zi, h], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"+{h+1}h", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65)
        fig.suptitle(f"{zone} attention shift over +1h...+24h", fontsize=12)
        fig.savefig(out_dir / f"zone_{zone}_sequential_panel.png", dpi=180)
        plt.close(fig)

        if HAS_IMAGEIO:
            try:
                frames = []
                vmin = float(zone_hour_map[zi].min())
                vmax = float(zone_hour_map[zi].max())
                for h in range(zone_hour_map.shape[1]):
                    fig, ax = plt.subplots(figsize=(4.5, 4.0))
                    ax.imshow(zone_hour_map[zi, h], cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"{zone} +{h+1}h")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.canvas.draw()
                    # Backend-safe pixel extraction for modern Matplotlib.
                    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    rgb = rgba[:, :, :3].copy()
                    frames.append(rgb)
                    plt.close(fig)
                imageio.mimsave(out_dir / f"zone_{zone}_attention.gif", frames, fps=2)
            except Exception as e:
                print(f"Warning: GIF generation skipped for zone={zone}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Future-token to spatial-token attention extraction")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to outputs/part_1/<run_name>")
    parser.add_argument("--part1-model-path", type=Path, default=Path("code/part_1/model.py"))
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--eval-year", type=int, default=2022)
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--zones",
        type=str,
        default="",
        help="Comma-separated zone names to keep (default: all zones in data)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing attention_analysis_part3 directory before writing new outputs",
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
    gh = int(cfg["patch_grid_h"])
    gw = int(cfg["patch_grid_w"])

    model_module = load_part1_model_module(args.part1_model_path.resolve())
    get_model = model_module.get_model

    energy_df, zone_cols_all = load_energy_df(data_dir)
    if args.zones.strip():
        requested = [z.strip() for z in args.zones.split(",") if z.strip()]
        missing = [z for z in requested if z not in zone_cols_all]
        if missing:
            raise ValueError(f"Requested zones not found: {missing}. Available: {zone_cols_all}")
        zone_cols = requested
    else:
        zone_cols = zone_cols_all
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
        patch_grid_h=gh,
        patch_grid_w=gw,
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

    anchors = build_eval_anchors(energy_df, args.eval_year, history_len, future_len, args.n_days)
    n_samples = len(anchors)

    out_dir = run_dir / "attention_analysis_part3"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using zones ({num_zones}): {zone_cols}")

    # Requested tensor: [sample, zone, forecast_hour, patch_y, patch_x]
    attn_tensor = np.zeros((n_samples, num_zones, future_len, gh, gw), dtype=np.float32)
    # Weather patch means for corresponding samples: [sample, forecast_hour, channel, patch_y, patch_x]
    weather_patch_tensor = np.zeros((n_samples, future_len, 7, gh, gw), dtype=np.float32)

    sample_meta_rows = []
    weather_cache: Dict[int, torch.Tensor] = {}

    for si, t_idx in enumerate(anchors):
        hist_slice = slice(t_idx - history_len, t_idx)
        fut_slice = slice(t_idx, t_idx + future_len)

        hist_hours = all_hours[hist_slice]
        fut_hours = all_hours[fut_slice]

        hist_weather_raw = torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in hist_hours]).unsqueeze(0).to(device)
        fut_weather_raw = torch.stack([load_weather_tensor(data_dir, int(h), weather_cache) for h in fut_hours]).unsqueeze(0).to(device)
        hist_energy = torch.from_numpy(energy_vals[hist_slice]).unsqueeze(0).to(device)
        fut_time = torch.from_numpy(fut_hours).to(torch.int64).unsqueeze(0).to(device)

        with torch.no_grad():
            hist_weather, hist_demand, hist_cal, fut_weather, fut_cal = model.adapt_inputs(
                hist_weather_raw, hist_energy, fut_weather_raw, fut_time
            )
            seq, hist_steps, fut_steps, tps = build_seq_from_adapted(
                model, hist_weather, hist_demand, hist_cal, fut_weather, fut_cal
            )

            fut_t_idx = torch.arange(hist_steps, hist_steps + fut_steps, device=device)
            q_pos = fut_t_idx * tps + model.num_patches
            q_rows = extract_layer0_attention_rows(model, seq, q_pos)  # [Tf,S]

            seq_idx = torch.arange(q_rows.shape[1], device=device)
            spatial_mask = (seq_idx % tps) < model.num_patches
            k_pos = seq_idx[spatial_mask]
            patch_ids = (k_pos % tps).to(torch.long)

            # Base map per forecast hour (shared by architecture). Replicate across zones.
            shared_maps = np.zeros((future_len, gh, gw), dtype=np.float32)
            for h in range(future_len):
                row = q_rows[h, k_pos]
                patch_sum = torch.zeros(gh * gw, device=device)
                patch_sum.scatter_add_(0, patch_ids, row)
                shared_maps[h] = patch_sum.reshape(gh, gw).cpu().numpy().astype(np.float32)
            attn_tensor[si] = np.repeat(shared_maps[None, :, :, :], repeats=num_zones, axis=0)

            # Save weather fields aligned to these samples as patch means.
            # fut_weather shape: [1,Tf,C,down_h,down_w]
            fw = fut_weather[0]  # [Tf,C,H,W]
            fw_patch = F.adaptive_avg_pool2d(fw.reshape(future_len * 7, fw.shape[-2], fw.shape[-1]).unsqueeze(1), (gh, gw))
            fw_patch = fw_patch.squeeze(1).reshape(future_len, 7, gh, gw)
            weather_patch_tensor[si] = fw_patch.cpu().numpy().astype(np.float32)

        sample_meta_rows.append(
            {
                "sample_index": si,
                "anchor_row_index": int(t_idx),
                "anchor_timestamp_utc": str(energy_df["timestamp_utc"].iloc[t_idx]),
                "future_start_hour_int": int(fut_hours[0]),
                "future_end_hour_int": int(fut_hours[-1]),
            }
        )

        print(f"[{si+1:>3}/{n_samples}] processed {energy_df['timestamp_utc'].iloc[t_idx].date()}")

    # Save core tensors
    np.save(out_dir / "attention_sample_zone_hour_patch.npy", attn_tensor)
    np.save(out_dir / "weather_sample_hour_channel_patch.npy", weather_patch_tensor)
    pd.DataFrame(sample_meta_rows).to_csv(out_dir / "sample_index_metadata.csv", index=False)

    # Patch coordinates (raw grid + projected + lat/lon if conversion works)
    dummy_h, dummy_w = 450, 449
    y0, y1, x0, x1 = model._resolve_crop_box(dummy_h, dummy_w)
    patch_coords = build_patch_coords(data_dir / "weather_data" / "metadata.pt", (y0, y1, x0, x1), gh, gw)
    patch_coords.to_csv(out_dir / "patch_coordinates.csv", index=False)

    # Summary metrics
    # Zone attention mean across samples: [Z,Tf,H,W]
    zone_hour_mean = attn_tensor.mean(axis=0)

    metric_rows = []
    topk_rows = []
    for zi, zone in enumerate(zone_cols):
        for h in range(future_len):
            m = zone_hour_mean[zi, h].astype(np.float64)
            flat = m.reshape(-1)
            idx = np.argsort(flat)[::-1][: args.topk]
            total = flat.sum() + 1e-12

            py, px = np.unravel_index(np.argmax(m), m.shape)
            com_y = float((m * np.arange(gh)[:, None]).sum() / total)
            com_x = float((m * np.arange(gw)[None, :]).sum() / total)
            ent = attention_entropy(m)

            metric_rows.append(
                {
                    "zone": zone,
                    "forecast_hour": h + 1,
                    "max_patch_y": int(py),
                    "max_patch_x": int(px),
                    "center_of_mass_y": com_y,
                    "center_of_mass_x": com_x,
                    "entropy": ent,
                }
            )

            for rank, j in enumerate(idx, start=1):
                topk_rows.append(
                    {
                        "zone": zone,
                        "forecast_hour": h + 1,
                        "rank": rank,
                        "patch_y": int(j // gw),
                        "patch_x": int(j % gw),
                        "attention": float(flat[j]),
                        "attention_norm": float(flat[j] / total),
                    }
                )

    pd.DataFrame(metric_rows).to_csv(out_dir / "zone_hour_attention_metrics.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(out_dir / "zone_hour_topk_patches.csv", index=False)

    # Pairwise zone attention similarity using mean map over forecast horizon
    zone_mean_map = zone_hour_mean.mean(axis=1)  # [Z,H,W]
    sim_df = pairwise_zone_cosine(zone_mean_map, zone_cols)
    sim_df.to_csv(out_dir / "pairwise_zone_attention_similarity.csv", index=False)

    # Plots
    save_zone_hour_plots(out_dir, zone_cols, zone_hour_mean)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "eval_year": int(args.eval_year),
        "n_days": int(args.n_days),
        "num_samples": int(n_samples),
        "zones": zone_cols,
        "future_len": future_len,
        "patch_grid": [gh, gw],
        "attention_tensor_shape": list(attn_tensor.shape),
        "weather_tensor_shape": list(weather_patch_tensor.shape),
        "has_matplotlib": HAS_MPL,
        "has_imageio": HAS_IMAGEIO,
        "has_pyproj": HAS_PYPROJ,
        "attention_layer_used": 0,
        "notes": [
            "This model's future query token stream is shared before zone projection.",
            "Therefore raw future->spatial attention is identical across zones for a given sample/hour.",
            "Zone axis is preserved without averaging by replicating shared maps per zone as requested.",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
