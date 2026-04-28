#!/usr/bin/env python3
"""Part 3 diagnostics for Part 1 model attention and zone attribution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
PART1_DIR = REPO_ROOT / "code" / "part_1"
if str(PART1_DIR) not in sys.path:
    sys.path.insert(0, str(PART1_DIR))

from model import get_model  # type: ignore  # noqa: E402
from train import (  # type: ignore  # noqa: E402
    EnergyWeatherDataset,
    build_aligned_table,
    infer_crop_box,
    load_weather_tensor,
    parse_years,
)

from attention_tools import (  # noqa: E402
    extract_attention_batch,
    reshape_patch_maps,
    summarize_top_patches,
)


def _as_path(v: str | Path) -> Path:
    return v if isinstance(v, Path) else Path(v)


def _resolve_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if checkpoint_name in {"best", "last"}:
        ckpt = ckpt_dir / f"{checkpoint_name}.pt"
        if ckpt.exists():
            return ckpt
    if checkpoint_name.endswith(".pt"):
        ckpt = _as_path(checkpoint_name)
        if ckpt.exists():
            return ckpt
    fallback_best = ckpt_dir / "best.pt"
    fallback_last = ckpt_dir / "last.pt"
    if fallback_best.exists():
        return fallback_best
    if fallback_last.exists():
        return fallback_last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_anchor_indices(run_dir: Path, split: str, max_samples: int, seed: int) -> np.ndarray:
    train_path = run_dir / "train_indices.csv"
    val_path = run_dir / "val_indices.csv"

    if split == "train":
        anchors = pd.read_csv(train_path)["anchor_index"].to_numpy(dtype=np.int64)
    elif split == "val":
        anchors = pd.read_csv(val_path)["anchor_index"].to_numpy(dtype=np.int64)
    else:
        train_a = pd.read_csv(train_path)["anchor_index"].to_numpy(dtype=np.int64)
        val_a = pd.read_csv(val_path)["anchor_index"].to_numpy(dtype=np.int64)
        anchors = np.sort(np.unique(np.concatenate([train_a, val_a])))

    if max_samples > 0 and len(anchors) > max_samples:
        rng = np.random.default_rng(seed)
        anchors = np.sort(rng.choice(anchors, size=max_samples, replace=False))
    return anchors


def _build_dataset(
    data_dir: Path,
    run_dir: Path,
    split: str,
    max_samples: int,
    seed: int,
    weather_cache_size: int,
) -> tuple[EnergyWeatherDataset, pd.DataFrame, List[str], Dict, Dict]:
    cfg = _load_json(run_dir / "config.json")
    norm = _load_json(run_dir / "normalization.json")

    years = parse_years(str(cfg["years"]))
    aligned_df, calendar_feats, weather_paths, zone_cols = build_aligned_table(data_dir, years)
    energy_values = aligned_df[zone_cols].values.astype(np.float32)
    anchors = _load_anchor_indices(run_dir, split=split, max_samples=max_samples, seed=seed)

    probe = load_weather_tensor(weather_paths[0])
    crop_cfg = norm.get("crop_box", {})
    crop_box = infer_crop_box(
        h=probe.shape[0],
        w=probe.shape[1],
        crop_mode=str(cfg.get("crop_mode", "new_england")),
        crop_y0=int(crop_cfg.get("y0", cfg.get("crop_y0", 0))),
        crop_y1=int(crop_cfg.get("y1", cfg.get("crop_y1", probe.shape[0]))),
        crop_x0=int(crop_cfg.get("x0", cfg.get("crop_x0", 0))),
        crop_x1=int(crop_cfg.get("x1", cfg.get("crop_x1", probe.shape[1]))),
    )

    w_mean = np.asarray(norm["weather_mean"], dtype=np.float32)
    w_std = np.asarray(norm["weather_std"], dtype=np.float32)
    e_mean = np.asarray(norm["energy_mean"], dtype=np.float32)
    e_std = np.asarray(norm["energy_std"], dtype=np.float32)
    downsample_h, downsample_w = [int(v) for v in norm.get("downsample_hw", [cfg["downsample_h"], cfg["downsample_w"]])]

    ds = EnergyWeatherDataset(
        anchors=anchors,
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        weather_paths=weather_paths,
        energy_values=energy_values,
        calendar_features=calendar_feats,
        crop_box=crop_box,
        out_h=downsample_h,
        out_w=downsample_w,
        weather_mean=w_mean,
        weather_std=w_std,
        normalize_weather=bool(cfg.get("normalize_weather", True)),
        energy_mean=e_mean,
        energy_std=e_std,
        normalize_energy=bool(cfg.get("normalize_energy", True)),
        weather_cache_size=weather_cache_size,
    )
    return ds, aligned_df, zone_cols, cfg, norm


def _build_model(cfg: Dict, norm: Dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    weather_channels = len(norm["weather_mean"])
    model = get_model(
        weather_channels=weather_channels,
        num_zones=len(norm["zone_columns"]),
        calendar_dim=7,
        future_steps=int(cfg["future_len"]),
        history_len=int(cfg["history_len"]),
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
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state: {checkpoint_path}")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def _make_output_dir(base_output_dir: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_output_dir / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Part 1 future->spatial attention maps for Part 3 diagnosis")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--part1-run-dir", type=Path, required=True, help="Path like outputs/part_1/<run_name>")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_3"))
    parser.add_argument("--checkpoint", type=str, default="best", help="'best', 'last', or explicit .pt path")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--weather-cache-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--compute-zone-grad-maps", action="store_true", default=True)
    parser.add_argument("--no-zone-grad-maps", action="store_false", dest="compute_zone_grad_maps")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    part1_run_dir = args.part1_run_dir.resolve()
    ckpt_path = _resolve_checkpoint(part1_run_dir, args.checkpoint)

    ds, aligned_df, zone_cols, cfg, norm = _build_dataset(
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

    loader = DataLoader(
        ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    patch_grid_h = int(cfg["patch_grid_h"])
    patch_grid_w = int(cfg["patch_grid_w"])

    anchors = ds.anchors.astype(np.int64)
    anchor_ts = aligned_df.iloc[anchors]["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    all_future_history: List[torch.Tensor] = []
    all_history_spatial: List[torch.Tensor] = []
    all_future_spatial: List[torch.Tensor] = []
    all_zone_spatial: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []

    print("=" * 80)
    print("Part 3 attention extraction")
    print(f"part1_run_dir: {part1_run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"split={args.split} samples={len(ds)} batch_size={args.batch_size} device={device}")
    print("=" * 80)

    for batch in loader:
        hist_weather = batch["hist_weather"].to(device, non_blocking=True)
        hist_demand = batch["hist_demand"].to(device, non_blocking=True)
        hist_calendar = batch["hist_calendar"].to(device, non_blocking=True)
        fut_weather = batch["fut_weather"].to(device, non_blocking=True)
        fut_calendar = batch["fut_calendar"].to(device, non_blocking=True)

        out = extract_attention_batch(
            model=model,
            hist_weather=hist_weather,
            hist_demand=hist_demand,
            hist_calendar=hist_calendar,
            fut_weather=fut_weather,
            fut_calendar=fut_calendar,
            compute_zone_grad_maps=args.compute_zone_grad_maps,
        )

        all_future_history.append(out.future_to_history.detach().cpu())
        all_history_spatial.append(out.history_to_spatial.detach().cpu())
        all_future_spatial.append(out.future_to_spatial.detach().cpu())
        all_preds.append(out.predictions.detach().cpu())
        if out.zone_grad_spatial is not None:
            all_zone_spatial.append(out.zone_grad_spatial.detach().cpu())

    future_history = torch.cat(all_future_history, dim=0)  # [N, Tf, Th]
    history_spatial = torch.cat(all_history_spatial, dim=0)  # [N, Th, P]
    future_spatial = torch.cat(all_future_spatial, dim=0)  # [N, Tf, P]
    preds = torch.cat(all_preds, dim=0)  # [N, Tf, Z]

    future_spatial_2d = reshape_patch_maps(future_spatial, patch_grid_h, patch_grid_w)
    history_spatial_2d = reshape_patch_maps(history_spatial, patch_grid_h, patch_grid_w)

    zone_spatial_2d = None
    if all_zone_spatial:
        zone_spatial = torch.cat(all_zone_spatial, dim=0)  # [N, Tf, Z, P]
        zone_spatial_2d = reshape_patch_maps(zone_spatial, patch_grid_h, patch_grid_w)

    run_name = f"{part1_run_dir.name}_{args.split}"
    out_dir = _make_output_dir(args.output_dir, run_name)

    npz_payload = {
        "anchor_index": anchors,
        "anchor_timestamp_utc": np.asarray(anchor_ts, dtype=object),
        "zone_names": np.asarray(zone_cols, dtype=object),
        "future_to_history": future_history.numpy(),
        "history_to_spatial": history_spatial_2d.numpy(),
        "future_to_spatial": future_spatial_2d.numpy(),
        "predictions": preds.numpy(),
        "patch_grid_h": np.array([patch_grid_h], dtype=np.int64),
        "patch_grid_w": np.array([patch_grid_w], dtype=np.int64),
    }
    if zone_spatial_2d is not None:
        npz_payload["zone_grad_spatial"] = zone_spatial_2d.numpy()

    np.savez_compressed(out_dir / "attention_maps.npz", **npz_payload)

    mean_future = future_spatial_2d.mean(dim=0).numpy()  # [Tf, Ph, Pw]
    rows = []
    for fut_idx in range(mean_future.shape[0]):
        for py in range(patch_grid_h):
            for px in range(patch_grid_w):
                rows.append(
                    {
                        "future_hour_idx": fut_idx,
                        "patch_y": py,
                        "patch_x": px,
                        "mean_attention": float(mean_future[fut_idx, py, px]),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "mean_future_to_spatial.csv", index=False)

    if zone_spatial_2d is not None:
        top_rows = summarize_top_patches(
            zone_maps=zone_spatial_2d,
            zone_names=zone_cols,
            patch_grid_h=patch_grid_h,
            patch_grid_w=patch_grid_w,
            top_k=max(1, args.top_k),
        )
        pd.DataFrame(top_rows).to_csv(out_dir / "top_patches_by_zone.csv", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_local": datetime.now().isoformat(timespec="seconds"),
                "part1_run_dir": str(part1_run_dir),
                "checkpoint": str(ckpt_path),
                "split": args.split,
                "num_samples": int(len(ds)),
                "batch_size": int(args.batch_size),
                "device": str(device),
                "future_len": int(cfg["future_len"]),
                "history_len": int(cfg["history_len"]),
                "patch_grid_h": patch_grid_h,
                "patch_grid_w": patch_grid_w,
                "zone_names": zone_cols,
                "compute_zone_grad_maps": bool(args.compute_zone_grad_maps),
                "notes": [
                    "future_to_spatial is composed from transformer self-attention-derived future->history and history->patch maps.",
                    "zone_grad_spatial reweights attention with per-zone gradient sensitivity to encoded history tokens.",
                ],
            },
            f,
            indent=2,
        )

    # Quick scalar summary for logs.
    mean_attn_entropy = float(
        -(
            future_spatial.reshape(future_spatial.shape[0], future_spatial.shape[1], -1).clamp_min(1e-8)
            * future_spatial.reshape(future_spatial.shape[0], future_spatial.shape[1], -1).clamp_min(1e-8).log()
        ).sum(dim=-1).mean().item()
    )
    print(f"Saved diagnostics to: {out_dir}")
    print(f"mean future->spatial entropy: {mean_attn_entropy:.4f}")
    print("Done.")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
