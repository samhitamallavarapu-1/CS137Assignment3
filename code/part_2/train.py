#!/usr/bin/env python3
"""Training script for Assignment 3 Part 2 architecture-search models.

Designed for HPC runs where many experiments are launched from one file by
varying command-line arguments in SLURM scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from model import get_model

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


# -----------------------------
# Utilities
# -----------------------------

def parse_years(year_spec: str) -> List[int]:
    years: List[int] = []
    for chunk in year_spec.split(","):
        c = chunk.strip()
        if not c:
            continue
        if "-" in c:
            a, b = c.split("-", maxsplit=1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid year range: {c}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(c))
    if not years:
        raise ValueError("No years parsed from --years")
    return sorted(set(years))


def parse_year_set(year_spec: str) -> set[int]:
    if not year_spec.strip():
        return set()
    return set(parse_years(year_spec))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CropBox:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def infer_crop_box(
    h: int,
    w: int,
    crop_mode: str,
    crop_y0: int,
    crop_y1: int,
    crop_x0: int,
    crop_x1: int,
) -> CropBox:
    if crop_mode == "full":
        box = CropBox(0, h, 0, w)
    elif crop_mode == "new_england":
        # Tuned crop for this dataset (raw target on 450x449: y:105:386, x:180:374).
        # Kept as ratios so it scales if map size changes.
        box = CropBox(
            y0=int(round(0.233 * h)),
            y1=int(round(0.858 * h)),
            x0=int(round(0.401 * w)),
            x1=int(round(0.833 * w)),
        )
    else:
        box = CropBox(crop_y0, crop_y1, crop_x0, crop_x1)

    if not (0 <= box.y0 < box.y1 <= h and 0 <= box.x0 < box.x1 <= w):
        raise ValueError(
            f"Invalid crop box {(box.y0, box.y1, box.x0, box.x1)} for tensor shape {(h, w)}"
        )
    return box


# -----------------------------
# Data indexing / alignment
# -----------------------------

def load_energy_df(energy_dir: Path, years: Sequence[int]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for y in years:
        p = energy_dir / f"target_energy_zonal_{y}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, parse_dates=["timestamp_utc"]))
    if not dfs:
        raise FileNotFoundError("No energy CSVs found for requested years")
    out = pd.concat(dfs, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    return out


def build_weather_path_map(weather_dir: Path, years: Sequence[int]) -> Dict[pd.Timestamp, Path]:
    mapping: Dict[pd.Timestamp, Path] = {}
    for y in years:
        year_dir = weather_dir / str(y)
        if not year_dir.exists():
            continue
        for p in sorted(year_dir.glob("X_*.pt")):
            ts = pd.to_datetime(p.stem.split("_")[-1], format="%Y%m%d%H", utc=False)
            mapping[ts] = p
    return mapping


def make_calendar_features(ts: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(ts)
    hour = dt.dt.hour.values.astype(np.float32)
    dow = dt.dt.dayofweek.values.astype(np.float32)
    month = dt.dt.month.values.astype(np.float32)

    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)
    month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    weekend = (dow >= 5.0).astype(np.float32)

    return np.stack(
        [hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, weekend], axis=1
    ).astype(np.float32)


def build_aligned_table(
    data_dir: Path,
    years: Sequence[int],
) -> Tuple[pd.DataFrame, np.ndarray, List[Path], List[str]]:
    energy_df = load_energy_df(data_dir / "energy_demand_data", years)
    zone_cols = [c for c in energy_df.columns if c != "timestamp_utc"]

    weather_map = build_weather_path_map(data_dir / "weather_data", years)
    if not weather_map:
        raise FileNotFoundError("No weather .pt files found for requested years")

    keep_mask = energy_df["timestamp_utc"].isin(weather_map.keys())
    aligned = energy_df.loc[keep_mask].copy().sort_values("timestamp_utc").reset_index(drop=True)
    if aligned.empty:
        raise RuntimeError("No overlap between energy timestamps and weather files")

    weather_paths = [weather_map[ts] for ts in aligned["timestamp_utc"]]
    calendar = make_calendar_features(aligned["timestamp_utc"])
    return aligned, calendar, weather_paths, zone_cols


def contiguous_anchor_indices(hour_ints: np.ndarray, history_len: int, future_len: int) -> np.ndarray:
    n = len(hour_ints)
    if n < history_len + future_len:
        return np.array([], dtype=np.int64)

    gap_ok = (np.diff(hour_ints) == 1).astype(np.int32)
    required_gaps = history_len + future_len - 1

    anchors: List[int] = []
    for i in range(history_len, n - future_len + 1):
        left = i - history_len
        right = i + future_len - 1
        if gap_ok[left:right].sum() == required_gaps:
            anchors.append(i)
    return np.array(anchors, dtype=np.int64)


def _purge_train_near_val(train_anchors: np.ndarray, val_anchors: np.ndarray, purge_gap: int) -> np.ndarray:
    if purge_gap <= 0 or len(train_anchors) == 0 or len(val_anchors) == 0:
        return train_anchors

    val_sorted = np.sort(val_anchors)
    keep = np.ones(len(train_anchors), dtype=bool)
    for i, a in enumerate(train_anchors.tolist()):
        pos = np.searchsorted(val_sorted, a)
        d0 = abs(int(val_sorted[pos]) - a) if pos < len(val_sorted) else 10**12
        d1 = abs(a - int(val_sorted[pos - 1])) if pos > 0 else 10**12
        keep[i] = min(d0, d1) > purge_gap
    return train_anchors[keep]


def split_anchors(
    anchors: np.ndarray,
    timestamps: pd.Series,
    val_ratio: float,
    seed: int,
    history_len: int,
    future_len: int,
    split_mode: str,
    val_years: set[int],
    blocked_val_position: str,
    purge_gap: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    anchors = np.sort(anchors)
    if len(anchors) < 2:
        raise RuntimeError("Need at least two anchors to form train/val split")

    effective_gap = (history_len + future_len) if purge_gap < 0 else purge_gap
    n_val = max(1, int(round(val_ratio * len(anchors))))
    n_val = min(n_val, len(anchors) - 1)

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        shuffled = anchors.copy()
        rng.shuffle(shuffled)
        val_anchors = np.sort(shuffled[:n_val])
        train_anchors = np.sort(shuffled[n_val:])
    elif split_mode == "blocked":
        if blocked_val_position == "tail":
            start = len(anchors) - n_val
        elif blocked_val_position == "head":
            start = 0
        else:
            start = max(0, (len(anchors) - n_val) // 2)
        end = start + n_val
        val_anchors = anchors[start:end]
        train_anchors = np.concatenate([anchors[:start], anchors[end:]])
    else:
        if not val_years:
            raise ValueError("--val-years is required when --split-mode holdout_year")
        anchor_years = timestamps.iloc[anchors].dt.year.values
        val_mask = np.isin(anchor_years, np.array(sorted(val_years), dtype=np.int32))
        val_anchors = anchors[val_mask]
        train_anchors = anchors[~val_mask]
        if len(val_anchors) == 0 or len(train_anchors) == 0:
            raise RuntimeError(
                "holdout_year split produced empty train or val set; choose different --val-years"
            )

    train_anchors = _purge_train_near_val(np.sort(train_anchors), np.sort(val_anchors), effective_gap)

    if len(train_anchors) == 0:
        raise RuntimeError("All training anchors were removed by purge gap")
    if len(val_anchors) == 0:
        raise RuntimeError("No validation anchors after split")

    return np.sort(train_anchors), np.sort(val_anchors), effective_gap


# -----------------------------
# Weather preprocessing / stats
# -----------------------------

class WeatherCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def get(self, path: Path) -> torch.Tensor | None:
        key = str(path)
        if key not in self._cache:
            return None
        val = self._cache.pop(key)
        self._cache[key] = val
        return val

    def put(self, path: Path, tensor: torch.Tensor) -> None:
        key = str(path)
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = tensor
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


def load_weather_tensor(path: Path) -> torch.Tensor:
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor in {path}, got {type(x)}")
    if x.ndim != 3:
        raise ValueError(f"Expected rank-3 weather tensor in {path}, got shape {tuple(x.shape)}")
    return x.float()  # [H, W, C]


def preprocess_weather(x_hwc: torch.Tensor, crop_box: CropBox, out_h: int, out_w: int) -> torch.Tensor:
    x = x_hwc.permute(2, 0, 1).contiguous()  # [C,H,W]
    x = x[:, crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1]
    x = F.interpolate(x.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)
    return x


def unique_time_indices_from_anchors(anchors: np.ndarray, history_len: int, future_len: int) -> np.ndarray:
    touched = set()
    for a in anchors.tolist():
        touched.update(range(a - history_len, a + future_len))
    return np.array(sorted(touched), dtype=np.int64)


def compute_weather_stats(
    weather_paths: Sequence[Path],
    time_indices: np.ndarray,
    crop_box: CropBox,
    out_h: int,
    out_w: int,
    max_timestamps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(time_indices) == 0:
        raise ValueError("No time indices provided for weather normalization")

    if max_timestamps > 0 and len(time_indices) > max_timestamps:
        pick = np.random.choice(time_indices, size=max_timestamps, replace=False)
        time_indices = np.sort(pick)

    x0 = preprocess_weather(load_weather_tensor(weather_paths[int(time_indices[0])]), crop_box, out_h, out_w)
    c = x0.shape[0]

    sum_c = np.zeros(c, dtype=np.float64)
    sq_c = np.zeros(c, dtype=np.float64)
    count = 0

    for i in time_indices.tolist():
        x = preprocess_weather(load_weather_tensor(weather_paths[int(i)]), crop_box, out_h, out_w).numpy()
        flat = x.reshape(c, -1)
        sum_c += flat.sum(axis=1)
        sq_c += np.square(flat).sum(axis=1)
        count += flat.shape[1]

    mean = sum_c / max(count, 1)
    var = sq_c / max(count, 1) - np.square(mean)
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_energy_stats(energy: np.ndarray, time_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(time_indices) == 0:
        raise ValueError("No time indices provided for energy normalization")
    x = energy[time_indices]
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


# -----------------------------
# Dataset
# -----------------------------

class EnergyWeatherDataset(Dataset):
    def __init__(
        self,
        anchors: np.ndarray,
        history_len: int,
        future_len: int,
        weather_paths: Sequence[Path],
        energy_values: np.ndarray,
        calendar_features: np.ndarray,
        crop_box: CropBox,
        out_h: int,
        out_w: int,
        weather_mean: np.ndarray,
        weather_std: np.ndarray,
        normalize_weather: bool,
        energy_mean: np.ndarray,
        energy_std: np.ndarray,
        normalize_energy: bool,
        weather_cache_size: int,
    ) -> None:
        self.anchors = anchors
        self.history_len = history_len
        self.future_len = future_len
        self.weather_paths = weather_paths
        self.energy_values = energy_values
        self.calendar_features = calendar_features
        self.crop_box = crop_box
        self.out_h = out_h
        self.out_w = out_w
        self.normalize_weather = normalize_weather
        self.normalize_energy = normalize_energy

        self.weather_mean_t = torch.from_numpy(weather_mean).view(-1, 1, 1)
        self.weather_std_t = torch.from_numpy(weather_std).view(-1, 1, 1)
        self.energy_mean_t = torch.from_numpy(energy_mean).view(1, -1)
        self.energy_std_t = torch.from_numpy(energy_std).view(1, -1)

        self.cache = WeatherCache(max_size=weather_cache_size)

    def __len__(self) -> int:
        return len(self.anchors)

    def _get_weather(self, idx: int) -> torch.Tensor:
        p = self.weather_paths[idx]
        cached = self.cache.get(p)
        if cached is not None:
            return cached
        x = preprocess_weather(load_weather_tensor(p), self.crop_box, self.out_h, self.out_w)
        if self.normalize_weather:
            x = (x - self.weather_mean_t) / self.weather_std_t
        self.cache.put(p, x)
        return x

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        a = int(self.anchors[item])
        hs, he = a - self.history_len, a
        fs, fe = a, a + self.future_len

        hist_weather = torch.stack([self._get_weather(i) for i in range(hs, he)], dim=0)
        fut_weather = torch.stack([self._get_weather(i) for i in range(fs, fe)], dim=0)

        hist_demand = torch.from_numpy(self.energy_values[hs:he]).float()
        target = torch.from_numpy(self.energy_values[fs:fe]).float()
        hist_calendar = torch.from_numpy(self.calendar_features[hs:he]).float()
        fut_calendar = torch.from_numpy(self.calendar_features[fs:fe]).float()

        if self.normalize_energy:
            hist_demand = (hist_demand - self.energy_mean_t) / self.energy_std_t
            target = (target - self.energy_mean_t) / self.energy_std_t

        return {
            "hist_weather": hist_weather,
            "hist_demand": hist_demand,
            "hist_calendar": hist_calendar,
            "fut_weather": fut_weather,
            "fut_calendar": fut_calendar,
            "target": target,
        }


# -----------------------------
# Training / eval
# -----------------------------

def denorm_energy(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return x * std.view(1, 1, -1) + mean.view(1, 1, -1)


def mape_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    denom = torch.clamp(target.abs(), min=eps)
    return ((pred - target).abs() / denom).mean().item() * 100.0


def build_criterion(loss_name: str, huber_delta: float) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    return nn.HuberLoss(delta=huber_delta)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> LambdaLR | None:
    if scheduler_name == "none" or epochs <= 1:
        return None

    warmup = max(0, min(warmup_epochs, epochs - 1))
    min_ratio = max(0.0, min(1.0, min_lr / max(base_lr, 1e-12)))

    def lr_lambda(step_idx: int) -> float:
        if warmup > 0 and step_idx < warmup:
            return max(1e-8, float(step_idx + 1) / float(warmup))

        tail = epochs - warmup
        if tail <= 1:
            return 1.0

        progress = min(1.0, max(0.0, float(step_idx - warmup) / float(tail - 1)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    criterion: nn.Module,
    train_mode: bool,
    use_amp: bool,
    energy_mean_t: torch.Tensor,
    energy_std_t: torch.Tensor,
    normalize_energy: bool,
    grad_clip: float,
    epoch: int,
    num_epochs: int,
    use_tqdm: bool,
) -> Dict[str, float]:
    model.train(train_mode)

    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_n = 0

    phase = "train" if train_mode else "val"

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:03d}/{num_epochs} [{phase}]",
        leave=True,
        dynamic_ncols=False,
        ascii=True,
        mininterval=5.0,
        file=sys.stdout,
        disable=not use_tqdm,
    )

    for batch in iterator:
        hist_weather = batch["hist_weather"].to(device, non_blocking=True)
        hist_demand = batch["hist_demand"].to(device, non_blocking=True)
        hist_calendar = batch["hist_calendar"].to(device, non_blocking=True)
        fut_weather = batch["fut_weather"].to(device, non_blocking=True)
        fut_calendar = batch["fut_calendar"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                pred = model(hist_weather, hist_demand, hist_calendar, fut_weather, fut_calendar)
                loss = criterion(pred, target)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        with torch.no_grad():
            pred_real = denorm_energy(pred, energy_mean_t, energy_std_t, normalize_energy)
            tgt_real = denorm_energy(target, energy_mean_t, energy_std_t, normalize_energy)
            mae = (pred_real - tgt_real).abs().mean().item()
            mape = mape_percent(pred_real, tgt_real)

        bs = hist_weather.shape[0]
        total_loss += loss.item() * bs
        total_mae += mae * bs
        total_mape += mape * bs
        total_n += bs

        if use_tqdm and total_n > 0:
            iterator.set_postfix(loss=f"{(total_loss/total_n):.4f}", mape=f"{(total_mape/total_n):.2f}%")

    if total_n == 0:
        return {"loss": math.nan, "mae": math.nan, "mape": math.nan}

    return {
        "loss": total_loss / total_n,
        "mae": total_mae / total_n,
        "mape": total_mape / total_n,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Part 2 hierarchical encoder-decoder models")

    # Data / split
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part_2"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--years", type=str, default="2019-2023")
    parser.add_argument("--history-len", type=int, default=168)
    parser.add_argument("--future-len", type=int, default=24)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["random", "blocked", "holdout_year"], default="blocked")
    parser.add_argument("--blocked-val-position", choices=["head", "middle", "tail"], default="tail")
    parser.add_argument("--val-years", type=str, default="")
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=-1,
        help="Exclude train anchors within this many indices of any val anchor; default is history+future",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    # Spatial preprocessing
    parser.add_argument("--crop-mode", choices=["new_england", "full", "custom"], default="new_england")
    parser.add_argument("--crop-y0", type=int, default=0)
    parser.add_argument("--crop-y1", type=int, default=450)
    parser.add_argument("--crop-x0", type=int, default=0)
    parser.add_argument("--crop-x1", type=int, default=449)
    parser.add_argument("--downsample-h", type=int, default=96)
    parser.add_argument("--downsample-w", type=int, default=96)

    # Normalization
    parser.add_argument("--normalize-weather", action="store_true", default=True)
    parser.add_argument("--no-normalize-weather", action="store_false", dest="normalize_weather")
    parser.add_argument("--normalize-energy", action="store_true", default=True)
    parser.add_argument("--no-normalize-energy", action="store_false", dest="normalize_energy")
    parser.add_argument("--norm-max-timestamps", type=int, default=2000)

    # Model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-hidden-dim", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--patch-grid-h", type=int, default=10)
    parser.add_argument("--patch-grid-w", type=int, default=10)
    parser.add_argument("--arch-variant", choices=["no_cnn", "residual_cnn"], default="no_cnn")
    parser.add_argument("--use-weather-stats", action="store_true", default=True)
    parser.add_argument("--no-weather-stats", action="store_false", dest="use_weather_stats")
    parser.add_argument(
        "--tokenizer-chunk-steps",
        type=int,
        default=4,
        help="Time-chunk size used inside residual_cnn tokenizer to reduce peak VRAM. 0 disables chunking.",
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # Runtime
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--weather-cache-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", action="store_false", dest="cudnn_benchmark")
    parser.add_argument("--resume-from", type=str, default="", help="Checkpoint path, or 'auto' for run_dir/checkpoints/last.pt")
    parser.add_argument("--tqdm", action="store_true", default=True, help="Show tqdm batch progress bars")
    parser.add_argument("--no-tqdm", action="store_false", dest="tqdm")

    args = parser.parse_args()

    set_seed(args.seed)

    years = parse_years(args.years)
    val_years = parse_year_set(args.val_years)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    print("=" * 80)
    print("Part 2 training")
    print(f"run_dir: {run_dir}")
    print(f"years: {years}")
    print(f"device: {device}")
    print("=" * 80)

    aligned_df, calendar_feats, weather_paths, zone_cols = build_aligned_table(args.data_dir, years)
    energy_values = aligned_df[zone_cols].values.astype(np.float32)
    hour_ints = aligned_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)

    anchors = contiguous_anchor_indices(hour_ints, args.history_len, args.future_len)
    if len(anchors) == 0:
        raise RuntimeError("No valid samples after contiguity filtering")

    train_anchors, val_anchors, effective_gap = split_anchors(
        anchors=anchors,
        timestamps=aligned_df["timestamp_utc"],
        val_ratio=args.val_ratio,
        seed=args.seed,
        history_len=args.history_len,
        future_len=args.future_len,
        split_mode=args.split_mode,
        val_years=val_years,
        blocked_val_position=args.blocked_val_position,
        purge_gap=args.purge_gap,
    )

    rng = np.random.default_rng(args.seed)
    if args.max_train_samples > 0 and len(train_anchors) > args.max_train_samples:
        train_anchors = np.sort(rng.choice(train_anchors, size=args.max_train_samples, replace=False))
    if args.max_val_samples > 0 and len(val_anchors) > args.max_val_samples:
        val_anchors = np.sort(rng.choice(val_anchors, size=args.max_val_samples, replace=False))

    probe = load_weather_tensor(weather_paths[0])
    crop_box = infer_crop_box(
        h=probe.shape[0],
        w=probe.shape[1],
        crop_mode=args.crop_mode,
        crop_y0=args.crop_y0,
        crop_y1=args.crop_y1,
        crop_x0=args.crop_x0,
        crop_x1=args.crop_x1,
    )
    weather_channels = int(probe.shape[2])

    print(
        f"aligned_hours={len(aligned_df)} total_samples={len(anchors)} "
        f"train={len(train_anchors)} val={len(val_anchors)}"
    )
    print(
        f"split_mode={args.split_mode} blocked_val_position={args.blocked_val_position} "
        f"purge_gap={effective_gap}"
    )
    print(
        f"crop_box=(y:{crop_box.y0}:{crop_box.y1}, x:{crop_box.x0}:{crop_box.x1}) "
        f"-> crop_hw=({crop_box.h},{crop_box.w}) downsample_hw=({args.downsample_h},{args.downsample_w})"
    )

    train_time_indices = unique_time_indices_from_anchors(train_anchors, args.history_len, args.future_len)

    if args.normalize_weather:
        w_mean, w_std = compute_weather_stats(
            weather_paths=weather_paths,
            time_indices=train_time_indices,
            crop_box=crop_box,
            out_h=args.downsample_h,
            out_w=args.downsample_w,
            max_timestamps=args.norm_max_timestamps,
        )
    else:
        w_mean = np.zeros(weather_channels, dtype=np.float32)
        w_std = np.ones(weather_channels, dtype=np.float32)

    if args.normalize_energy:
        e_mean, e_std = compute_energy_stats(energy_values, train_time_indices)
    else:
        e_mean = np.zeros(len(zone_cols), dtype=np.float32)
        e_std = np.ones(len(zone_cols), dtype=np.float32)

    train_ds = EnergyWeatherDataset(
        anchors=train_anchors,
        history_len=args.history_len,
        future_len=args.future_len,
        weather_paths=weather_paths,
        energy_values=energy_values,
        calendar_features=calendar_feats,
        crop_box=crop_box,
        out_h=args.downsample_h,
        out_w=args.downsample_w,
        weather_mean=w_mean,
        weather_std=w_std,
        normalize_weather=args.normalize_weather,
        energy_mean=e_mean,
        energy_std=e_std,
        normalize_energy=args.normalize_energy,
        weather_cache_size=args.weather_cache_size,
    )
    val_ds = EnergyWeatherDataset(
        anchors=val_anchors,
        history_len=args.history_len,
        future_len=args.future_len,
        weather_paths=weather_paths,
        energy_values=energy_values,
        calendar_features=calendar_feats,
        crop_box=crop_box,
        out_h=args.downsample_h,
        out_w=args.downsample_w,
        weather_mean=w_mean,
        weather_std=w_std,
        normalize_weather=args.normalize_weather,
        energy_mean=e_mean,
        energy_std=e_std,
        normalize_energy=args.normalize_energy,
        weather_cache_size=max(32, args.weather_cache_size // 2),
    )

    loader_kwargs: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = get_model(
        weather_channels=weather_channels,
        num_zones=len(zone_cols),
        calendar_dim=calendar_feats.shape[1],
        future_steps=args.future_len,
        history_len=args.history_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_layers=args.decoder_layers,
        spatial_layers=args.spatial_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        cnn_hidden_dim=args.cnn_hidden_dim,
        residual_blocks=args.residual_blocks,
        patch_grid_h=args.patch_grid_h,
        patch_grid_w=args.patch_grid_w,
        arch_variant=args.arch_variant,
        use_weather_stats=args.use_weather_stats,
        crop_mode=args.crop_mode,
        crop_y0=args.crop_y0,
        crop_y1=args.crop_y1,
        crop_x0=args.crop_x0,
        crop_x1=args.crop_x1,
        downsample_h=args.downsample_h,
        downsample_w=args.downsample_w,
        tokenizer_chunk_steps=args.tokenizer_chunk_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = build_criterion(args.loss, args.huber_delta)

    energy_mean_t = torch.from_numpy(e_mean).to(device)
    energy_std_t = torch.from_numpy(e_std).to(device)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    with open(run_dir / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "zone_columns": zone_cols,
                "weather_mean": w_mean.tolist(),
                "weather_std": w_std.tolist(),
                "energy_mean": e_mean.tolist(),
                "energy_std": e_std.tolist(),
                "crop_box": {"y0": crop_box.y0, "y1": crop_box.y1, "x0": crop_box.x0, "x1": crop_box.x1},
                "downsample_hw": [args.downsample_h, args.downsample_w],
                "split": {
                    "mode": args.split_mode,
                    "blocked_val_position": args.blocked_val_position,
                    "val_years": sorted(val_years),
                    "purge_gap": effective_gap,
                },
            },
            f,
            indent=2,
        )

    pd.DataFrame({"anchor_index": train_anchors}).to_csv(run_dir / "train_indices.csv", index=False)
    pd.DataFrame({"anchor_index": val_anchors}).to_csv(run_dir / "val_indices.csv", index=False)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    start_epoch = 1

    resume_path: Path | None = None
    if args.resume_from:
        if args.resume_from.strip().lower() == "auto":
            candidate = ckpt_dir / "last.pt"
            if candidate.exists():
                resume_path = candidate
        else:
            candidate = Path(args.resume_from)
            if candidate.exists():
                resume_path = candidate

    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        try:
            prev_df = pd.read_csv(metrics_path)
            if not prev_df.empty:
                history = prev_df.to_dict("records")
                best_val = float(prev_df["val_loss"].min())
        except Exception:
            history = []

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and scheduler is not None and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and ckpt["scaler_state"] is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])
        if not history and "history" in ckpt and isinstance(ckpt["history"], list):
            history = ckpt["history"]

    if start_epoch > args.epochs:
        print(f"Run already reached epoch {start_epoch - 1} which is >= --epochs {args.epochs}; exiting.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            train_mode=True,
            use_amp=args.amp,
            energy_mean_t=energy_mean_t,
            energy_std_t=energy_std_t,
            normalize_energy=args.normalize_energy,
            grad_clip=args.grad_clip,
            epoch=epoch,
            num_epochs=args.epochs,
            use_tqdm=args.tqdm,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            train_mode=False,
            use_amp=args.amp,
            energy_mean_t=energy_mean_t,
            energy_std_t=energy_std_t,
            normalize_energy=args.normalize_energy,
            grad_clip=args.grad_clip,
            epoch=epoch,
            num_epochs=args.epochs,
            use_tqdm=args.tqdm,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_mape": train_metrics["mape"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_mape": val_metrics["mape"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} | "
            f"train_mape={row['train_mape']:.3f}% val_mape={row['val_mape']:.3f}%"
        )

        if scheduler is not None:
            scheduler.step()

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "args": vars(args),
            "zone_cols": zone_cols,
            "weather_mean": w_mean,
            "weather_std": w_std,
            "energy_mean": e_mean,
            "energy_std": e_std,
            "crop_box": (crop_box.y0, crop_box.y1, crop_box.x0, crop_box.x1),
            "downsample_hw": (args.downsample_h, args.downsample_w),
            "metrics": row,
            "best_val": best_val,
            "history": history,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, ckpt_dir / "best.pt")

        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    print("Training complete.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
