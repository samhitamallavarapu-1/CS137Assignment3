"""Baseline CNN-Transformer patch model for CS137 Assignment 3 (Part 1).

Supports two usage paths:
1) Training path: call forward(...) directly with preprocessed tensors.
2) Evaluation path: call adapt_inputs(raw_inputs) then forward(*adapted).
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(
        self,
        weather_channels: int,
        num_zones: int,
        calendar_dim: int,
        future_steps: int = 24,
        history_len: int = 168,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        cnn_hidden_dim: int = 64,
        patch_grid_h: int = 10,
        patch_grid_w: int = 10,
        crop_mode: str = "new_england",
        crop_y0: int = 0,
        crop_y1: int = 450,
        crop_x0: int = 0,
        crop_x1: int = 449,
        downsample_h: int = 96,
        downsample_w: int = 96,
    ) -> None:
        self.weather_channels = int(weather_channels)
        self.num_zones = int(num_zones)
        self.calendar_dim = int(calendar_dim)
        self.future_steps = int(future_steps)
        self.history_len = int(history_len)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.cnn_hidden_dim = int(cnn_hidden_dim)
        self.patch_grid_h = int(patch_grid_h)
        self.patch_grid_w = int(patch_grid_w)
        self.crop_mode = str(crop_mode)
        self.crop_y0 = int(crop_y0)
        self.crop_y1 = int(crop_y1)
        self.crop_x0 = int(crop_x0)
        self.crop_x1 = int(crop_x1)
        self.downsample_h = int(downsample_h)
        self.downsample_w = int(downsample_w)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WeatherPatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        hidden_dim: int,
        patch_grid_h: int,
        patch_grid_w: int,
    ) -> None:
        super().__init__()
        self.patch_grid_h = patch_grid_h
        self.patch_grid_w = patch_grid_w

        self.cnn = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim * 2),
            ConvBlock(hidden_dim * 2, hidden_dim * 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((patch_grid_h, patch_grid_w))
        self.proj = nn.Conv2d(hidden_dim * 2, d_model, kernel_size=1)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        # weather: [B, T, Cw, H, W]
        if weather.ndim != 5:
            raise ValueError(f"weather must be [B,T,C,H,W], got {tuple(weather.shape)}")
        bsz, steps, _, _, _ = weather.shape
        x = weather.reshape(bsz * steps, *weather.shape[2:])
        x = self.cnn(x)
        x = self.pool(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(bsz, steps, self.patch_grid_h * self.patch_grid_w, -1)
        return x


class BaselineCNNTransformerPatch(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_tokenizer = WeatherPatchTokenizer(
            in_channels=config.weather_channels,
            d_model=config.d_model,
            hidden_dim=config.cnn_hidden_dim,
            patch_grid_h=config.patch_grid_h,
            patch_grid_w=config.patch_grid_w,
        )

        self.num_patches = config.patch_grid_h * config.patch_grid_w

        self.hist_tabular_embed = nn.Linear(config.num_zones + config.calendar_dim, config.d_model)
        self.fut_tabular_embed = nn.Linear(config.num_zones + config.calendar_dim, config.d_model)

        self.future_demand_mask = nn.Parameter(torch.zeros(config.num_zones))
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, config.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.dropout = nn.Dropout(config.dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_zones),
        )

        # Optional buffers if caller wants to set normalization stats.
        self.register_buffer("weather_mean", None, persistent=False)
        self.register_buffer("weather_std", None, persistent=False)
        self.register_buffer("energy_mean", None, persistent=False)
        self.register_buffer("energy_std", None, persistent=False)
        self._denormalize_output = False

    # -----------------------------
    # Eval adapter
    # -----------------------------
    def _resolve_crop_box(self, h: int, w: int) -> tuple[int, int, int, int]:
        if self.config.crop_mode == "full":
            y0, y1, x0, x1 = 0, h, 0, w
        elif self.config.crop_mode == "new_england":
            # Match train.py tuned defaults.
            y0 = int(round(0.233 * h))
            y1 = int(round(0.858 * h))
            x0 = int(round(0.401 * w))
            x1 = int(round(0.833 * w))
        else:
            y0, y1, x0, x1 = self.config.crop_y0, self.config.crop_y1, self.config.crop_x0, self.config.crop_x1

        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise ValueError(f"Invalid crop box {(y0,y1,x0,x1)} for shape {(h,w)}")
        return y0, y1, x0, x1

    def _calendar_from_hours(self, hours_since_epoch: torch.Tensor) -> torch.Tensor:
        if hours_since_epoch.ndim != 2:
            raise ValueError("Expected [B,T] hours tensor")

        arr = hours_since_epoch.detach().cpu().numpy().astype(np.int64)
        bsz, steps = arr.shape
        dt = pd.to_datetime(arr.reshape(-1), unit="h")

        hour = dt.hour.to_numpy(dtype=np.float32).reshape(bsz, steps)
        dow = dt.dayofweek.to_numpy(dtype=np.float32).reshape(bsz, steps)
        month = dt.month.to_numpy(dtype=np.float32).reshape(bsz, steps)

        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
        dow_cos = np.cos(2.0 * np.pi * dow / 7.0)
        month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        weekend = (dow >= 5.0).astype(np.float32)

        feats = np.stack(
            [hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, weekend],
            axis=-1,
        ).astype(np.float32)
        return torch.from_numpy(feats).to(hours_since_epoch.device)

    def _preprocess_raw_weather(self, weather: torch.Tensor) -> torch.Tensor:
        if weather.ndim != 5:
            raise ValueError(f"weather must be rank-5, got {tuple(weather.shape)}")

        if weather.shape[-1] == self.config.weather_channels:
            weather = weather.permute(0, 1, 4, 2, 3).contiguous()  # [B,T,H,W,C] -> [B,T,C,H,W]
        elif weather.shape[2] == self.config.weather_channels:
            weather = weather.contiguous()  # already [B,T,C,H,W]
        else:
            raise ValueError(
                f"Cannot infer weather channel axis from {tuple(weather.shape)} "
                f"for C={self.config.weather_channels}"
            )

        bsz, steps, ch, h, w = weather.shape
        y0, y1, x0, x1 = self._resolve_crop_box(h, w)
        weather = weather[:, :, :, y0:y1, x0:x1]

        weather = weather.reshape(bsz * steps, ch, y1 - y0, x1 - x0)
        weather = F.interpolate(
            weather,
            size=(self.config.downsample_h, self.config.downsample_w),
            mode="bilinear",
            align_corners=False,
        )
        weather = weather.reshape(bsz, steps, ch, self.config.downsample_h, self.config.downsample_w)

        if self.weather_mean is not None and self.weather_std is not None:
            weather = (weather - self.weather_mean.view(1, 1, -1, 1, 1)) / self.weather_std.view(1, 1, -1, 1, 1)

        return weather

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adapt evaluator inputs to this model's forward signature."""
        if history_energy.ndim != 3 or future_time.ndim != 2:
            raise ValueError("history_energy must be [B,Th,Z], future_time must be [B,Tf]")

        bsz, hist_steps, _ = history_energy.shape

        hist_weather_proc = self._preprocess_raw_weather(history_weather)
        fut_weather_proc = self._preprocess_raw_weather(future_weather)

        # Reconstruct history timestamps from first future timestamp.
        first_future = future_time[:, :1].to(torch.int64)
        if hist_steps == self.config.history_len:
            offsets = torch.arange(self.config.history_len, 0, -1, device=future_time.device, dtype=torch.int64).view(1, -1)
        else:
            offsets = torch.arange(hist_steps, 0, -1, device=future_time.device, dtype=torch.int64).view(1, -1)
        history_time = first_future - offsets

        hist_calendar = self._calendar_from_hours(history_time)
        fut_calendar = self._calendar_from_hours(future_time.to(torch.int64))

        hist_demand = history_energy.float()
        if self.energy_mean is not None and self.energy_std is not None:
            hist_demand = (hist_demand - self.energy_mean.view(1, 1, -1)) / self.energy_std.view(1, 1, -1)

        return (
            hist_weather_proc.float(),
            hist_demand,
            hist_calendar.float(),
            fut_weather_proc.float(),
            fut_calendar.float(),
        )

    # -----------------------------
    # Core forward
    # -----------------------------
    def _temporal_encoding(self, total_steps: int, device: torch.device) -> torch.Tensor:
        d_model = self.config.d_model
        position = torch.arange(total_steps, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(total_steps, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).unsqueeze(2)

    def forward(
        self,
        hist_weather: torch.Tensor,
        hist_demand: torch.Tensor,
        hist_calendar: torch.Tensor,
        fut_weather: torch.Tensor,
        fut_calendar: torch.Tensor,
    ) -> torch.Tensor:
        if hist_weather.ndim != 5 or fut_weather.ndim != 5:
            raise ValueError("hist_weather and fut_weather must both be rank-5")
        if hist_demand.ndim != 3 or hist_calendar.ndim != 3 or fut_calendar.ndim != 3:
            raise ValueError("hist_demand/hist_calendar/fut_calendar must all be rank-3")

        bsz, hist_steps = hist_weather.shape[:2]
        fut_steps = fut_weather.shape[1]

        if hist_demand.shape[:2] != (bsz, hist_steps):
            raise ValueError("hist_demand must align with hist_weather")
        if hist_calendar.shape[:2] != (bsz, hist_steps):
            raise ValueError("hist_calendar must align with hist_weather")
        if fut_calendar.shape[:2] != (bsz, fut_steps):
            raise ValueError("fut_calendar must align with fut_weather")

        if hist_demand.shape[-1] != self.config.num_zones:
            raise ValueError(f"Expected num_zones={self.config.num_zones}, got {hist_demand.shape[-1]}")
        if hist_calendar.shape[-1] != self.config.calendar_dim or fut_calendar.shape[-1] != self.config.calendar_dim:
            raise ValueError(f"Expected calendar_dim={self.config.calendar_dim}")

        if self.config.future_steps > 0 and fut_steps != self.config.future_steps:
            raise ValueError(f"Expected future_steps={self.config.future_steps}, got {fut_steps}")

        hist_spatial = self.patch_tokenizer(hist_weather) + self.spatial_pos_embed
        fut_spatial = self.patch_tokenizer(fut_weather) + self.spatial_pos_embed

        hist_tab = self.hist_tabular_embed(torch.cat([hist_demand, hist_calendar], dim=-1)).unsqueeze(2)
        masked_future_demand = self.future_demand_mask.view(1, 1, -1).expand(bsz, fut_steps, -1)
        fut_tab = self.fut_tabular_embed(torch.cat([masked_future_demand, fut_calendar], dim=-1)).unsqueeze(2)

        hist_group = torch.cat([hist_spatial, hist_tab], dim=2)
        fut_group = torch.cat([fut_spatial, fut_tab], dim=2)
        all_group = torch.cat([hist_group, fut_group], dim=1)

        total_steps = hist_steps + fut_steps
        all_group = all_group + self._temporal_encoding(total_steps, all_group.device)

        tokens_per_step = self.num_patches + 1
        seq = all_group.reshape(bsz, total_steps * tokens_per_step, self.config.d_model)
        seq = self.dropout(seq)

        encoded = self.transformer(seq)

        fut_t_idx = torch.arange(hist_steps, total_steps, device=encoded.device)
        fut_tab_pos = fut_t_idx * tokens_per_step + self.num_patches
        fut_states = encoded[:, fut_tab_pos, :]

        preds = self.pred_head(fut_states)
        if self._denormalize_output and self.energy_mean is not None and self.energy_std is not None:
            preds = preds * self.energy_std.view(1, 1, -1) + self.energy_mean.view(1, 1, -1)
        return preds


# -----------------------------
# Factory
# -----------------------------

def _resolve_checkpoint_path(base_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(base_dir, "best.pt"),
        os.path.join(base_dir, "last.pt"),
        os.path.join(base_dir, "checkpoints", "best.pt"),
        os.path.join(base_dir, "checkpoints", "last.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_checkpoint(checkpoint_path: str) -> Tuple[dict, dict]:
    try:
        with torch.serialization.safe_globals([pathlib.PosixPath]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except AttributeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any(isinstance(v, dict) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    return state_dict, checkpoint


def _load_optional_tensor(checkpoint: dict, key: str) -> torch.Tensor | None:
    if key not in checkpoint:
        return None
    value = checkpoint[key]
    tensor = torch.as_tensor(value).float()
    return tensor


def _model_from_metadata(metadata: Dict[str, Any]) -> BaselineCNNTransformerPatch:
    cfg = ModelConfig(
        weather_channels=int(metadata.get("n_weather_vars", 7)),
        num_zones=int(metadata["n_zones"]),
        calendar_dim=7,
        history_len=int(metadata.get("history_len", 168)),
        future_steps=int(metadata.get("future_len", 24)),
    )
    model = BaselineCNNTransformerPatch(cfg)

    checkpoint_path = _resolve_checkpoint_path(os.path.dirname(__file__))
    if checkpoint_path is not None:
        state_dict, checkpoint = _load_checkpoint(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        model._denormalize_output = True
        print(f"Loaded checkpoint from {checkpoint_path}")

        energy_mean = _load_optional_tensor(checkpoint, "energy_mean")
        energy_std = _load_optional_tensor(checkpoint, "energy_std")
        weather_mean = _load_optional_tensor(checkpoint, "weather_mean")
        weather_std = _load_optional_tensor(checkpoint, "weather_std")

        if energy_mean is not None and energy_std is not None:
            model.energy_mean = energy_mean.view(1, 1, -1)
            model.energy_std = energy_std.view(1, 1, -1)

        if weather_mean is not None and weather_std is not None:
            model.weather_mean = weather_mean.view(-1, 1, 1)
            model.weather_std = weather_std.view(-1, 1, 1)

    return model


def get_model(*args, **kwargs) -> BaselineCNNTransformerPatch:
    """Support both evaluator and training calls.

    Evaluator style:
        get_model(metadata: dict)

    Training style:
        get_model(weather_channels=..., num_zones=..., calendar_dim=..., ...)
    """
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return _model_from_metadata(args[0])

    cfg = ModelConfig(
        weather_channels=kwargs["weather_channels"],
        num_zones=kwargs["num_zones"],
        calendar_dim=kwargs["calendar_dim"],
        future_steps=kwargs.get("future_steps", 24),
        history_len=kwargs.get("history_len", 168),
        d_model=kwargs.get("d_model", 256),
        num_heads=kwargs.get("num_heads", 8),
        num_layers=kwargs.get("num_layers", 4),
        ff_dim=kwargs.get("ff_dim", 1024),
        dropout=kwargs.get("dropout", 0.1),
        cnn_hidden_dim=kwargs.get("cnn_hidden_dim", 64),
        patch_grid_h=kwargs.get("patch_grid_h", 10),
        patch_grid_w=kwargs.get("patch_grid_w", 10),
        crop_mode=kwargs.get("crop_mode", "new_england"),
        crop_y0=kwargs.get("crop_y0", 0),
        crop_y1=kwargs.get("crop_y1", 450),
        crop_x0=kwargs.get("crop_x0", 0),
        crop_x1=kwargs.get("crop_x1", 449),
        downsample_h=kwargs.get("downsample_h", 96),
        downsample_w=kwargs.get("downsample_w", 96),
    )
    return BaselineCNNTransformerPatch(cfg)
