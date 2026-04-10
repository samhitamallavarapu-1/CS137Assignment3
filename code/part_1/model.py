"""Baseline CNN-Transformer patch model for CS137 Assignment 3 (Part 1).

This module implements the sequence-to-sequence baseline described in the assignment:
- CNN downsampling of weather maps into spatial patch tokens for all timesteps
- tabular token per timestep (historical: demand+calendar, future: masked-demand+calendar)
- learnable spatial positional embeddings + sinusoidal timestep encodings
- Transformer encoder over the unified token sequence
- MLP head to predict per-zone demand for future timesteps
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    weather_channels: int
    num_zones: int
    calendar_dim: int
    future_steps: int = 24
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    cnn_hidden_dim: int = 64
    patch_grid_h: int = 10
    patch_grid_w: int = 10


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
    """Maps weather tensors to patch tokens of shape [B, T, P, D]."""

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
            raise ValueError(f"weather must be [B, T, Cw, H, W], got shape={tuple(weather.shape)}")
        bsz, steps, _, _, _ = weather.shape
        x = weather.reshape(bsz * steps, *weather.shape[2:])
        x = self.cnn(x)
        x = self.pool(x)
        x = self.proj(x)
        # [B*T, D, Gh, Gw] -> [B, T, P, D]
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

        # Learned replacement for unknown future demand values.
        self.future_demand_mask = nn.Parameter(torch.zeros(config.num_zones))

        # Shared spatial positional embedding for all timesteps.
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

    def _temporal_encoding(self, total_steps: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal encoding with shape [1, T, 1, D]."""
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
        """Predict future zonal demand.

        Args:
            hist_weather: [B, Th, Cw, H, W]
            hist_demand: [B, Th, Z]
            hist_calendar: [B, Th, Cc]
            fut_weather: [B, Tf, Cw, H, W]
            fut_calendar: [B, Tf, Cc]

        Returns:
            preds: [B, Tf, Z]
        """
        if hist_weather.ndim != 5 or fut_weather.ndim != 5:
            raise ValueError("hist_weather and fut_weather must both be rank-5 tensors")
        if hist_demand.ndim != 3 or hist_calendar.ndim != 3 or fut_calendar.ndim != 3:
            raise ValueError("hist_demand/hist_calendar/fut_calendar must all be rank-3 tensors")

        bsz, hist_steps = hist_weather.shape[:2]
        fut_steps = fut_weather.shape[1]

        if hist_demand.shape[:2] != (bsz, hist_steps):
            raise ValueError("hist_demand must align with [B, Th] of hist_weather")
        if hist_calendar.shape[:2] != (bsz, hist_steps):
            raise ValueError("hist_calendar must align with [B, Th] of hist_weather")
        if fut_calendar.shape[:2] != (bsz, fut_steps):
            raise ValueError("fut_calendar must align with [B, Tf] of fut_weather")

        if hist_demand.shape[-1] != self.config.num_zones:
            raise ValueError(
                f"hist_demand last dim must be num_zones={self.config.num_zones}, "
                f"got {hist_demand.shape[-1]}"
            )
        if hist_calendar.shape[-1] != self.config.calendar_dim or fut_calendar.shape[-1] != self.config.calendar_dim:
            raise ValueError(
                f"calendar last dim must be calendar_dim={self.config.calendar_dim}"
            )

        if self.config.future_steps > 0 and fut_steps != self.config.future_steps:
            raise ValueError(
                f"expected Tf={self.config.future_steps} from config, but got Tf={fut_steps}"
            )

        # Spatial tokens for all timesteps.
        hist_spatial = self.patch_tokenizer(hist_weather) + self.spatial_pos_embed
        fut_spatial = self.patch_tokenizer(fut_weather) + self.spatial_pos_embed

        # Historical tabular token: demand + calendar.
        hist_tab_in = torch.cat([hist_demand, hist_calendar], dim=-1)
        hist_tab = self.hist_tabular_embed(hist_tab_in).unsqueeze(2)

        # Future tabular token: masked-demand vector + future calendar.
        masked_future_demand = self.future_demand_mask.view(1, 1, -1).expand(bsz, fut_steps, -1)
        fut_tab_in = torch.cat([masked_future_demand, fut_calendar], dim=-1)
        fut_tab = self.fut_tabular_embed(fut_tab_in).unsqueeze(2)

        # Per-timestep groups of [P spatial + 1 tabular] tokens.
        hist_group = torch.cat([hist_spatial, hist_tab], dim=2)  # [B, Th, P+1, D]
        fut_group = torch.cat([fut_spatial, fut_tab], dim=2)     # [B, Tf, P+1, D]
        all_group = torch.cat([hist_group, fut_group], dim=1)    # [B, Ttotal, P+1, D]

        # Add temporal encoding to every token in each timestep group.
        total_steps = hist_steps + fut_steps
        all_group = all_group + self._temporal_encoding(total_steps, all_group.device)

        # Flatten to Transformer sequence [B, Ttotal*(P+1), D].
        tokens_per_step = self.num_patches + 1
        seq = all_group.reshape(bsz, total_steps * tokens_per_step, self.config.d_model)
        seq = self.dropout(seq)

        encoded = self.transformer(seq)

        # Extract tabular token states for future timesteps.
        base = self.num_patches
        fut_t_idx = torch.arange(hist_steps, total_steps, device=encoded.device)
        fut_tab_pos = fut_t_idx * tokens_per_step + base
        fut_states = encoded[:, fut_tab_pos, :]  # [B, Tf, D]

        preds = self.pred_head(fut_states)  # [B, Tf, Z]
        return preds


def get_model(
    weather_channels: int,
    num_zones: int,
    calendar_dim: int,
    future_steps: int = 24,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    ff_dim: int = 1024,
    dropout: float = 0.1,
    cnn_hidden_dim: int = 64,
    patch_grid_h: int = 10,
    patch_grid_w: int = 10,
) -> BaselineCNNTransformerPatch:
    """Factory required by assignment evaluation code.

    Example:
        model = get_model(weather_channels=8, num_zones=8, calendar_dim=10)
    """
    cfg = ModelConfig(
        weather_channels=weather_channels,
        num_zones=num_zones,
        calendar_dim=calendar_dim,
        future_steps=future_steps,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        cnn_hidden_dim=cnn_hidden_dim,
        patch_grid_h=patch_grid_h,
        patch_grid_w=patch_grid_w,
    )
    return BaselineCNNTransformerPatch(cfg)
