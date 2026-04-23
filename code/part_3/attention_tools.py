"""Attention and attribution helpers for Part 3 diagnostics.

This module replays the Part 2 model in analysis mode so we can expose:
1) Future-token -> history-hour decoder cross-attention.
2) History-hour -> spatial-patch summarizer attention.
3) Composed future-token -> spatial-patch attention maps.
4) Zone-conditioned maps using gradient sensitivity to encoder memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import torch


@dataclass
class AttentionBatchOutput:
    future_to_history: torch.Tensor  # [B, Tf, Th]
    history_to_spatial: torch.Tensor  # [B, Th, P]
    future_to_spatial: torch.Tensor  # [B, Tf, P]
    predictions: torch.Tensor  # [B, Tf, Z]
    zone_grad_spatial: torch.Tensor | None  # [B, Tf, Z, P]


def _summarize_hour_tokens_with_weights(
    model: torch.nn.Module,
    patch_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hour tokens and query->patch attention from HourlyWeatherSummarizer."""
    bsz, steps, num_patches, d_model = patch_tokens.shape
    x = patch_tokens.reshape(bsz * steps, num_patches, d_model)
    x = model.weather_summarizer.spatial_encoder(x)

    q = model.weather_summarizer.query.expand(bsz * steps, -1, -1)
    hour_token, attn_weights = model.weather_summarizer.attn(
        q,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    hour_token = model.weather_summarizer.norm(hour_token).reshape(bsz, steps, d_model)
    attn_weights = attn_weights.reshape(
        bsz,
        steps,
        model.config.num_heads,
        num_patches,
    )
    return hour_token, attn_weights


def _decoder_with_cross_attention_weights(
    model: torch.nn.Module,
    dec_in: torch.Tensor,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Run TransformerDecoder while extracting per-layer cross-attention weights."""
    x = dec_in
    cross_weights: List[torch.Tensor] = []

    for layer in model.temporal_decoder.layers:
        if layer.norm_first:
            x = x + layer._sa_block(layer.norm1(x), None, None, False)

            x_norm = layer.norm2(x)
            mha_out, mha_w = layer.multihead_attn(
                x_norm,
                memory,
                memory,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=False,
                is_causal=False,
            )
            x = x + layer.dropout2(mha_out)
            cross_weights.append(mha_w)
            x = x + layer._ff_block(layer.norm3(x))
        else:
            x2 = layer._sa_block(x, None, None, False)
            x = layer.norm1(x + x2)

            mha_out, mha_w = layer.multihead_attn(
                x,
                memory,
                memory,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=False,
                is_causal=False,
            )
            x = layer.norm2(x + layer.dropout2(mha_out))
            cross_weights.append(mha_w)
            x = layer.norm3(x + layer._ff_block(x))

    if model.temporal_decoder.norm is not None:
        x = model.temporal_decoder.norm(x)
    return x, cross_weights


def extract_attention_batch(
    model: torch.nn.Module,
    hist_weather: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_calendar: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_calendar: torch.Tensor,
    compute_zone_grad_maps: bool = True,
) -> AttentionBatchOutput:
    """Extract composed attention maps and optional zone-conditioned maps."""
    bsz = hist_weather.shape[0]
    th = hist_weather.shape[1]
    tf = fut_weather.shape[1]
    num_patches = model.config.patch_grid_h * model.config.patch_grid_w
    num_zones = model.config.num_zones

    hist_patch = model.patch_tokenizer(hist_weather) + model.spatial_pos_embed
    fut_patch = model.patch_tokenizer(fut_weather) + model.spatial_pos_embed

    hist_hour, hist_patch_attn = _summarize_hour_tokens_with_weights(model, hist_patch)
    fut_hour, _ = _summarize_hour_tokens_with_weights(model, fut_patch)

    if model.weather_stats_proj is not None:
        hist_hour = hist_hour + model.weather_stats_proj(model._weather_stats(hist_weather))
        fut_hour = fut_hour + model.weather_stats_proj(model._weather_stats(fut_weather))

    hist_tab = model.hist_tabular_embed(torch.cat([hist_demand, hist_calendar], dim=-1))
    masked_future_demand = model.future_demand_mask.view(1, 1, -1).expand(bsz, tf, -1)
    fut_tab = model.fut_tabular_embed(torch.cat([masked_future_demand, fut_calendar], dim=-1))

    enc_in = hist_hour + hist_tab
    dec_in = fut_hour + fut_tab
    enc_in = model.dropout(enc_in + model._temporal_encoding(th, enc_in.device))
    dec_in = model.dropout(dec_in + model._temporal_encoding(tf, dec_in.device))

    memory = model.temporal_encoder(enc_in)
    if compute_zone_grad_maps:
        memory.retain_grad()

    decoded, cross_w_by_layer = _decoder_with_cross_attention_weights(model, dec_in, memory)
    preds = model.pred_head(decoded)

    # Aggregate over attention heads then over decoder layers.
    # cross_w_by_layer[i]: [B, H, Tf, Th]
    cross_stack = torch.stack(cross_w_by_layer, dim=0)  # [L, B, H, Tf, Th]
    future_to_history = cross_stack.mean(dim=2).mean(dim=0)  # [B, Tf, Th]

    # hist_patch_attn: [B, Th, H, P]
    history_to_spatial = hist_patch_attn.mean(dim=2)  # [B, Th, P]

    # Compose attention chain: (future->history) @ (history->patch) -> future->patch
    future_to_spatial = torch.matmul(future_to_history, history_to_spatial)  # [B, Tf, P]

    zone_grad_spatial = None
    if compute_zone_grad_maps:
        zone_grad_spatial = torch.zeros(
            bsz,
            tf,
            num_zones,
            num_patches,
            device=preds.device,
            dtype=preds.dtype,
        )
        eps = 1e-8
        for fut_idx in range(tf):
            for zone_idx in range(num_zones):
                score = preds[:, fut_idx, zone_idx].sum()
                grad_mem = torch.autograd.grad(
                    score,
                    memory,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]  # [B, Th, D]

                hour_importance = grad_mem.abs().mean(dim=-1)  # [B, Th]
                # Keep attention grounding while making map zone-conditional.
                hour_importance = hour_importance * future_to_history[:, fut_idx, :]
                hour_importance = hour_importance / (hour_importance.sum(dim=-1, keepdim=True) + eps)
                zone_grad_spatial[:, fut_idx, zone_idx, :] = torch.matmul(
                    hour_importance.unsqueeze(1),
                    history_to_spatial,
                ).squeeze(1)

    return AttentionBatchOutput(
        future_to_history=future_to_history,
        history_to_spatial=history_to_spatial,
        future_to_spatial=future_to_spatial,
        predictions=preds,
        zone_grad_spatial=zone_grad_spatial,
    )


def reshape_patch_maps(
    patch_maps: torch.Tensor,
    patch_grid_h: int,
    patch_grid_w: int,
) -> torch.Tensor:
    """Reshape [..., P] patch vectors to [..., patch_grid_h, patch_grid_w]."""
    if patch_maps.shape[-1] != patch_grid_h * patch_grid_w:
        raise ValueError(
            f"Patch dimension mismatch: got {patch_maps.shape[-1]}, expected {patch_grid_h * patch_grid_w}"
        )
    return patch_maps.reshape(*patch_maps.shape[:-1], patch_grid_h, patch_grid_w)


def summarize_top_patches(
    zone_maps: torch.Tensor,
    zone_names: List[str],
    patch_grid_h: int,
    patch_grid_w: int,
    top_k: int,
) -> List[Dict[str, float | int | str]]:
    """Return top-k patch cells by mean zone-conditioned score."""
    if zone_maps.ndim != 5:
        raise ValueError(f"Expected zone_maps [N,Tf,Z,Ph,Pw], got {tuple(zone_maps.shape)}")

    n_samples, fut_steps, num_zones, _, _ = zone_maps.shape
    rows: List[Dict[str, float | int | str]] = []
    flat = zone_maps.mean(dim=(0, 1)).reshape(num_zones, patch_grid_h * patch_grid_w)

    for z in range(num_zones):
        vals = flat[z]
        k = min(top_k, vals.numel())
        scores, idx = torch.topk(vals, k=k, largest=True, sorted=True)
        for rank in range(k):
            patch_idx = int(idx[rank].item())
            py = patch_idx // patch_grid_w
            px = patch_idx % patch_grid_w
            rows.append(
                {
                    "zone": zone_names[z],
                    "rank": rank + 1,
                    "patch_y": py,
                    "patch_x": px,
                    "mean_score": float(scores[rank].item()),
                    "n_samples": int(n_samples),
                    "future_steps": int(fut_steps),
                }
            )
    return rows


def extract_layerwise_future_to_spatial_batch(
    model: torch.nn.Module,
    hist_weather: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_calendar: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_calendar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-layer future->spatial maps.

    Returns:
        layer_future_to_spatial: [L, B, Tf, P]
        history_to_spatial: [B, Th, P]
        layer_future_to_history: [L, B, Tf, Th]
    """
    bsz = hist_weather.shape[0]
    th = hist_weather.shape[1]
    tf = fut_weather.shape[1]

    hist_patch = model.patch_tokenizer(hist_weather) + model.spatial_pos_embed
    fut_patch = model.patch_tokenizer(fut_weather) + model.spatial_pos_embed

    hist_hour, hist_patch_attn = _summarize_hour_tokens_with_weights(model, hist_patch)
    fut_hour, _ = _summarize_hour_tokens_with_weights(model, fut_patch)

    if model.weather_stats_proj is not None:
        hist_hour = hist_hour + model.weather_stats_proj(model._weather_stats(hist_weather))
        fut_hour = fut_hour + model.weather_stats_proj(model._weather_stats(fut_weather))

    hist_tab = model.hist_tabular_embed(torch.cat([hist_demand, hist_calendar], dim=-1))
    masked_future_demand = model.future_demand_mask.view(1, 1, -1).expand(bsz, tf, -1)
    fut_tab = model.fut_tabular_embed(torch.cat([masked_future_demand, fut_calendar], dim=-1))

    enc_in = hist_hour + hist_tab
    dec_in = fut_hour + fut_tab
    enc_in = model.dropout(enc_in + model._temporal_encoding(th, enc_in.device))
    dec_in = model.dropout(dec_in + model._temporal_encoding(tf, dec_in.device))

    memory = model.temporal_encoder(enc_in)
    _, cross_w_by_layer = _decoder_with_cross_attention_weights(model, dec_in, memory)

    history_to_spatial = hist_patch_attn.mean(dim=2)  # [B, Th, P]
    layer_future_to_history = torch.stack(cross_w_by_layer, dim=0).mean(dim=2)  # [L,B,Tf,Th]
    layer_future_to_spatial = torch.einsum(
        "lbth,bhp->lbtp",
        layer_future_to_history,
        history_to_spatial,
    )  # [L,B,Tf,P]
    return layer_future_to_spatial, history_to_spatial, layer_future_to_history


def predict_with_cross_attention_mask(
    model: torch.nn.Module,
    hist_weather: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_calendar: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_calendar: torch.Tensor,
    active_layers: Set[int] | None = None,
) -> torch.Tensor:
    """Forward pass with optional decoder cross-attention layer masking.

    If `active_layers` is None, all decoder cross-attention layers are active.
    Otherwise, only layers in `active_layers` use cross-attention; all others are zeroed.
    """
    bsz = hist_weather.shape[0]
    th = hist_weather.shape[1]
    tf = fut_weather.shape[1]

    hist_patch = model.patch_tokenizer(hist_weather) + model.spatial_pos_embed
    fut_patch = model.patch_tokenizer(fut_weather) + model.spatial_pos_embed

    hist_hour, _ = _summarize_hour_tokens_with_weights(model, hist_patch)
    fut_hour, _ = _summarize_hour_tokens_with_weights(model, fut_patch)

    if model.weather_stats_proj is not None:
        hist_hour = hist_hour + model.weather_stats_proj(model._weather_stats(hist_weather))
        fut_hour = fut_hour + model.weather_stats_proj(model._weather_stats(fut_weather))

    hist_tab = model.hist_tabular_embed(torch.cat([hist_demand, hist_calendar], dim=-1))
    masked_future_demand = model.future_demand_mask.view(1, 1, -1).expand(bsz, tf, -1)
    fut_tab = model.fut_tabular_embed(torch.cat([masked_future_demand, fut_calendar], dim=-1))

    enc_in = hist_hour + hist_tab
    dec_in = fut_hour + fut_tab
    enc_in = model.dropout(enc_in + model._temporal_encoding(th, enc_in.device))
    dec_in = model.dropout(dec_in + model._temporal_encoding(tf, dec_in.device))
    memory = model.temporal_encoder(enc_in)

    x = dec_in
    for layer_idx, layer in enumerate(model.temporal_decoder.layers):
        use_cross = active_layers is None or layer_idx in active_layers

        if layer.norm_first:
            x = x + layer._sa_block(layer.norm1(x), None, None, False)
            x_norm = layer.norm2(x)
            if use_cross:
                mha_out, _ = layer.multihead_attn(
                    x_norm,
                    memory,
                    memory,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=False,
                    average_attn_weights=False,
                    is_causal=False,
                )
            else:
                mha_out = torch.zeros_like(x_norm)
            x = x + layer.dropout2(mha_out)
            x = x + layer._ff_block(layer.norm3(x))
        else:
            x2 = layer._sa_block(x, None, None, False)
            x = layer.norm1(x + x2)
            if use_cross:
                mha_out, _ = layer.multihead_attn(
                    x,
                    memory,
                    memory,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=False,
                    average_attn_weights=False,
                    is_causal=False,
                )
            else:
                mha_out = torch.zeros_like(x)
            x = layer.norm2(x + layer.dropout2(mha_out))
            x = layer.norm3(x + layer._ff_block(x))

    if model.temporal_decoder.norm is not None:
        x = model.temporal_decoder.norm(x)
    return model.pred_head(x)
