"""Attention and attribution helpers for Part 3 diagnostics (Part 1 only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import torch
import torch.nn.functional as F


@dataclass
class AttentionBatchOutput:
    future_to_history: torch.Tensor  # [B, Tf, Th]
    history_to_spatial: torch.Tensor  # [B, Th, P]
    future_to_spatial: torch.Tensor  # [B, Tf, P]
    predictions: torch.Tensor  # [B, Tf, Z]
    zone_grad_spatial: torch.Tensor | None  # [B, Tf, Z, P]


def _encoder_with_self_attention_weights(
    model: torch.nn.Module,
    seq: torch.Tensor,
    active_layers: Set[int] | None = None,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Run TransformerEncoder while extracting per-layer self-attention weights."""
    x = seq
    self_weights: List[torch.Tensor] = []

    for layer_idx, layer in enumerate(model.transformer.layers):
        use_self_attn = active_layers is None or layer_idx in active_layers
        if layer.norm_first:
            x_norm = layer.norm1(x)
            if use_self_attn:
                sa_out, sa_w = layer.self_attn(
                    x_norm,
                    x_norm,
                    x_norm,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False,
                    is_causal=False,
                )
            else:
                sa_out = torch.zeros_like(x_norm)
                bsz, seqlen, _ = x_norm.shape
                heads = layer.self_attn.num_heads
                sa_w = torch.zeros(bsz, heads, seqlen, seqlen, device=x_norm.device, dtype=x_norm.dtype)
            x = x + layer.dropout1(sa_out)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            if use_self_attn:
                sa_out, sa_w = layer.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False,
                    is_causal=False,
                )
            else:
                sa_out = torch.zeros_like(x)
                bsz, seqlen, _ = x.shape
                heads = layer.self_attn.num_heads
                sa_w = torch.zeros(bsz, heads, seqlen, seqlen, device=x.device, dtype=x.dtype)
            x = layer.norm1(x + layer.dropout1(sa_out))
            x = layer.norm2(x + layer._ff_block(x))

        self_weights.append(sa_w)

    if model.transformer.norm is not None:
        x = model.transformer.norm(x)
    return x, self_weights


def _project_qk(
    attn: torch.nn.MultiheadAttention,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project sequence embeddings to multi-head Q/K tensors."""
    embed_dim = attn.embed_dim
    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads

    q_proj_weight, k_proj_weight, _ = attn.in_proj_weight.chunk(3, dim=0)
    if attn.in_proj_bias is not None:
        q_proj_bias, k_proj_bias, _ = attn.in_proj_bias.chunk(3, dim=0)
    else:
        q_proj_bias = None
        k_proj_bias = None

    q = F.linear(x, q_proj_weight, q_proj_bias)
    k = F.linear(x, k_proj_weight, k_proj_bias)

    bsz, seqlen, _ = q.shape
    q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()
    return q, k


def _gather_history_patch_scores(
    probs: torch.Tensor,
    history_patch_pos: torch.Tensor,
    query_start: int,
    query_len: int,
) -> torch.Tensor:
    """Gather per-history-step patch scores from a history-query probability block."""
    patch_slice = history_patch_pos[query_start : query_start + query_len].to(probs.device)
    gather_index = patch_slice.unsqueeze(0).unsqueeze(0).expand(probs.shape[0], probs.shape[1], -1, -1)
    return torch.gather(probs, dim=-1, index=gather_index)


def _extract_target_maps_for_layer(
    layer: torch.nn.TransformerEncoderLayer,
    attn_input: torch.Tensor,
    future_tab_pos: torch.Tensor,
    history_tab_pos: torch.Tensor,
    history_patch_pos: torch.Tensor,
    query_chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the attention slices needed for Part 3 diagnostics."""
    q_all, k_all = _project_qk(layer.self_attn, attn_input)
    scale = layer.self_attn.head_dim ** -0.5
    k_t = k_all.transpose(-2, -1).contiguous()

    future_tab_pos = future_tab_pos.to(attn_input.device)
    history_tab_pos = history_tab_pos.to(attn_input.device)

    future_chunks: List[torch.Tensor] = []
    for start in range(0, future_tab_pos.numel(), query_chunk_size):
        q_idx = future_tab_pos[start : start + query_chunk_size]
        q_chunk = q_all[:, :, q_idx, :]
        logits = torch.matmul(q_chunk * scale, k_t)
        probs = torch.softmax(logits, dim=-1)
        future_chunks.append(probs[:, :, :, history_tab_pos].mean(dim=1))
        del logits, probs

    history_chunks: List[torch.Tensor] = []
    for start in range(0, history_tab_pos.numel(), query_chunk_size):
        q_idx = history_tab_pos[start : start + query_chunk_size]
        q_chunk = q_all[:, :, q_idx, :]
        logits = torch.matmul(q_chunk * scale, k_t)
        probs = torch.softmax(logits, dim=-1)
        hist_patch = _gather_history_patch_scores(
            probs=probs,
            history_patch_pos=history_patch_pos,
            query_start=start,
            query_len=q_idx.numel(),
        )
        history_chunks.append(hist_patch.mean(dim=1))
        del logits, probs, hist_patch

    future_to_history = torch.cat(future_chunks, dim=1) if future_chunks else torch.empty(0, device=attn_input.device)
    history_to_spatial = (
        torch.cat(history_chunks, dim=1) if history_chunks else torch.empty(0, device=attn_input.device)
    )
    return future_to_history, history_to_spatial


def _encoder_with_target_maps(
    model: torch.nn.Module,
    seq: torch.Tensor,
    future_tab_pos: torch.Tensor,
    history_tab_pos: torch.Tensor,
    history_patch_pos: torch.Tensor,
    active_layers: Set[int] | None = None,
) -> tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Run TransformerEncoder and extract only the needed attention slices."""
    x = seq
    future_history_weights: List[torch.Tensor] = []
    history_spatial_weights: List[torch.Tensor] = []

    for layer_idx, layer in enumerate(model.transformer.layers):
        use_self_attn = active_layers is None or layer_idx in active_layers
        layer_input = layer.norm1(x) if layer.norm_first else x

        if use_self_attn:
            future_hist, history_spatial = _extract_target_maps_for_layer(
                layer=layer,
                attn_input=layer_input,
                future_tab_pos=future_tab_pos,
                history_tab_pos=history_tab_pos,
                history_patch_pos=history_patch_pos,
            )
            sa_out, _ = layer.self_attn(
                layer_input,
                layer_input,
                layer_input,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
                average_attn_weights=False,
                is_causal=False,
            )
        else:
            future_hist = torch.zeros(
                x.shape[0],
                future_tab_pos.numel(),
                history_tab_pos.numel(),
                device=x.device,
                dtype=x.dtype,
            )
            history_spatial = torch.zeros(
                x.shape[0],
                history_tab_pos.numel(),
                history_patch_pos.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
            sa_out = torch.zeros_like(layer_input)

        if layer.norm_first:
            x = x + layer.dropout1(sa_out)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            x = layer.norm1(x + layer.dropout1(sa_out))
            x = layer.norm2(x + layer._ff_block(x))

        future_history_weights.append(future_hist)
        history_spatial_weights.append(history_spatial)

    if model.transformer.norm is not None:
        x = model.transformer.norm(x)
    return x, future_history_weights, history_spatial_weights


def _build_part1_seq(
    model: torch.nn.Module,
    hist_weather: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_calendar: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_calendar: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
    """Construct Part 1 token sequence and bookkeeping indices."""
    bsz = hist_weather.shape[0]
    th = hist_weather.shape[1]
    tf = fut_weather.shape[1]
    num_patches = model.config.patch_grid_h * model.config.patch_grid_w
    tokens_per_step = num_patches + 1

    hist_spatial = model.patch_tokenizer(hist_weather) + model.spatial_pos_embed
    fut_spatial = model.patch_tokenizer(fut_weather) + model.spatial_pos_embed

    hist_tab = model.hist_tabular_embed(torch.cat([hist_demand, hist_calendar], dim=-1)).unsqueeze(2)
    masked_future_demand = model.future_demand_mask.view(1, 1, -1).expand(bsz, tf, -1)
    fut_tab = model.fut_tabular_embed(torch.cat([masked_future_demand, fut_calendar], dim=-1)).unsqueeze(2)

    hist_group = torch.cat([hist_spatial, hist_tab], dim=2)  # [B,Th,P+1,D]
    fut_group = torch.cat([fut_spatial, fut_tab], dim=2)  # [B,Tf,P+1,D]
    all_group = torch.cat([hist_group, fut_group], dim=1)  # [B,Th+Tf,P+1,D]

    total_steps = th + tf
    all_group = all_group + model._temporal_encoding(total_steps, all_group.device)
    seq = model.dropout(all_group.reshape(bsz, total_steps * tokens_per_step, model.config.d_model))

    hist_step_idx = torch.arange(th, device=seq.device, dtype=torch.long)
    fut_step_idx = torch.arange(th, th + tf, device=seq.device, dtype=torch.long)
    history_tab_pos = hist_step_idx * tokens_per_step + num_patches  # [Th]
    future_tab_pos = fut_step_idx * tokens_per_step + num_patches  # [Tf]
    history_patch_pos = (
        hist_step_idx.unsqueeze(1) * tokens_per_step
        + torch.arange(num_patches, device=seq.device, dtype=torch.long).unsqueeze(0)
    )  # [Th,P]

    info: Dict[str, torch.Tensor | int] = {
        "history_tab_pos": history_tab_pos,
        "future_tab_pos": future_tab_pos,
        "history_patch_pos": history_patch_pos,
        "num_patches": num_patches,
    }
    return seq, info


def _maps_from_part1_self_attention(
    attn_by_layer: List[torch.Tensor],
    history_tab_pos: torch.Tensor,
    future_tab_pos: torch.Tensor,
    history_patch_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive attention maps from Part 1 self-attention layers."""
    layer_future_history: List[torch.Tensor] = []
    layer_history_spatial: List[torch.Tensor] = []
    layer_future_spatial: List[torch.Tensor] = []

    for layer_w in attn_by_layer:
        # layer_w: [B, H, S, S]
        attn = layer_w.mean(dim=1)  # [B,S,S]

        fut_hist = attn[:, future_tab_pos, :][:, :, history_tab_pos]  # [B,Tf,Th]
        hist_to_spatial_all = attn[:, history_tab_pos, :]  # [B,Th,S]
        hist_patch_idx = history_patch_pos.unsqueeze(0).expand(attn.shape[0], -1, -1)  # [B,Th,P]
        hist_spatial = torch.gather(hist_to_spatial_all, dim=2, index=hist_patch_idx)  # [B,Th,P]
        fut_spatial = torch.matmul(fut_hist, hist_spatial)  # [B,Tf,P]

        layer_future_history.append(fut_hist)
        layer_history_spatial.append(hist_spatial)
        layer_future_spatial.append(fut_spatial)

    layer_future_history_t = torch.stack(layer_future_history, dim=0)  # [L,B,Tf,Th]
    layer_history_spatial_t = torch.stack(layer_history_spatial, dim=0)  # [L,B,Th,P]
    layer_future_spatial_t = torch.stack(layer_future_spatial, dim=0)  # [L,B,Tf,P]

    future_to_history = layer_future_history_t.mean(dim=0)
    history_to_spatial = layer_history_spatial_t.mean(dim=0)
    future_to_spatial = layer_future_spatial_t.mean(dim=0)
    return (
        layer_future_spatial_t,
        layer_history_spatial_t,
        layer_future_history_t,
        future_to_history,
        history_to_spatial,
        future_to_spatial,
    )


def extract_attention_batch(
    model: torch.nn.Module,
    hist_weather: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_calendar: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_calendar: torch.Tensor,
    compute_zone_grad_maps: bool = True,
) -> AttentionBatchOutput:
    """Extract composed attention maps and optional zone-conditioned maps for Part 1."""
    bsz = hist_weather.shape[0]
    tf = fut_weather.shape[1]
    num_patches = model.config.patch_grid_h * model.config.patch_grid_w
    num_zones = model.config.num_zones

    seq, info = _build_part1_seq(
        model=model,
        hist_weather=hist_weather,
        hist_demand=hist_demand,
        hist_calendar=hist_calendar,
        fut_weather=fut_weather,
        fut_calendar=fut_calendar,
    )
    if compute_zone_grad_maps:
        # Zone-conditioned attribution needs gradients wrt the input token sequence,
        # since the final encoded tensor is only read at future-token positions.
        seq = seq.detach().requires_grad_(True)
    history_tab_pos = info["history_tab_pos"]
    future_tab_pos = info["future_tab_pos"]
    history_patch_pos = info["history_patch_pos"]
    encoded, future_history_by_layer, history_spatial_by_layer = _encoder_with_target_maps(
        model=model,
        seq=seq,
        future_tab_pos=future_tab_pos,
        history_tab_pos=history_tab_pos,
        history_patch_pos=history_patch_pos,
        active_layers=None,
    )
    assert isinstance(history_tab_pos, torch.Tensor)
    assert isinstance(future_tab_pos, torch.Tensor)
    assert isinstance(history_patch_pos, torch.Tensor)
    layer_future_history = torch.stack(future_history_by_layer, dim=0)
    layer_history_spatial = torch.stack(history_spatial_by_layer, dim=0)
    future_to_history = layer_future_history.mean(dim=0)
    history_to_spatial = layer_history_spatial.mean(dim=0)
    future_to_spatial = torch.matmul(future_to_history, history_to_spatial)

    fut_states = encoded[:, future_tab_pos, :]
    preds = model.pred_head(fut_states)

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
                grad_seq = torch.autograd.grad(
                    score,
                    seq,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]  # [B, S, D]
                hour_importance = grad_seq[:, history_tab_pos, :].abs().mean(dim=-1)  # [B, Th]
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
    """Return per-layer future->spatial maps for Part 1.

    Returns:
        layer_future_to_spatial: [L, B, Tf, P]
        history_to_spatial: [B, Th, P]
        layer_future_to_history: [L, B, Tf, Th]
    """
    seq, info = _build_part1_seq(
        model=model,
        hist_weather=hist_weather,
        hist_demand=hist_demand,
        hist_calendar=hist_calendar,
        fut_weather=fut_weather,
        fut_calendar=fut_calendar,
    )
    history_tab_pos = info["history_tab_pos"]
    future_tab_pos = info["future_tab_pos"]
    history_patch_pos = info["history_patch_pos"]
    _, future_history_by_layer, history_spatial_by_layer = _encoder_with_target_maps(
        model=model,
        seq=seq,
        future_tab_pos=future_tab_pos,
        history_tab_pos=history_tab_pos,
        history_patch_pos=history_patch_pos,
        active_layers=None,
    )
    assert isinstance(history_tab_pos, torch.Tensor)
    assert isinstance(future_tab_pos, torch.Tensor)
    assert isinstance(history_patch_pos, torch.Tensor)
    layer_future_to_history = torch.stack(future_history_by_layer, dim=0)
    layer_history_to_spatial = torch.stack(history_spatial_by_layer, dim=0)
    layer_future_to_spatial = torch.einsum("lbth,lbhp->lbtp", layer_future_to_history, layer_history_to_spatial)
    history_to_spatial = layer_history_to_spatial.mean(dim=0)
    return layer_future_to_spatial, history_to_spatial, layer_future_to_history


def predict_with_layer_mask(
    model: torch.nn.Module,
    hist_weather: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_calendar: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_calendar: torch.Tensor,
    active_layers: Set[int] | None = None,
) -> torch.Tensor:
    """Forward pass with optional Part 1 transformer-layer masking."""
    seq, info = _build_part1_seq(
        model=model,
        hist_weather=hist_weather,
        hist_demand=hist_demand,
        hist_calendar=hist_calendar,
        fut_weather=fut_weather,
        fut_calendar=fut_calendar,
    )
    encoded, _ = _encoder_with_self_attention_weights(model, seq, active_layers=active_layers)
    future_tab_pos = info["future_tab_pos"]
    assert isinstance(future_tab_pos, torch.Tensor)
    return model.pred_head(encoded[:, future_tab_pos, :])
