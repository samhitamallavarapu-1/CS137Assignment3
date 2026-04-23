#!/usr/bin/env python3
"""Plot Part 3 attention maps saved by code/part_3/run.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def _parse_indices(spec: str, upper: int) -> List[int]:
    spec = spec.strip().lower()
    if spec in {"", "all"}:
        return list(range(upper))

    out: List[int] = []
    for chunk in spec.split(","):
        c = chunk.strip()
        if not c:
            continue
        if "-" in c:
            a, b = c.split("-", maxsplit=1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid range: {c}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(c))

    out = sorted(set(out))
    for idx in out:
        if idx < 0 or idx >= upper:
            raise ValueError(f"Index {idx} out of bounds for [0, {upper - 1}]")
    return out


def _safe_zone_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _plot_panel(
    maps: Iterable[np.ndarray],
    titles: Iterable[str],
    out_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    suptitle: str,
) -> None:
    maps_list = list(maps)
    titles_list = list(titles)
    n = len(maps_list)
    rows, cols = _grid_shape(n)

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    # global color scale if not provided
    if vmin is None:
        vmin = min(float(np.min(m)) for m in maps_list)
    if vmax is None:
        vmax = max(float(np.max(m)) for m in maps_list)

    for i, ax in enumerate(axes_arr.flat):
        if i >= n:
            ax.axis("off")
            continue
        im = ax.imshow(maps_list[i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(titles_list[i], fontsize=10)
        ax.set_xlabel("Patch X")
        ax.set_ylabel("Patch Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Part 3 attention maps from attention_maps.npz")
    parser.add_argument("--attention-npz", type=Path, required=True, help="Path to attention_maps.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures; default: <npz_parent>/figures",
    )
    parser.add_argument("--future-hours", type=str, default="all", help="Indices like 'all', '0-23', or '0,6,12,18'")
    parser.add_argument("--zones", type=str, default="all", help="Zone indices like 'all' or '0,2,4'")
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    npz_path = args.attention_npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    payload = np.load(npz_path, allow_pickle=True)
    future_to_spatial = payload["future_to_spatial"]  # [N, Tf, Ph, Pw]
    zone_names = [str(z) for z in payload["zone_names"].tolist()]
    has_zone_maps = "zone_grad_spatial" in payload.files
    zone_grad_spatial = payload["zone_grad_spatial"] if has_zone_maps else None  # [N, Tf, Z, Ph, Pw]

    if future_to_spatial.ndim != 4:
        raise ValueError(f"future_to_spatial must be rank 4 [N,Tf,Ph,Pw], got {future_to_spatial.shape}")

    n_samples, tf, ph, pw = future_to_spatial.shape
    fut_idx = _parse_indices(args.future_hours, tf)
    zone_idx = _parse_indices(args.zones, len(zone_names)) if has_zone_maps else []

    out_dir = args.output_dir.resolve() if args.output_dir is not None else npz_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mean future->spatial (not zone-specific) per selected forecast hour.
    mean_future = future_to_spatial.mean(axis=0)  # [Tf, Ph, Pw]
    panel_maps = [mean_future[t] for t in fut_idx]
    panel_titles = [f"Forecast hour {t}" for t in fut_idx]
    _plot_panel(
        maps=panel_maps,
        titles=panel_titles,
        out_path=out_dir / "future_to_spatial_overview.png",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        suptitle=f"Mean Future->Spatial Attention (N={n_samples}, grid={ph}x{pw})",
    )

    # Also export each hour separately for paper-quality inclusion.
    for t in fut_idx:
        fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
        im = ax.imshow(mean_future[t], cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, origin="lower")
        ax.set_title(f"Mean Future->Spatial Attention | Forecast hour {t}")
        ax.set_xlabel("Patch X")
        ax.set_ylabel("Patch Y")
        fig.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"future_to_spatial_hour_{t:02d}.png", dpi=220)
        plt.close(fig)

    # 2) Zone-conditioned maps (if present): one panel per zone across selected hours.
    if has_zone_maps and zone_grad_spatial is not None:
        mean_zone = zone_grad_spatial.mean(axis=0)  # [Tf, Z, Ph, Pw]
        for z in zone_idx:
            z_name = zone_names[z]
            maps = [mean_zone[t, z] for t in fut_idx]
            titles = [f"Hour {t}" for t in fut_idx]
            _plot_panel(
                maps=maps,
                titles=titles,
                out_path=out_dir / f"zone_{z:02d}_{_safe_zone_name(z_name)}_hours.png",
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                suptitle=f"Zone-conditioned Spatial Attribution | {z_name}",
            )

        # Also aggregate across forecast horizon for each zone.
        horizon_mean = mean_zone.mean(axis=0)  # [Z, Ph, Pw]
        maps = [horizon_mean[z] for z in zone_idx]
        titles = [zone_names[z] for z in zone_idx]
        _plot_panel(
            maps=maps,
            titles=titles,
            out_path=out_dir / "zone_horizon_mean_overview.png",
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            suptitle="Zone-conditioned Spatial Attribution (Mean across forecast horizon)",
        )
    else:
        print("zone_grad_spatial not found in npz; skipping zone-conditioned plots.")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
