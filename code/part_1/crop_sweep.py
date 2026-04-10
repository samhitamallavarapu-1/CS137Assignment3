#!/usr/bin/env python3
"""Visual utility to tune New England crop bounds.

Coordinate convention used here (oriented space):
- Image is transformed to north-up, west-left orientation.
- Box format is y0:y1:x0:x1 in oriented coordinates.

Conversion back to raw tensor coordinates used by train.py:
- raw_y0 = oriented_x0
- raw_y1 = oriented_x1
- raw_x0 = oriented_y0
- raw_x1 = oriented_y1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class Box:
    name: str
    oy0: int
    oy1: int
    ox0: int
    ox1: int

    def raw_coords(self) -> tuple[int, int, int, int]:
        # oriented[y, x] = raw[x, y]
        ry0, ry1 = self.ox0, self.ox1
        rx0, rx1 = self.oy0, self.oy1
        return ry0, ry1, rx0, rx1


def parse_box(spec: str) -> Box:
    # Format: name:oy0:oy1:ox0:ox1
    parts = spec.split(":")
    if len(parts) != 5:
        raise ValueError(f"Bad box spec '{spec}', expected name:oy0:oy1:ox0:ox1")
    n, oy0, oy1, ox0, ox1 = parts
    return Box(n, int(oy0), int(oy1), int(ox0), int(ox1))


def default_boxes() -> List[Box]:
    # Candidate boxes around New England after orientation correction.
    return [
        Box("old", 247, 427, 81, 248),
        Box("ne_a", 90, 220, 120, 250),
        Box("ne_b", 95, 235, 135, 275),
        Box("ne_c", 80, 220, 145, 295),
        Box("ne_d", 110, 250, 150, 300),
        Box("ne_e", 85, 205, 165, 305),
    ]


def robust_norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep crop boxes for oriented weather maps")
    ap.add_argument("--weather-pt", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--downsample-h", type=int, default=96)
    ap.add_argument("--downsample-w", type=int, default=96)
    ap.add_argument(
        "--box",
        action="append",
        default=[],
        help="box as name:oy0:oy1:ox0:ox1 (in oriented coords). Repeat for multiple.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(args.weather_pt, map_location="cpu")
    if raw.ndim != 3:
        raise ValueError(f"Expected weather tensor [H,W,C], got {tuple(raw.shape)}")
    h, w, c = raw.shape
    if not (0 <= args.channel < c):
        raise ValueError(f"channel must be in [0, {c-1}]")

    # Oriented image: north-up, west-left.
    raw_ch = raw[:, :, args.channel].numpy()   # [H,W]
    ori_ch = raw_ch.T                          # [W,H]
    oh, ow = ori_ch.shape

    boxes = [parse_box(b) for b in args.box] if args.box else default_boxes()

    # Validate boxes.
    for b in boxes:
        if not (0 <= b.oy0 < b.oy1 <= oh and 0 <= b.ox0 < b.ox1 <= ow):
            raise ValueError(f"Box {b.name} out of bounds for oriented shape {(oh, ow)}")

    # Save summary CSV.
    csv_path = args.output_dir / "crop_candidates_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "name", "oriented_y0", "oriented_y1", "oriented_x0", "oriented_x1",
            "raw_y0", "raw_y1", "raw_x0", "raw_x1",
        ])
        for b in boxes:
            ry0, ry1, rx0, rx1 = b.raw_coords()
            wr.writerow([b.name, b.oy0, b.oy1, b.ox0, b.ox1, ry0, ry1, rx0, rx1])

    # Figure 1: oriented map + all boxes.
    fig1, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)
    ax.imshow(robust_norm(ori_ch), cmap="viridis", origin="upper")
    palette = ["red", "orange", "cyan", "magenta", "lime", "white", "yellow"]
    for i, b in enumerate(boxes):
        color = palette[i % len(palette)]
        ax.add_patch(Rectangle((b.ox0, b.oy0), b.ox1 - b.ox0, b.oy1 - b.oy0, fill=False, edgecolor=color, linewidth=2))
        ax.text(b.ox0 + 2, b.oy0 + 12, b.name, color=color, fontsize=9, weight="bold", bbox=dict(facecolor="black", alpha=0.25, pad=1))
    ax.set_title("Oriented weather map (north-up, west-left) with candidate crops")
    ax.set_xlabel("x (west -> east)")
    ax.set_ylabel("y (north -> south)")
    ax.set_xticks(np.arange(0, ow, 25))
    ax.set_yticks(np.arange(0, oh, 25))
    ax.grid(color="white", alpha=0.15, linewidth=0.5)
    fig1.savefig(args.output_dir / "crop_candidates_overlay.png", dpi=180)

    # Figure 2: each candidate crop downsampled.
    n = len(boxes)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig2, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    raw_chw = raw.permute(2, 0, 1).contiguous()
    for i, b in enumerate(boxes):
        ax_i = axes[i]
        ry0, ry1, rx0, rx1 = b.raw_coords()
        crop = raw_chw[:, ry0:ry1, rx0:rx1].unsqueeze(0)
        down = F.interpolate(crop, size=(args.downsample_h, args.downsample_w), mode="bilinear", align_corners=False)[0, args.channel].numpy()
        ax_i.imshow(robust_norm(down), cmap="viridis", origin="upper")
        ax_i.set_title(f"{b.name} -> raw(y:{ry0}:{ry1}, x:{rx0}:{rx1})")
        ax_i.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig2.savefig(args.output_dir / "crop_candidates_downsampled.png", dpi=180)

    print(f"Saved: {csv_path}")
    print(f"Saved: {args.output_dir / 'crop_candidates_overlay.png'}")
    print(f"Saved: {args.output_dir / 'crop_candidates_downsampled.png'}")


if __name__ == "__main__":
    main()
