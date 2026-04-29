import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text, fc="#f4f7fb", ec="#24415a", fontsize=9, roundness=0.06):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={roundness}",
        linewidth=1.5, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, p1, p2, color="#1d3557"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.4, color=color))


def full_model(out_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Part 1 BaselineCNNTransformerPatch Architecture", ha="center", va="center", fontsize=16, weight="bold")

    box(ax, (0.03, 0.72), 0.2, 0.14, "History weather\n[B, Th, Cw, H, W]", fc="#e8f2ff")
    box(ax, (0.03, 0.52), 0.2, 0.14, "Future weather\n[B, Tf, Cw, H, W]", fc="#e8f2ff")
    box(ax, (0.03, 0.31), 0.2, 0.14, "History demand\n[B, Th, Z]", fc="#fff5e6")
    box(ax, (0.03, 0.11), 0.2, 0.14, "Calendar feats\n(hist+future)\n[B, T, 7]", fc="#fff5e6")

    box(ax, (0.29, 0.62), 0.2, 0.24, "WeatherPatchTokenizer\n(Shared weights)\n\n4x ConvBlock (s=2)\nAdaptiveAvgPool -> 10x10\n1x1 Conv -> d_model\n\nOutput: [B, T, 100, d_model]", fc="#dff3e4")

    box(ax, (0.54, 0.70), 0.16, 0.12, "+ Spatial Pos Embed\n[1,1,100,d_model]", fc="#f1eaff")
    box(ax, (0.54, 0.53), 0.16, 0.12, "Hist tab embed\nLinear(Z+7 -> d_model)\n-> [B,Th,1,d_model]", fc="#ffe8cc")
    box(ax, (0.54, 0.36), 0.16, 0.12, "Future demand mask\nlearned [Z]\n+ Fut tab embed\n-> [B,Tf,1,d_model]", fc="#ffe8cc")

    box(ax, (0.74, 0.60), 0.2, 0.22, "Token grouping\nconcat spatial+tab per step\n\nHist: [B,Th,101,d]\nFuture: [B,Tf,101,d]\nAll: [B,Th+Tf,101,d]", fc="#eaf4ff")

    box(ax, (0.74, 0.31), 0.2, 0.2, "Temporal sinusoidal PE\nadded over time-step axis\n\nReshape -> sequence\n[B,(Th+Tf)*101,d]\nDropout", fc="#eaf4ff")

    box(ax, (0.29, 0.24), 0.2, 0.22, "TransformerEncoder\nL=4, heads=8\nff=1024, GELU\nnorm_first=True\n\nOutput seq: [B,S,d]", fc="#fcefee")

    box(ax, (0.54, 0.17), 0.16, 0.16, "Select future tab tokens\npositions:\nt*101 + 100\nfor t in future steps\n\n[B,Tf,d]", fc="#edf6ff")

    box(ax, (0.74, 0.08), 0.2, 0.16, "Prediction head\nLinear(d->d) + GELU + Dropout\nLinear(d->Z)\n\nOutput: [B,Tf,Z]", fc="#e5ffe9")

    arrow(ax, (0.23, 0.79), (0.29, 0.74))
    arrow(ax, (0.23, 0.59), (0.29, 0.70))
    arrow(ax, (0.49, 0.74), (0.54, 0.76))
    arrow(ax, (0.49, 0.66), (0.54, 0.76))

    arrow(ax, (0.23, 0.38), (0.54, 0.59))
    arrow(ax, (0.23, 0.18), (0.54, 0.57))
    arrow(ax, (0.23, 0.18), (0.54, 0.40))

    arrow(ax, (0.70, 0.76), (0.74, 0.71))
    arrow(ax, (0.70, 0.59), (0.74, 0.69))
    arrow(ax, (0.70, 0.42), (0.74, 0.66))

    arrow(ax, (0.84, 0.60), (0.84, 0.51))
    arrow(ax, (0.74, 0.42), (0.49, 0.35))
    arrow(ax, (0.49, 0.32), (0.54, 0.25))
    arrow(ax, (0.70, 0.20), (0.74, 0.16))

    ax.text(0.03, 0.02, "From code/part_1/model.py: BaselineCNNTransformerPatch + WeatherPatchTokenizer", fontsize=9, color="#444")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def tokenizer_only(out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "Part 1 CNN Tokenizer (WeatherPatchTokenizer)", ha="center", va="center", fontsize=16, weight="bold")

    box(ax, (0.03, 0.38), 0.16, 0.2, "Input weather\n[B,T,Cw,H,W]", fc="#e8f2ff")
    box(ax, (0.24, 0.38), 0.17, 0.2, "Reshape\n[B*T,Cw,H,W]", fc="#f3f7ff")

    box(ax, (0.45, 0.66), 0.2, 0.12, "ConvBlock 1\nConv3x3 s=2 p=1\nBN + GELU\nCw -> h", fc="#dff3e4")
    box(ax, (0.45, 0.51), 0.2, 0.12, "ConvBlock 2\nConv3x3 s=2 p=1\nBN + GELU\nh -> h", fc="#dff3e4")
    box(ax, (0.45, 0.36), 0.2, 0.12, "ConvBlock 3\nConv3x3 s=2 p=1\nBN + GELU\nh -> 2h", fc="#dff3e4")
    box(ax, (0.45, 0.21), 0.2, 0.12, "ConvBlock 4\nConv3x3 s=2 p=1\nBN + GELU\n2h -> 2h", fc="#dff3e4")

    box(ax, (0.69, 0.46), 0.14, 0.16, "AdaptiveAvgPool2d\n-> (gh, gw)\ndefault: 10x10", fc="#fff5e6")
    box(ax, (0.85, 0.46), 0.12, 0.16, "1x1 Conv\n2h -> d_model", fc="#fff5e6")

    box(ax, (0.69, 0.18), 0.28, 0.16, "Flatten spatial + transpose\n[B*T, d_model, gh, gw] -> [B*T, gh*gw, d_model]\nReshape -> [B, T, gh*gw, d_model]", fc="#eaf4ff")

    arrow(ax, (0.19, 0.48), (0.24, 0.48))
    arrow(ax, (0.41, 0.48), (0.45, 0.72))
    arrow(ax, (0.55, 0.66), (0.55, 0.63))
    arrow(ax, (0.55, 0.51), (0.55, 0.48))
    arrow(ax, (0.55, 0.36), (0.55, 0.33))
    arrow(ax, (0.65, 0.27), (0.69, 0.54))
    arrow(ax, (0.83, 0.54), (0.85, 0.54))
    arrow(ax, (0.91, 0.46), (0.83, 0.28))

    ax.text(0.03, 0.05, "Conv stack downsamples by 2^4 = 16x before adaptive pooling; output patch tokens count = gh*gw", fontsize=9, color="#444")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    full_model("CS137Assignment3/outputs/part_1/architecture/part1_model_architecture.png")
    tokenizer_only("CS137Assignment3/outputs/part_1/architecture/part1_cnn_tokenizer_architecture.png")
    print("Wrote diagrams to CS137Assignment3/outputs/part_1/architecture/")
