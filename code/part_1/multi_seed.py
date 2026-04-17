#!/usr/bin/env python3
"""Run multiple seeds for Part 1 training and summarize mean/std metrics."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd


def parse_seeds(seed_spec: str) -> list[int]:
    out: list[int] = []
    for chunk in seed_spec.split(","):
        c = chunk.strip()
        if not c:
            continue
        out.append(int(c))
    if not out:
        raise ValueError("No seeds were provided")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train.py for multiple seeds and summarize metrics")
    parser.add_argument("--python", type=str, required=True)
    parser.add_argument("--train-script", type=str, default="code/part_1/train.py")
    parser.add_argument("--run-name-prefix", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="40,41,42")
    parser.add_argument(
        "--shared-args",
        type=str,
        default="",
        help="Single quoted string of args forwarded to train.py",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    shared_args = shlex.split(args.shared_args)

    output_dir = Path("outputs/part_1")
    rows: list[dict] = []

    for seed in seeds:
        run_name = f"{args.run_name_prefix}_s{seed}"
        cmd = [args.python, args.train_script, "--run-name", run_name, "--seed", str(seed), *shared_args]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        metrics_path = output_dir / run_name / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        df = pd.read_csv(metrics_path)
        if df.empty:
            raise RuntimeError(f"No metrics rows found in {metrics_path}")

        best_row = df.loc[df["val_loss"].idxmin()]
        rows.append(
            {
                "seed": seed,
                "run_name": run_name,
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": float(best_row["val_loss"]),
                "best_val_mae": float(best_row["val_mae"]),
                "best_val_mape": float(best_row["val_mape"]),
                "last_epoch": int(df.iloc[-1]["epoch"]),
                "last_val_loss": float(df.iloc[-1]["val_loss"]),
                "last_val_mae": float(df.iloc[-1]["val_mae"]),
                "last_val_mape": float(df.iloc[-1]["val_mape"]),
            }
        )

    per_seed_df = pd.DataFrame(rows).sort_values("seed")

    summary = {
        "n_runs": int(len(per_seed_df)),
        "seeds": [int(x) for x in per_seed_df["seed"].tolist()],
        "best_val_loss_mean": float(per_seed_df["best_val_loss"].mean()),
        "best_val_loss_std": float(per_seed_df["best_val_loss"].std(ddof=1)) if len(per_seed_df) > 1 else 0.0,
        "best_val_mae_mean": float(per_seed_df["best_val_mae"].mean()),
        "best_val_mae_std": float(per_seed_df["best_val_mae"].std(ddof=1)) if len(per_seed_df) > 1 else 0.0,
        "best_val_mape_mean": float(per_seed_df["best_val_mape"].mean()),
        "best_val_mape_std": float(per_seed_df["best_val_mape"].std(ddof=1)) if len(per_seed_df) > 1 else 0.0,
    }

    summary_dir = output_dir / f"{args.run_name_prefix}_multiseed"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = summary_dir / "per_seed_metrics.csv"
    summary_json = summary_dir / "summary.json"

    per_seed_df.to_csv(per_seed_path, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-seed metrics:")
    print(per_seed_df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {per_seed_path}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
