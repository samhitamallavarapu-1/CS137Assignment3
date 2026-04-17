#!/usr/bin/env python3
"""
Evaluate an energy demand forecasting model on the test set.

Usage:
    python evaluate.py <MODEL_NAME> [N_DAYS]

Configuration
-------------
    MODEL_NAME : str
        Name of the model folder inside evaluation/.  The folder must contain
        a model.py file that exposes:
            get_model(metadata: dict) -> torch.nn.Module
        The model must implement two methods:

        adapt_inputs(history_weather, history_energy, future_weather, future_time) -> tuple
            Called first on the raw evaluation inputs.  Override to reduce or
            reshape data (e.g. spatial pooling, history subsampling) before the
            forward pass.  The default implementation is an identity pass-through
            that returns all four inputs unchanged.
                history_weather : (B, 168, 450, 449, 7) float32
                history_energy  : (B, 168, n_zones) float32
                future_weather  : (B, 24, 450, 449, 7) float32
                future_time     : (B, 24) int64  -- hours since Unix epoch

        forward(*adapt_inputs(...)) -> (B, 24, n_zones) float32
            Called with the unpacked output of adapt_inputs().  The signature
            must match whatever adapt_inputs() returns.

    TEST_YEAR : int
        Calendar year of the test data.

Metrics
-------
  - MAPE per energy zone
  - Overall MAPE (mean across all zones and timesteps)
"""

import sys
import importlib.util
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# ============================================================
# Configuration — edit here to switch models or test periods
# ============================================================

MODEL_NAME = sys.argv[1]
TEST_YEAR  = 2022

if len(sys.argv) >= 3:
    n_eval_days = int(sys.argv[2])
else:
    n_eval_days = 2

print(f"Evaluating model: {MODEL_NAME}  |  test year: {TEST_YEAR}  |  n_eval_days: {n_eval_days}")

# ============================================================
# Paths (derived automatically — no need to edit)
# ============================================================

EVAL_DIR    = Path(__file__).parent          # evaluation/
ROOT        = EVAL_DIR.parent                # assignment3_data/
WEATHER_DIR = ROOT / "weather_data"
ENERGY_DIR  = ROOT / "energy_demand_data"

HISTORY_LEN = 168   # hours of history fed to the model
FUTURE_LEN  = 24    # hours to predict

# ============================================================
# Load and validate energy demand data
# ============================================================

print("Loading energy demand data …")

dfs = []
for csv_path in sorted(ENERGY_DIR.glob("target_energy_zonal_*.csv")):
    dfs.append(pd.read_csv(csv_path, parse_dates=["timestamp_utc"]))
energy_df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)

# Verify that the data is contiguous in time (no gaps)
deltas = energy_df["timestamp_utc"].diff().dropna()
assert (deltas == pd.Timedelta("1h")).all(), \
    "Energy demand data is NOT contiguous in time — found gaps!"
print("  Contiguity check : PASSED")

ZONE_COLS = [c for c in energy_df.columns if c != "timestamp_utc"]
n_zones   = len(ZONE_COLS)
print(f"  Zones ({n_zones})    : {ZONE_COLS}")
print(f"  Time range        : {energy_df['timestamp_utc'].iloc[0]}  →  {energy_df['timestamp_utc'].iloc[-1]}")
print(f"  Total hours       : {len(energy_df)}")

# Fast numpy array for indexing energy values
energy_values = energy_df[ZONE_COLS].values.astype(np.float32)  # (T, n_zones)
# Integer hours-since-epoch for each row (datetime64[h] cast to int64)
all_hours = energy_df["timestamp_utc"].values.astype("datetime64[h]").astype(np.int64)  # (T,)

# ============================================================
# Identify test dates
#
#   Each element of test_dates is the row index of midnight (00:00 UTC)
#   that begins a 24-hour prediction window.
#   History  : rows [t_idx - HISTORY_LEN  ..  t_idx - 1]
#   Prediction: rows [t_idx              ..  t_idx + FUTURE_LEN - 1]
# ============================================================

midnight_mask = (
    (energy_df["timestamp_utc"].dt.year == TEST_YEAR) &
    (energy_df["timestamp_utc"].dt.hour == 0)
)
test_dates = np.where(midnight_mask)[0]
# Keep only dates where both history and future windows are in-bounds
test_dates = test_dates[
    (test_dates >= HISTORY_LEN) &
    (test_dates + FUTURE_LEN <= len(energy_df))
]
# Limit evaluation to the last n_eval_days dates
test_dates = test_dates[-n_eval_days:]

print(f"\nTest configuration")
print(f"  Model    : {MODEL_NAME}")
print(f"  Test year: {TEST_YEAR}")
print(f"  # dates  : {len(test_dates)}")
t0 = energy_df["timestamp_utc"].iloc[test_dates[0]]
t1 = energy_df["timestamp_utc"].iloc[test_dates[-1]]
print(f"  Dates    : {t0.date()}  →  {t1.date()}")

# ============================================================
# Helpers
# ============================================================

# Simple dict cache so adjacent windows share weather tensors
_weather_cache: dict = {}

def load_weather(hour_int64: int) -> torch.Tensor:
    """Return (450, 449, 7) float32 tensor for the given hours-since-epoch."""
    if hour_int64 in _weather_cache:
        return _weather_cache[hour_int64]
    dt = pd.Timestamp(int(hour_int64), unit="h")
    path = WEATHER_DIR / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    t = torch.load(path, weights_only=True).float()
    _weather_cache[hour_int64] = t
    # Evict oldest entries to keep cache bounded (~200 tensors ≈ 1 GB)
    if len(_weather_cache) > 200:
        oldest = next(iter(_weather_cache))
        del _weather_cache[oldest]
    return t


# ============================================================
# Load model
#
#   Dynamically loads evaluation/<MODEL_NAME>/model.py and calls get_model().
#   Any model folder dropped into evaluation/ works as long as its model.py
#   follows the interface described at the top of this file.
# ============================================================

model_path = EVAL_DIR / MODEL_NAME / "model.py"
if not model_path.exists():
    raise FileNotFoundError(
        f"No model file found at {model_path}\n"
        f"Create evaluation/{MODEL_NAME}/model.py with a get_model(metadata) function."
    )

# Add the model's own directory to sys.path so sibling .py files can be imported.
model_dir = str(EVAL_DIR / MODEL_NAME)
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

metadata = {
    "zone_names":     ZONE_COLS,
    "n_zones":        n_zones,
    "history_len":    HISTORY_LEN,
    "future_len":     FUTURE_LEN,
    "n_weather_vars": 7,
}

spec   = importlib.util.spec_from_file_location("model", model_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model  = module.get_model(metadata)
model.eval()
print(f"\nLoaded model : {MODEL_NAME}  ({model.__class__.__name__})")

# ============================================================
# Inference loop
# ============================================================

all_preds   = []   # list of (24, n_zones) tensors
all_targets = []   # list of (24, n_zones) tensors

print(f"\nRunning inference …")

with torch.no_grad():
    for step, t_idx in enumerate(test_dates):
        hist_slice   = slice(t_idx - HISTORY_LEN, t_idx)
        future_slice = slice(t_idx, t_idx + FUTURE_LEN)

        hist_hours_arr   = all_hours[hist_slice]    # (168,) int64
        future_hours_arr = all_hours[future_slice]  # (24,)  int64

        # --- Energy history ---
        hist_energy = energy_values[hist_slice]     # (168, n_zones) float32
        if np.isnan(hist_energy).any():
            continue

        # --- History weather ---
        try:
            hist_weather = torch.stack([load_weather(int(h)) for h in hist_hours_arr])
        except FileNotFoundError:
            continue
        if torch.isnan(hist_weather).any():
            continue

        # --- Future weather ---
        try:
            fut_weather = torch.stack([load_weather(int(h)) for h in future_hours_arr])
        except FileNotFoundError:
            continue
        if torch.isnan(fut_weather).any():
            continue

        # --- Future time: (B, 24) int64  -- hours since Unix epoch ---
        fut_time = torch.tensor(future_hours_arr, dtype=torch.int64)  # (24,)

        # --- Forward pass (add and remove batch dimension) ---
        raw_inputs = (
            hist_weather.unsqueeze(0),                              # (1, 168, 450, 449, 7)
            torch.from_numpy(hist_energy).unsqueeze(0),            # (1, 168, n_zones)
            fut_weather.unsqueeze(0),                               # (1, 24, 450, 449, 7)
            fut_time.unsqueeze(0),                                  # (1, 24)
        )
        pred = model(*model.adapt_inputs(*raw_inputs)).squeeze(0)  # (24, n_zones)

        target = torch.from_numpy(energy_values[future_slice])     # (24, n_zones)

        all_preds.append(pred)
        all_targets.append(target)

        date_str = energy_df["timestamp_utc"].iloc[t_idx].date()
        print(f"  [{step+1:>3}/{len(test_dates)}]  {date_str}")

preds   = torch.stack(all_preds).float()    # (N, 24, n_zones)
targets = torch.stack(all_targets).float()  # (N, 24, n_zones)

# ============================================================
# Metrics: MAPE per zone + overall MAPE
# ============================================================

print("\n" + "=" * 65)
print(f"Results  —  model: {MODEL_NAME}   test year: {TEST_YEAR}")
print("=" * 65)

def mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Percentage Error, skipping timesteps where target == 0."""
    mask = target != 0
    return (((pred[mask] - target[mask]).abs() / target[mask].abs()).mean() * 100).item()

for j, zone in enumerate(ZONE_COLS):
    m = mape(preds[:, :, j], targets[:, :, j])
    print(f"  {zone:20s}  MAPE: {m:7.2f} %")

overall_mape = mape(preds, targets)
print(f"\n  {'Overall':20s}  MAPE: {overall_mape:7.2f} %")
print("=" * 65)
