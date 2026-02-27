# ths_zero/derived_B.py
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ===== Derived columns contract (B.1b) =====
DERIVED_FUEL_COLUMNS_REQUIRED: tuple[str, ...] = (
    "fuel_step_g",
    "fuel_cum_g",
    "E_fuel_step_J",
    "E_fuel_cum_J",
)

DERIVED_FUEL_COLUMNS_OPTIONAL: tuple[str, ...] = (
    "fuel_cum_kg",
    "fuel_cum_L",
    "fuel_flow_Lph",
)

_REQUIRED_INPUT_COLS: tuple[str, ...] = (
    "dt_s",
    "mdot_fuel_gps",
    "P_fuel_W",
)


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"timeseries missing required columns for derived fuel integrals: {missing}")


def compute_phaseB_derived_fuel_integrals(
    ts: pd.DataFrame,
    *,
    fuel_density_kg_per_L: float | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Compute derived fuel integrals for Phase B.1b (Option A).

    JP:
      - canonical 85列の契約を変えないため、本関数は「派生列を付与した df」を返すだけ。
      - 保存するかどうかは呼び出し側（run_phaseB / dashboard）が決める。

    EN:
      - To preserve the frozen 85-col contract, this function returns a df with derived columns appended.
      - The caller decides whether to save it (run script) or keep transient (dashboard).

    Required inputs:
      - dt_s (s)
      - mdot_fuel_gps (g/s)
      - P_fuel_W (W)

    Derived outputs (required):
      - fuel_step_g   = mdot_fuel_gps * dt_s
      - fuel_cum_g    = cumsum(fuel_step_g)
      - E_fuel_step_J = P_fuel_W * dt_s
      - E_fuel_cum_J  = cumsum(E_fuel_step_J)

    Optional outputs (if fuel_density_kg_per_L is provided):
      - fuel_cum_kg
      - fuel_cum_L
      - fuel_flow_Lph
    """
    _require_cols(ts, _REQUIRED_INPUT_COLS)

    df = ts.copy() if copy else ts

    dt = df["dt_s"].to_numpy(dtype=float)
    mdot_gps = df["mdot_fuel_gps"].to_numpy(dtype=float)
    P_fuel_W = df["P_fuel_W"].to_numpy(dtype=float)

    # Step integrals
    fuel_step_g = mdot_gps * dt
    df["fuel_step_g"] = fuel_step_g
    df["fuel_cum_g"] = np.cumsum(fuel_step_g)

    E_fuel_step_J = P_fuel_W * dt
    df["E_fuel_step_J"] = E_fuel_step_J
    df["E_fuel_cum_J"] = np.cumsum(E_fuel_step_J)

    # Optional volumetric conversions
    if fuel_density_kg_per_L is not None:
        if fuel_density_kg_per_L <= 0:
            raise ValueError(f"fuel_density_kg_per_L must be > 0, got {fuel_density_kg_per_L}")

        fuel_cum_kg = df["fuel_cum_g"].to_numpy(dtype=float) / 1000.0
        df["fuel_cum_kg"] = fuel_cum_kg
        df["fuel_cum_L"] = fuel_cum_kg / fuel_density_kg_per_L

        mdot_kgps = mdot_gps / 1000.0
        df["fuel_flow_Lph"] = (mdot_kgps / fuel_density_kg_per_L) * 3600.0

    return df


def validate_phaseB_derived_fuel_integrals(df: pd.DataFrame) -> None:
    """Lightweight acceptance checks for B.1b derived columns."""
    _require_cols(df, DERIVED_FUEL_COLUMNS_REQUIRED)

    for c in DERIVED_FUEL_COLUMNS_REQUIRED:
        x = df[c].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"derived column has NaN/inf: {c}")

    # Monotonic non-decreasing checks
    if (np.diff(df["fuel_cum_g"].to_numpy(dtype=float)) < -1e-12).any():
        raise ValueError("fuel_cum_g must be monotonic non-decreasing")
    if (np.diff(df["E_fuel_cum_J"].to_numpy(dtype=float)) < -1e-6).any():
        raise ValueError("E_fuel_cum_J must be monotonic non-decreasing")