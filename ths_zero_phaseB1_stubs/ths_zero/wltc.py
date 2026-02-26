from __future__ import annotations

import pandas as pd
import numpy as np


def load_wltc(csv_path: str) -> pd.DataFrame:
    """
    Requires columns:
      Phase, Total_time_s, Speed_kmh
    """
    df = pd.read_csv(csv_path)
    for c in ["Total_time_s", "Speed_kmh"]:
        if c not in df.columns:
            raise ValueError(f"WLTC csv missing '{c}'. columns={list(df.columns)}")

    t = df["Total_time_s"].to_numpy(dtype=float)
    
    if np.any(np.diff(t) <= 0):
        raise ValueError("Total_time_s must be strictly increasing")
    
    dt = (pd.Series(t).diff()).fillna(1.0).to_numpy()
    # last sample has no forward interval
    dt[-1] = 0.0

    v_kmh = df["Speed_kmh"].to_numpy(dtype=float)
    v = v_kmh / 3.6
    a = np.gradient(v, t)

    out = pd.DataFrame({
        "phase": df["Phase"].astype(str) if "Phase" in df.columns else "NA",
        "t_s": t,
        "dt_s": dt,
        "veh_spd_mps": v,
        "veh_spd_kmh": v_kmh,
        "veh_acc_mps2": a,
    })
    return out