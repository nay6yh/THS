from __future__ import annotations

import numpy as np
import pandas as pd


def _rms(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def _amax_abs(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.max(np.abs(x)))


def integrate_J(ts: pd.DataFrame, col: str) -> float:
    if col not in ts.columns:
        return float("nan")
    t = ts["t_s"].to_numpy(dtype=float)
    dt = np.gradient(t)
    return float(np.sum(ts[col].to_numpy(dtype=float) * dt))


def compute_soc_reconstruction_residual(ts: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct SOC by integrating P_batt_act_W and compare to soc_pct.

    IMPORTANT:
    - Use the SAME battery constants and the SAME clipping as the simulator.
    - Never infer E_usable_Wh from data.
    """
    t = ts["t_s"].to_numpy(dtype=float)
    dt = np.gradient(t)

    soc = ts["soc_pct"].to_numpy(dtype=float)
    P = ts["P_batt_act_W"].to_numpy(dtype=float)

    # --- Use constants written by simulator (no inference)
    if "batt_E_usable_Wh" not in ts.columns or "batt_Emin_Wh" not in ts.columns or "batt_Emax_Wh" not in ts.columns:
        raise KeyError("Missing batt_E_usable_Wh/batt_Emin_Wh/batt_Emax_Wh in timeseries. Add them in sim_grid_A.py.")

    Euse = float(ts["batt_E_usable_Wh"].iloc[0])
    Emin = float(ts["batt_Emin_Wh"].iloc[0])
    Emax = float(ts["batt_Emax_Wh"].iloc[0])

    # --- Re-integrate energy WITH the same clipping
    E = np.empty_like(P, dtype=float)
    E[0] = float(ts["E_batt_Wh"].iloc[0])

    for k in range(1, len(E)):
        E[k] = E[k-1] - (P[k] * dt[k]) / 3600.0
        E[k] = np.clip(E[k], Emin, Emax)

    soc_rec = (E / max(Euse, 1e-9)) * 100.0
    resid = soc - soc_rec
    return resid


def compute_audit_outputs(ts: pd.DataFrame, cons: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    # add SOC recon residual to constraints
    soc_resid = compute_soc_reconstruction_residual(ts)
    cons2 = cons.copy()
    cons2["resid_soc_recon_pct"] = soc_resid

    # KPIs
    kpis = {
        "psd_speed_resid_rms_rpm": _rms(cons2["resid_psd_speed_rpm"].to_numpy(dtype=float)),
        "psd_speed_resid_max_rpm": _amax_abs(cons2["resid_psd_speed_rpm"].to_numpy(dtype=float)),

        "ring_torque_resid_rms_Nm": _rms(cons2["resid_ring_torque_Nm"].to_numpy(dtype=float)),
        "ring_torque_resid_max_Nm": _amax_abs(cons2["resid_ring_torque_Nm"].to_numpy(dtype=float)),

        "wheel_power_resid_rms_W": _rms(cons2["resid_wheel_power_W"].to_numpy(dtype=float)),
        "wheel_power_resid_max_W": _amax_abs(cons2["resid_wheel_power_W"].to_numpy(dtype=float)),

        "elec_power_resid_rms_W": _rms(cons2["resid_elec_power_W"].to_numpy(dtype=float)),
        "elec_power_resid_max_W": _amax_abs(cons2["resid_elec_power_W"].to_numpy(dtype=float)),

        "soc_recon_resid_rms_pct": _rms(cons2["resid_soc_recon_pct"].to_numpy(dtype=float)),
        "soc_recon_resid_max_pct": _amax_abs(cons2["resid_soc_recon_pct"].to_numpy(dtype=float)),

        "soc_start_pct": float(ts["soc_pct"].iloc[0]),
        "soc_end_pct": float(ts["soc_pct"].iloc[-1]),
        "soc_delta_pct": float(ts["soc_pct"].iloc[-1] - ts["soc_pct"].iloc[0]),

        "count_flag_eng_sat": int(ts["flag_eng_sat"].sum()),
        "count_flag_mg1_sat": int(ts["flag_mg1_sat"].sum()),
        "count_flag_mg2_sat": int(ts["flag_mg2_sat"].sum()),
        "count_flag_batt_sat": int(ts["flag_batt_sat"].sum()),
        "count_flag_mg1_overspeed": int(ts["flag_mg1_overspeed"].sum()),
        "count_flag_mg2_overspeed": int(ts["flag_mg2_overspeed"].sum()),
    }

    # energy budgets
    budgets = {
        "E_wheel_req_MJ": integrate_J(ts, "P_wheel_req_W") / 1e6,
        "E_wheel_deliv_MJ": integrate_J(ts, "P_wheel_deliv_W") / 1e6,
        "E_eng_mech_MJ": integrate_J(ts, "P_eng_mech_W") / 1e6,
        "E_batt_act_MJ": integrate_J(ts, "P_batt_act_W") / 1e6,
        "E_batt_req_MJ": integrate_J(ts, "P_batt_req_W") / 1e6,
        "E_brake_fric_MJ": integrate_J(ts, "P_brake_fric_W") / 1e6,
        "E_hvac_MJ": integrate_J(ts, "P_hvac_W") / 1e6,
        "E_aux_MJ": integrate_J(ts, "P_aux_W") / 1e6,
        "E_loop_MJ": integrate_J(ts, "P_loop_W") / 1e6,
    }

    return cons2, kpis, budgets