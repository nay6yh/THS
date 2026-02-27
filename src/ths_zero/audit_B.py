from __future__ import annotations

"""Phase B audit checks (frozen KPIs).

FROZEN AUDIT TARGETS (Phase B):
1) Fuel local balance (per-step):
      P_fuel_W ≈ P_eng_mech_W + loss_engine_W
2) DC bus local balance (per-step, Phase A carryover):
      P_batt_act_W ≈ P_mg1_elec_W + P_mg2_elec_W + P_aux_W
3) SOC reconstruction (Phase B):
      Reconstruct using P_batt_chem_W with the SAME discrete update style as the simulator.
4) Regen metrics must be in [0, 1]:
      regen_utilization = E_regen_to_batt / max(E_wheel_neg, eps)

Import firewall:
- MUST NOT import audit.py (Phase A).
- MUST NOT import step_A.py / sim_grid_A.py at runtime.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-12


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def compute_soc_reconstruction_residual_B(ts: pd.DataFrame) -> pd.Series:
    """Reconstruct SOC by integrating P_batt_chem_W and compare to soc_pct.

    IMPORTANT (frozen semantics):
      - timeseries['E_batt_Wh'] and ['soc_pct'] are treated as **end-of-step** state values.
        (i.e., row k stores the battery energy/SOC *after* applying row k power over dt_s[k]).
      - To avoid an ill-posed initial condition (the energy *before* row 0 is not logged),
        we define the reconstruction residual at k=0 as 0, and start the forward update from k=1.

    Discrete update (matches sim_grid_B chemical update):
        E_end[k] = clip(E_end[k-1] - P_chem[k] * dt[k] / 3600, [Emin[k], Emax[k]])   for k>=1

    Units:
      - soc_pct is in percent [0..100].
      - residual is in percent points.

    Required columns:
      - dt_s
      - P_batt_chem_W
      - batt_E_usable_Wh
      - batt_Emin_Wh
      - batt_Emax_Wh
      - E_batt_Wh
      - soc_pct
    """
    _require_cols(
        ts,
        ["dt_s", "P_batt_chem_W", "batt_E_usable_Wh", "batt_Emin_Wh", "batt_Emax_Wh", "E_batt_Wh", "soc_pct"],
        "timeseries",
    )

    if len(ts) == 0:
        return pd.Series([], dtype=float)

    dt = ts["dt_s"].to_numpy(dtype=float)
    if np.any(dt < -_EPS):
        raise ValueError("dt_s must be non-negative")

    Pchem = ts["P_batt_chem_W"].to_numpy(dtype=float)
    E_usable = ts["batt_E_usable_Wh"].to_numpy(dtype=float)
    Emin = ts["batt_Emin_Wh"].to_numpy(dtype=float)
    Emax = ts["batt_Emax_Wh"].to_numpy(dtype=float)

    E_rep = ts["E_batt_Wh"].to_numpy(dtype=float)
    soc_rep = ts["soc_pct"].to_numpy(dtype=float)

    # End-of-step reconstruction (E_rep[k] is end-of-step)
    E_rec = np.empty_like(E_rep)
    E_rec[0] = E_rep[0]

    for k in range(1, len(E_rec)):
        # use previous row's power to get current end-of-step
        E_next = E_rec[k - 1] - (Pchem[k - 1] * dt[k - 1] / 3600.0)

        # clip using current step limits (k) is fine; or k-1 if you log those as start-of-step
        if np.isfinite(Emin[k]):
            E_next = max(E_next, Emin[k])
        if np.isfinite(Emax[k]):
            E_next = min(E_next, Emax[k])

        E_rec[k] = E_next

    # SOC definition (your test data shows Emin=0 so either formula matches here)
    soc_rec = 100.0 * ((E_rec - Emin) / np.maximum(E_usable, _EPS))
    resid = soc_rec - soc_rep

    # frozen semantics: residual at k=0 defined as 0
    resid[0] = 0.0
    return pd.Series(resid, index=ts.index, dtype=float)

def compute_audit_outputs_B(ts: pd.DataFrame, cons: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Compute Phase B constraints + KPIs + energy budgets.

    Args:
        ts: Phase B timeseries DataFrame (Phase A 71 cols + Phase B ext 14 cols)
        cons: constraints DataFrame (may be empty; will be aligned to ts length)

    Returns:
        cons2: constraints df with Phase B residuals appended
        kpis: dict
        budgets: dict

    Raises:
        ValueError: if required columns are missing.
    """
    # Required for residual checks
    _require_cols(ts, ["dt_s"], "timeseries")
    _require_cols(ts, ["P_fuel_W", "P_eng_mech_W", "loss_engine_W"], "timeseries")
    _require_cols(ts, ["P_batt_act_W", "P_mg1_elec_W", "P_mg2_elec_W", "P_aux_W"], "timeseries")
    _require_cols(ts, ["P_batt_chg_from_regen_W", "P_batt_chg_from_engine_W"], "timeseries")
    _require_cols(ts, ["loss_mg1_W", "loss_mg2_W", "loss_inv1_W", "loss_inv2_W", "loss_inv_W", "loss_batt_W"], "timeseries")
    _require_cols(ts, ["P_wheel_deliv_W_dbg"], "timeseries")

    dt = ts["dt_s"].to_numpy(dtype=float)
    if len(dt) == 0:
        # trivial case
        cons2 = cons.copy() if cons is not None else pd.DataFrame()
        return cons2, {}, {}

    # --- residuals ---
    fuel_balance_resid_W = (ts["P_fuel_W"] - (ts["P_eng_mech_W"] + ts["loss_engine_W"])).abs()
    bus_balance_resid_W = (ts["P_batt_act_W"] - (ts["P_mg1_elec_W"] + ts["P_mg2_elec_W"] + ts["P_aux_W"])).abs()
    soc_recon_resid_pct = compute_soc_reconstruction_residual_B(ts)

    # Align constraints
    if cons is None or len(cons) == 0:
        cons2 = pd.DataFrame(index=ts.index)
    else:
        cons2 = cons.copy()
        if len(cons2) != len(ts):
            # best effort align by index length
            cons2 = cons2.reindex(ts.index)

    cons2["fuel_balance_resid_W"] = fuel_balance_resid_W.to_numpy()
    cons2["bus_balance_resid_W"] = bus_balance_resid_W.to_numpy()
    cons2["soc_recon_resid_pct"] = soc_recon_resid_pct.to_numpy()

    # --- energies / KPIs ---
    P_fuel = ts["P_fuel_W"].to_numpy(dtype=float)
    P_wheel = ts["P_wheel_deliv_W_dbg"].to_numpy(dtype=float)

    E_fuel_MJ = float((P_fuel * dt).sum() / 1e6)
    E_wheel_pos_MJ = float((np.clip(P_wheel, 0.0, None) * dt).sum() / 1e6)
    E_wheel_neg_MJ = float((np.clip(-P_wheel, 0.0, None) * dt).sum() / 1e6)

    P_regen_to_batt = ts["P_batt_chg_from_regen_W"].to_numpy(dtype=float)
    P_engine_to_batt = ts["P_batt_chg_from_engine_W"].to_numpy(dtype=float)
    E_regen_to_batt_MJ = float((P_regen_to_batt * dt).sum() / 1e6)
    E_engine_to_batt_MJ = float((P_engine_to_batt * dt).sum() / 1e6)

    # Optional friction share if available
    E_fric_MJ = None
    if "P_brake_fric_W" in ts.columns:
        P_fric = ts["P_brake_fric_W"].to_numpy(dtype=float)
        E_fric_MJ = float((np.clip(P_fric, 0.0, None) * dt).sum() / 1e6)

    TTW_eff = E_wheel_pos_MJ / max(E_fuel_MJ, 1e-12)
    regen_utilization = E_regen_to_batt_MJ / max(E_wheel_neg_MJ, 1e-12)
    friction_share = None if E_fric_MJ is None else (E_fric_MJ / max(E_wheel_neg_MJ, 1e-12))

    # EV share time: prefer fuel_cut flag, else mdot_fuel_gps==0
    if "fuel_cut" in ts.columns:
        EV_share_time = float((ts["fuel_cut"].to_numpy(dtype=float) > 0.5).mean())
        # start/stop counts from fuel_cut transitions (fuel_cut: 1=OFF, 0=ON)
        fc = ts["fuel_cut"].to_numpy(dtype=int)
        if fc.size >= 2:
            prev = fc[:-1]
            cur = fc[1:]
            count_eng_start = int(np.sum((prev == 1) & (cur == 0)))
            count_eng_stop = int(np.sum((prev == 0) & (cur == 1)))
        else:
            count_eng_start = 0
            count_eng_stop = 0
    else:
        _require_cols(ts, ["mdot_fuel_gps"], "timeseries")
        EV_share_time = float((ts["mdot_fuel_gps"].to_numpy(dtype=float) <= 0.0).mean())
        count_eng_start = int(0)
        count_eng_stop = int(0)
    # EV share energy proxy: fraction of traction energy sourced by battery discharge at terminal
    P_batt_term = ts["P_batt_act_W"].to_numpy(dtype=float)
    E_batt_dis_MJ = float((np.clip(P_batt_term, 0.0, None) * dt).sum() / 1e6)
    EV_share_energy_proxy = E_batt_dis_MJ / max(E_wheel_pos_MJ, 1e-12)

    # Residual summaries (useful for validation gates)
    fuel_balance_resid_max_W = float(np.max(fuel_balance_resid_W.to_numpy(dtype=float)))
    bus_balance_resid_max_W = float(np.max(bus_balance_resid_W.to_numpy(dtype=float)))
    soc_recon_resid_max_abs_pct = float(np.max(np.abs(soc_recon_resid_pct.to_numpy(dtype=float))))

    # Duration / average fuel power (for relative residual checks)
    duration_s = float(np.sum(dt))
    P_fuel_avg_W = float((P_fuel * dt).sum() / max(duration_s, 1e-12))

    # ---- B.1c: derived fuel KPIs (append-only; does NOT change timeseries contract) ----
    mdot = ts["mdot_fuel_gps"].to_numpy(dtype=float) if "mdot_fuel_gps" in ts.columns else None
    if mdot is not None:
        fuel_total_g = float((mdot * dt).sum())
        fuel_on_time_ratio = float((mdot > 0.0).mean())
    else:
        fuel_total_g = float("nan")
        fuel_on_time_ratio = float("nan")

    # ---- B.1d: distance-normalized KPIs (append-only) ----
    if "veh_spd_mps" in ts.columns:
        v_mps = ts["veh_spd_mps"].to_numpy(dtype=float)
        distance_km = float((np.clip(v_mps, 0.0, None) * dt).sum() / 1000.0)
    else:
        distance_km = float("nan")

    fuel_g_per_km = fuel_total_g / max(distance_km, 1e-12) if np.isfinite(distance_km) else float("nan")
    fuel_g_per_100km = fuel_g_per_km * 100.0 if np.isfinite(fuel_g_per_km) else float("nan")

    E_fuel_MJ_per_km = E_fuel_MJ / max(distance_km, 1e-12) if np.isfinite(distance_km) else float("nan")
    E_wheel_pos_MJ_per_km = E_wheel_pos_MJ / max(distance_km, 1e-12) if np.isfinite(distance_km) else float("nan")

    # Implied BSFC from (mdot, P_eng_mech): bsfc = mdot[g/s]*3600 / (P[kW])
    P_eng_mech = ts["P_eng_mech_W"].to_numpy(dtype=float)
    if mdot is not None:
        mask_bsfc = (P_eng_mech > 1.0) & (P_fuel > 1.0) & (mdot > 0.0)
        if mask_bsfc.any():
            P_kW = P_eng_mech[mask_bsfc] / 1000.0
            bsfc = (mdot[mask_bsfc] * 3600.0) / np.maximum(P_kW, 1e-12)  # g/kWh
            bsfc = bsfc[np.isfinite(bsfc)]
        else:
            bsfc = np.array([], dtype=float)
    else:
        bsfc = np.array([], dtype=float)

    def _q(x: np.ndarray, q: float) -> float:
        return float(np.quantile(x, q)) if x.size else float("nan")

    bsfc_mean = float(bsfc.mean()) if bsfc.size else float("nan")

    # Engine efficiency: eta = P_eng_mech / P_fuel (fuel_on only)
    mask_fuel = (P_fuel > 1.0) & (P_eng_mech > 1.0)
    if mdot is not None:
        mask_fuel &= (mdot > 0.0)

    if mask_fuel.any():
        eta = (P_eng_mech[mask_fuel] / np.maximum(P_fuel[mask_fuel], 1e-12))
        eta = eta[np.isfinite(eta)]
    else:
        eta = np.array([], dtype=float)

    eta_mean = float(eta.mean()) if eta.size else float("nan")
    eta_max = float(eta.max()) if eta.size else float("nan")

    kpis: Dict[str, Any] = {
        "TTW_eff": TTW_eff,
        "regen_utilization": regen_utilization,
        "EV_share_time": EV_share_time,
        "EV_share_energy_proxy": EV_share_energy_proxy,
        "duration_s": duration_s,
        "P_fuel_avg_W": P_fuel_avg_W,
        "fuel_balance_resid_max_W": fuel_balance_resid_max_W,
        "bus_balance_resid_max_W": bus_balance_resid_max_W,
        "soc_recon_resid_max_abs_pct": soc_recon_resid_max_abs_pct,
        "E_fuel_MJ": E_fuel_MJ,
        "E_wheel_pos_MJ": E_wheel_pos_MJ,
        "E_wheel_neg_MJ": E_wheel_neg_MJ,
        "E_regen_to_batt_MJ": E_regen_to_batt_MJ,
        "E_engine_to_batt_MJ": E_engine_to_batt_MJ,
        "fuel_total_g": fuel_total_g,
        "fuel_total_kg": fuel_total_g / 1000.0 if np.isfinite(fuel_total_g) else float("nan"),
        "fuel_on_time_ratio": fuel_on_time_ratio,
        "count_eng_start": count_eng_start,
        "count_eng_stop": count_eng_stop,

        "bsfc_implied_mean_g_per_kWh": bsfc_mean,
        "bsfc_implied_p50_g_per_kWh": _q(bsfc, 0.50),
        "bsfc_implied_p90_g_per_kWh": _q(bsfc, 0.90),

        "eta_eng_mean": eta_mean,
        "eta_eng_p50": _q(eta, 0.50),
        "eta_eng_p90": _q(eta, 0.90),
        "eta_eng_max": eta_max,

        "distance_km": distance_km,
        "fuel_g_per_km": fuel_g_per_km,
        "fuel_g_per_100km": fuel_g_per_100km,
        "E_fuel_MJ_per_km": E_fuel_MJ_per_km,
        "E_wheel_pos_MJ_per_km": E_wheel_pos_MJ_per_km,
    }
    if E_fric_MJ is not None:
        kpis["E_fric_MJ"] = E_fric_MJ
    if friction_share is not None:
        kpis["friction_share"] = friction_share

    # --- budgets ---
    loss_engine = ts["loss_engine_W"].to_numpy(dtype=float)
    loss_mg = (ts["loss_mg1_W"].to_numpy(dtype=float) + ts["loss_mg2_W"].to_numpy(dtype=float))
    loss_inv = ts["loss_inv_W"].to_numpy(dtype=float)
    loss_batt = ts["loss_batt_W"].to_numpy(dtype=float)

    E_loss_engine_MJ = float((loss_engine * dt).sum() / 1e6)
    E_loss_mg_MJ = float((loss_mg * dt).sum() / 1e6)
    E_loss_inv_MJ = float((loss_inv * dt).sum() / 1e6)
    E_loss_batt_MJ = float((loss_batt * dt).sum() / 1e6)

    budgets: Dict[str, Any] = {
        "E_fuel_MJ": E_fuel_MJ,
        "E_loss_engine_MJ": E_loss_engine_MJ,
        "E_loss_mg_MJ": E_loss_mg_MJ,
        "E_loss_inv_MJ": E_loss_inv_MJ,
        "E_loss_batt_MJ": E_loss_batt_MJ,
        "E_loss_total_MJ": E_loss_engine_MJ + E_loss_mg_MJ + E_loss_inv_MJ + E_loss_batt_MJ,
    }

    return cons2, kpis, budgets


def validate_audit_outputs_B(
    kpis: Dict[str, Any],
    budgets: Dict[str, Any],
    cons2: pd.DataFrame,
    *,
    strict: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Validate Phase B audit outputs against frozen/sanity constraints.

    This is intended to be used as a CI / gating function.

    Args:
        kpis: KPI dict returned by compute_audit_outputs_B.
        budgets: Energy budget dict returned by compute_audit_outputs_B.
        cons2: constraints df returned by compute_audit_outputs_B (contains residual columns).
        strict: If True, raise ValueError on any failed check.

    Returns:
        violations: dict keyed by check name, each value includes:
          - value: measured value
          - threshold: threshold or (min,max)
          - passed: bool

    Raises:
        ValueError: if strict=True and any check fails.
    """
    violations: Dict[str, Dict[str, Any]] = {}

    def _add(name: str, value: Any, threshold: Any, passed: bool) -> None:
        violations[name] = {"value": value, "threshold": threshold, "passed": bool(passed)}

    regen = float(kpis.get("regen_utilization", float("nan")))
    _add("regen_utilization_in_0_1", regen, (0.0, 1.0), (0.0 <= regen <= 1.0))

    ttw = float(kpis.get("TTW_eff", float("nan")))
    # Loose sanity range (HEV typical ~0.3-0.4, but allow wide band)
    _add("TTW_eff_sanity", ttw, (0.05, 0.8), (0.05 <= ttw <= 0.8))

    # Residual magnitude checks
    soc_resid = float(kpis.get("soc_recon_resid_max_abs_pct", float("nan")))
    _add("soc_recon_resid_pct_lt_1", soc_resid, 1.0, (abs(soc_resid) < 1.0))

    fuel_resid_max = float(kpis.get("fuel_balance_resid_max_W", float("nan")))
    P_fuel_avg = float(kpis.get("P_fuel_avg_W", float("nan")))
    # Relative residual threshold (default 2% of avg fuel power)
    if P_fuel_avg > 0:
        rel = fuel_resid_max / max(P_fuel_avg, 1e-12)
        _add("fuel_balance_resid_rel_lt_0_02", rel, 0.02, (rel < 0.02))
    else:
        # if no fuel, allow any (fuel-cut full cycle)
        _add("fuel_balance_resid_rel_lt_0_02", float("nan"), 0.02, True)

    bus_resid_max = float(kpis.get("bus_balance_resid_max_W", float("nan")))
    # Absolute threshold on bus residual (should be near 0); keep loose
    _add("bus_balance_resid_max_W_lt_50", bus_resid_max, 50.0, (bus_resid_max < 50.0))

    # Energy budget non-negativity
    for key in ["E_loss_engine_MJ", "E_loss_mg_MJ", "E_loss_inv_MJ", "E_loss_batt_MJ", "E_loss_total_MJ"]:
        val = float(budgets.get(key, float("nan")))
        _add(f"{key}_ge_0", val, 0.0, (val >= -1e-12))

    # If strict, raise with a readable message
    failed = {k: v for k, v in violations.items() if not v["passed"]}
    if strict and failed:
        lines = ["Audit validation failed:"]
        for name, info in failed.items():
            lines.append(f"  - {name}: value={info['value']} threshold={info['threshold']}")
        raise ValueError("\n".join(lines))

    return violations
