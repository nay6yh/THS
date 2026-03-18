# run_phaseB_suite.py  (place OUTSIDE the ths_zero/ package)
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from typing import Any, Dict, Tuple, Callable

import numpy as np
import pandas as pd

from ths_zero.wltc import load_wltc
from ths_zero.configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig

# ---- Phase B (package code) ----
from ths_zero.sim_grid_B import simulate_ths_grid_B
from ths_zero.step_B import StepInputsB
from ths_zero.battery_B import BatteryConfigB
from ths_zero.losses_B import SimConfigB
from ths_zero.audit_B import compute_audit_outputs_B

# ---- Phase A solver (allowed here; NOT inside Phase B modules) ----
from ths_zero.step_A import solve_step_A, StepInputs, StepWeights


# ============================================================
# Gates (Phase B completion)
# ============================================================
NUM_TOL = 1e-9

# Phase A gates that MUST still hold under Phase B
# NOTE: Phase B intentionally updates SOC using chemical power, so we DO NOT gate Phase-A-style SOC recon here.
GATES_A = {
    "psd_speed_resid_rms_rpm": 100.0,
    "psd_speed_resid_max_rpm": 300.0,
    "elec_power_resid_rms_W": 100.0,
    "elec_power_resid_max_W": 500.0,
    "shortfall_step_ratio": 0.001,  # 0.1%
    "E_short_over_E_trac": 0.001,   # 0.1%
}

# Phase B gates (new completion conditions)
GATES_B = {
    "soc_recon_resid_max_abs_pct": 0.01,  # percent-point; should be ~0 if end-of-step semantics match
    "fuel_balance_resid_rel": 0.02,       # 2% of avg fuel power
    "bus_balance_resid_max_W": 50.0,
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def _dt(ts: pd.DataFrame) -> np.ndarray:
    # Prefer dt_s (frozen)
    if "dt_s" in ts.columns:
        return ts["dt_s"].to_numpy(dtype=float)
    t = ts["t_s"].to_numpy(dtype=float)
    dt = np.gradient(t)
    return dt


# ============================================================
# Robust constant map helpers (avoid sim_grid_A signature traps)
# ============================================================
def make_const_eta_map(eta: float) -> Callable[[float, float], float]:
    def f(_rpm: float, _tq: float) -> float:
        return float(eta)
    return f


def make_const_bsfc_map(bsfc_g_per_kWh: float = 250.0) -> Callable[[float, float], float]:
    def f(_rpm: float, _tq: float) -> float:
        return float(bsfc_g_per_kWh)
    return f


def make_eng_tq_max_flat_map(max_tq_Nm: float) -> Callable[[float], float]:
    def f(_rpm: float) -> float:
        return float(max_tq_Nm)
    return f


def make_eng_drag_min_simple_map(drag_Nm: float = 0.0) -> Callable[[float], float]:
    def f(_rpm: float) -> float:
        return float(drag_Nm)
    return f


# ============================================================
# Phase A KPI recomputation from Phase B timeseries
# ============================================================
def compute_phaseA_kpis_from_ts(ts: pd.DataFrame, veh: VehicleConfig) -> Dict[str, Any]:
    req = [
        "dt_s",
        "ring_rpm", "mg1_rpm", "eng_rpm",
        "P_batt_act_W", "P_mg1_elec_W", "P_mg2_elec_W", "P_aux_W",
        "shortfall_power_W", "P_wheel_req_W",
        "flag_mg1_overspeed", "flag_mg2_overspeed",
        "flag_batt_sat", "flag_eng_sat", "flag_mg2_sat",
    ]
    missing = [c for c in req if c not in ts.columns]
    if missing:
        raise ValueError(f"PhaseA KPI recompute missing columns: {missing}")

    dt = _dt(ts)

    # PSD residual
    Zs = float(veh.Zs)
    Zr = float(veh.Zr)
    alpha = Zs / (Zs + Zr)
    beta = Zr / (Zs + Zr)
    resid_psd = ts["eng_rpm"].to_numpy(dtype=float) - (
        alpha * ts["mg1_rpm"].to_numpy(dtype=float) + beta * ts["ring_rpm"].to_numpy(dtype=float)
    )

    # DC bus electrical residual
    resid_bus = ts["P_batt_act_W"].to_numpy(dtype=float) - (
        ts["P_mg1_elec_W"].to_numpy(dtype=float)
        + ts["P_mg2_elec_W"].to_numpy(dtype=float)
        + ts["P_aux_W"].to_numpy(dtype=float)
    )

    kpis: Dict[str, Any] = {
        "psd_speed_resid_rms_rpm": _rms(resid_psd),
        "psd_speed_resid_max_rpm": float(np.max(np.abs(resid_psd))),
        "elec_power_resid_rms_W": _rms(resid_bus),
        "elec_power_resid_max_W": float(np.max(np.abs(resid_bus))),
        "count_flag_mg1_overspeed": int(np.sum(ts["flag_mg1_overspeed"].to_numpy(dtype=float) > 0.5)),
        "count_flag_mg2_overspeed": int(np.sum(ts["flag_mg2_overspeed"].to_numpy(dtype=float) > 0.5)),
        "count_flag_batt_sat": int(np.sum(ts["flag_batt_sat"].to_numpy(dtype=float) > 0.5)),
        "count_flag_eng_sat": int(np.sum(ts["flag_eng_sat"].to_numpy(dtype=float) > 0.5)),
        "count_flag_mg2_sat": int(np.sum(ts["flag_mg2_sat"].to_numpy(dtype=float) > 0.5)),
    }

    # Demand tracking (shortfall)
    P_short = np.maximum(ts["shortfall_power_W"].to_numpy(dtype=float), 0.0)
    short_steps = float(np.mean(P_short > 1e-6))
    kpis["shortfall_step_ratio"] = short_steps

    P_trac_pos = np.maximum(ts["P_wheel_req_W"].to_numpy(dtype=float), 0.0)
    E_short = float(np.sum(P_short * dt))
    E_trac = float(np.sum(P_trac_pos * dt))
    kpis["E_short_over_E_trac"] = (E_short / E_trac) if E_trac > 1e-9 else 0.0

    return kpis


def eval_phaseA_gates(kpisA: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    d: Dict[str, Any] = {}

    def _lt(key: str, thr: float) -> bool:
        v = float(kpisA.get(key, float("nan")))
        ok = np.isfinite(v) and (v <= thr + NUM_TOL)
        d[f"A_{key}"] = {"value": v, "thr": thr, "ok": bool(ok)}
        return bool(ok)

    ok = True
    ok &= _lt("psd_speed_resid_rms_rpm", GATES_A["psd_speed_resid_rms_rpm"])
    ok &= _lt("psd_speed_resid_max_rpm", GATES_A["psd_speed_resid_max_rpm"])
    ok &= _lt("elec_power_resid_rms_W", GATES_A["elec_power_resid_rms_W"])
    ok &= _lt("elec_power_resid_max_W", GATES_A["elec_power_resid_max_W"])

    # Overspeed must be zero
    mg1_os = int(kpisA.get("count_flag_mg1_overspeed", -999))
    mg2_os = int(kpisA.get("count_flag_mg2_overspeed", -999))
    d["A_count_flag_mg1_overspeed"] = {"value": mg1_os, "thr": 0, "ok": (mg1_os == 0)}
    d["A_count_flag_mg2_overspeed"] = {"value": mg2_os, "thr": 0, "ok": (mg2_os == 0)}
    ok &= (mg1_os == 0) and (mg2_os == 0)

    # Tracking
    ok &= _lt("shortfall_step_ratio", GATES_A["shortfall_step_ratio"])
    ok &= _lt("E_short_over_E_trac", GATES_A["E_short_over_E_trac"])

    d["PASS"] = bool(ok)
    return bool(ok), d


# ============================================================
# Phase B checks from timeseries (independent of audit_B)
# ============================================================
def _soc_recon_resid_pct_end_of_step(ts: pd.DataFrame) -> np.ndarray:
    """
    End-of-step SOC recon (percent points), matching sim_grid_B chemical update:
        E_end[k] = clip(E_end[k-1] - P_chem[k]*dt[k]/3600, [Emin[k],Emax[k]])  (k>=1)
        soc_end[k] = 100 * E_end[k]/E_usable[k]
        resid[k] = soc_end[k] - soc_rep[k]
    Convention: resid[0]=0.
    """
    req = ["dt_s", "P_batt_chem_W", "E_batt_Wh", "soc_pct", "batt_E_usable_Wh", "batt_Emin_Wh", "batt_Emax_Wh"]
    missing = [c for c in req if c not in ts.columns]
    if missing:
        raise ValueError(f"SOC recon missing required columns: {missing}")

    dt = ts["dt_s"].to_numpy(dtype=float)
    P = ts["P_batt_chem_W"].to_numpy(dtype=float)
    E_usable = ts["batt_E_usable_Wh"].to_numpy(dtype=float)
    Emin = ts["batt_Emin_Wh"].to_numpy(dtype=float)
    Emax = ts["batt_Emax_Wh"].to_numpy(dtype=float)

    E_rep = ts["E_batt_Wh"].to_numpy(dtype=float)
    soc_rep = ts["soc_pct"].to_numpy(dtype=float)

    n = len(ts)
    resid = np.zeros(n, dtype=float)
    if n == 0:
        return resid

    # Scale sanity: soc_pct must be percent [0..100], not fraction [0..1]
    med = float(np.nanmedian(soc_rep))
    if np.isfinite(med) and (med < 2.0):
        # almost certainly 0..1
        raise ValueError(f"soc_pct scale looks like 0..1 (median={med}). Phase B contract requires percent 0..100.")

    E_rec = np.empty(n, dtype=float)
    E_rec[0] = E_rep[0]

    for k in range(1, n):
        E_next = E_rec[k - 1] - (P[k] * dt[k] / 3600.0)
        if np.isfinite(Emin[k]):
            E_next = max(E_next, Emin[k])
        if np.isfinite(Emax[k]):
            E_next = min(E_next, Emax[k])
        E_rec[k] = E_next

    soc_rec = 100.0 * (E_rec / np.maximum(E_usable, 1e-12))
    resid = soc_rec - soc_rep
    resid[0] = 0.0
    return resid


def eval_phaseB_gates(ts: pd.DataFrame, kpisB: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    d: Dict[str, Any] = {}

    # SOC recon (independent)
    soc_resid = float(np.max(np.abs(_soc_recon_resid_pct_end_of_step(ts))))
    d["B_soc_recon_resid_max_abs_pct_ref"] = {"value": soc_resid, "thr": GATES_B["soc_recon_resid_max_abs_pct"], "ok": (soc_resid <= GATES_B["soc_recon_resid_max_abs_pct"] + 1e-12)}

    # Also check audit_B's KPI if present (to ensure audit_B is consistent)
    soc_audit = float(kpisB.get("soc_recon_resid_max_abs_pct", float("nan")))
    if np.isfinite(soc_audit):
        d["B_soc_recon_resid_max_abs_pct"] = {"value": soc_audit, "thr": GATES_B["soc_recon_resid_max_abs_pct"], "ok": (abs(soc_audit) <= GATES_B["soc_recon_resid_max_abs_pct"] + 1e-12)}
    else:
        d["B_soc_recon_resid_max_abs_pct"] = {"value": soc_audit, "thr": GATES_B["soc_recon_resid_max_abs_pct"], "ok": False}

    # Fuel balance residual rel (from audit_B)
    fuel_resid = float(kpisB.get("fuel_balance_resid_max_W", float("nan")))
    P_fuel_avg = float(kpisB.get("P_fuel_avg_W", float("nan")))
    rel = float("nan")
    ok_fuel = True
    if np.isfinite(fuel_resid) and np.isfinite(P_fuel_avg) and P_fuel_avg > 0:
        rel = fuel_resid / max(P_fuel_avg, 1e-12)
        ok_fuel = (rel <= GATES_B["fuel_balance_resid_rel"] + 1e-12)
    d["B_fuel_balance_resid_rel"] = {"value": rel, "thr": GATES_B["fuel_balance_resid_rel"], "ok": bool(ok_fuel)}

    # Bus balance residual max (from audit_B)
    bus_resid = float(kpisB.get("bus_balance_resid_max_W", float("nan")))
    ok_bus = np.isfinite(bus_resid) and (bus_resid <= GATES_B["bus_balance_resid_max_W"] + 1e-12)
    d["B_bus_balance_resid_max_W"] = {"value": bus_resid, "thr": GATES_B["bus_balance_resid_max_W"], "ok": bool(ok_bus)}

    # Sanity: regen in [0,1]
    regen = float(kpisB.get("regen_utilization", float("nan")))
    ok_regen = np.isfinite(regen) and (0.0 - 1e-12 <= regen <= 1.0 + 1e-12)
    d["B_regen_utilization_in_0_1"] = {"value": regen, "thr": (0.0, 1.0), "ok": bool(ok_regen)}

    # Sanity: TTW_eff
    ttw = float(kpisB.get("TTW_eff", float("nan")))
    ok_ttw = np.isfinite(ttw) and (0.05 <= ttw <= 0.8)
    d["B_TTW_eff_sanity"] = {"value": ttw, "thr": (0.05, 0.8), "ok": bool(ok_ttw)}

    ok = (
        bool(d["B_soc_recon_resid_max_abs_pct_ref"]["ok"])
        and bool(d["B_soc_recon_resid_max_abs_pct"]["ok"])
        and bool(d["B_fuel_balance_resid_rel"]["ok"])
        and bool(d["B_bus_balance_resid_max_W"]["ok"])
        and bool(d["B_regen_utilization_in_0_1"]["ok"])
        and bool(d["B_TTW_eff_sanity"]["ok"])
    )
    d["PASS"] = bool(ok)
    return bool(ok), d


# ============================================================
# Runner
# ============================================================
def solve_step_A_adapter(x, **kwargs):
    if isinstance(x, StepInputsB):
        xA = StepInputs(**asdict(x))
    else:
        xA = x
    return solve_step_A(xA, **kwargs)


def run_once_B(
    name: str,
    wltc: pd.DataFrame,
    common: CommonConfig,
    veh: VehicleConfig,
    batt: BatteryConfig,
    init: InitialState,
    env: EnvironmentConfig,
    out_dir: str,
    *,
    eng_rpm_step: float,
    eng_tq_step: float,
    engine_gate: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    vdir = os.path.join(out_dir, name)
    _ensure_dir(vdir)

    # ---- Phase A solver kwargs (use robust local map closures) ----
    weights = StepWeights()

    solver_kwargs = dict(
        weights=weights,
        bsfc_map=make_const_bsfc_map(250.0),
        eng_tq_max_map=make_eng_tq_max_flat_map(veh.eng_tq_max_Nm),
        eta_mg1_map=make_const_eta_map(veh.eta_mg1),
        eta_mg2_map=make_const_eta_map(veh.eta_mg2),
        eng_drag_min_map=make_eng_drag_min_simple_map(0.0),
        eng_rpm_step=eng_rpm_step,
        eng_tq_step=eng_tq_step,
        soc_target=common.soc_target,
        soc_band=common.soc_band,
    )

    # ---- Phase B configs ----
    batt_cfg_B = BatteryConfigB(
        model="B0",
        eta_charge=batt.eta_charge,
        eta_discharge=batt.eta_discharge,
    )
    sim_cfg_B = SimConfigB(inv_loss_mode="embedded_in_mg_eta")

    # Use the same BSFC proxy for Phase B accounting (constant)
    bsfc_map_B = make_const_bsfc_map(250.0)

    # ---- simulate ----
    ts, cons = simulate_ths_grid_B(
        wltc,
        common=common,
        veh=veh,
        batt=batt,
        init=init,
        env=env,
        batt_cfg=batt_cfg_B,
        sim_cfg=sim_cfg_B,
        bsfc_map=bsfc_map_B,
        solve_step_base_fn=solve_step_A_adapter,
        solver_kwargs=solver_kwargs,
        engine_gate=engine_gate,
    )

    # ---- audits ----
    cons2, kpisB, budgetsB = compute_audit_outputs_B(ts, cons)
    kpisA = compute_phaseA_kpis_from_ts(ts, veh)

    # ---- gate ----
    try:
        okA, gateA = eval_phaseA_gates(kpisA)
    except Exception as e:
        okA, gateA = False, {"PASS": False, "error": f"PhaseA gate eval error: {type(e).__name__}: {e}"}

    try:
        okB, gateB = eval_phaseB_gates(ts, kpisB)
    except Exception as e:
        okB, gateB = False, {"PASS": False, "error": f"PhaseB gate eval error: {type(e).__name__}: {e}"}

    gate_detail = {"A": gateA, "B": gateB, "PASS": bool(okA and okB)}

    # ---- save artifacts ----
    ts.to_csv(os.path.join(vdir, "timeseries.csv"), index=False)
    cons.to_csv(os.path.join(vdir, "constraints.csv"), index=False)
    cons2.to_csv(os.path.join(vdir, "constraints_with_audit.csv"), index=False)

    with open(os.path.join(vdir, "kpis_phaseA.json"), "w", encoding="utf-8") as f:
        json.dump(kpisA, f, indent=2)

    with open(os.path.join(vdir, "kpis_phaseB.json"), "w", encoding="utf-8") as f:
        json.dump(kpisB, f, indent=2)

    with open(os.path.join(vdir, "budgets_phaseB.json"), "w", encoding="utf-8") as f:
        json.dump(budgetsB, f, indent=2)

    with open(os.path.join(vdir, "gate_detail.json"), "w", encoding="utf-8") as f:
        json.dump(gate_detail, f, indent=2)

    # one-row gate debug counters from simulate_ths_grid_B (if gate enabled)
    if not cons.empty:
        cons.to_json(os.path.join(vdir, "gate_counters.json"), orient="records", indent=2)

    return {"variant": name, "PASS": bool(okA and okB)}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_gate_counters_or_empty(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, list) and rows:
        return dict(rows[0])
    return {}


def _make_baseline_compare(out_dir: str, baseline_variant: str, candidate_variant: str, out_name: str) -> Dict[str, Any]:
    base_dir = os.path.join(out_dir, baseline_variant)
    cand_dir = os.path.join(out_dir, candidate_variant)

    k_base = _load_json(os.path.join(base_dir, "kpis_phaseB.json"))
    k_cand = _load_json(os.path.join(cand_dir, "kpis_phaseB.json"))
    g_base = _load_gate_counters_or_empty(os.path.join(base_dir, "gate_counters.json"))
    g_cand = _load_gate_counters_or_empty(os.path.join(cand_dir, "gate_counters.json"))

    compare = {
        "baseline": baseline_variant,
        "candidate": candidate_variant,
        "delta": {
            "count_eng_start": float(k_cand.get("count_eng_start", float("nan"))) - float(k_base.get("count_eng_start", float("nan"))),
            "fuel_g_per_km": float(k_cand.get("fuel_g_per_km", float("nan"))) - float(k_base.get("fuel_g_per_km", float("nan"))),
            "fuel_balance_resid_max_W": float(k_cand.get("fuel_balance_resid_max_W", float("nan"))) - float(k_base.get("fuel_balance_resid_max_W", float("nan"))),
            "bus_balance_resid_max_W": float(k_cand.get("bus_balance_resid_max_W", float("nan"))) - float(k_base.get("bus_balance_resid_max_W", float("nan"))),
            "regen_utilization": float(k_cand.get("regen_utilization", float("nan"))) - float(k_base.get("regen_utilization", float("nan"))),
        },
        "gate_debug": {
            "baseline": g_base,
            "candidate": g_cand,
        },
    }

    compare["criteria"] = {
        "chattering_reduced": bool(compare["delta"]["count_eng_start"] <= 0.0),
        "residuals_preserved": bool(
            float(k_cand.get("fuel_balance_resid_max_W", float("inf"))) <= float(k_base.get("fuel_balance_resid_max_W", float("inf"))) + 1e-9
            and float(k_cand.get("bus_balance_resid_max_W", float("inf"))) <= float(k_base.get("bus_balance_resid_max_W", float("inf"))) + 1e-9
        ),
        "fuel_not_worse": bool(compare["delta"]["fuel_g_per_km"] <= 1e-9),
    }

    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)
    return compare


def _compare_dicts_close(a: Dict[str, Any], b: Dict[str, Any], *, atol: float = 1e-10, rtol: float = 1e-10) -> Tuple[bool, Dict[str, Any]]:
    diffs: Dict[str, Any] = {}
    keys = sorted(set(a.keys()) | set(b.keys()))
    ok = True
    for k in keys:
        va = a.get(k, None)
        vb = b.get(k, None)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not (np.isfinite(va) and np.isfinite(vb)):
                ok = False
                diffs[k] = {"a": va, "b": vb, "ok": False}
                continue
            close = bool(np.isclose(float(va), float(vb), atol=atol, rtol=rtol))
            ok &= close
            if not close:
                diffs[k] = {"a": va, "b": vb, "ok": False}
        else:
            same = (va == vb)
            ok &= same
            if not same:
                diffs[k] = {"a": va, "b": vb, "ok": False}
    return ok, diffs

def _evaluate_determinism_compare(out_dir: str) -> Dict[str, Any]:
    kpi1_path = os.path.join(out_dir, "B09_determinism_run1", "kpis_phaseB.json")
    kpi2_path = os.path.join(out_dir, "B09_determinism_run2", "kpis_phaseB.json")

    missing = [path for path in (kpi1_path, kpi2_path) if not os.path.exists(path)]
    if missing:
        missing_rel = [
            os.path.relpath(path, out_dir).replace(os.sep, "/")
            for path in missing
        ]
        return {
            "PASS": False,
            "primary_failure_reason": "missing_artifact",
            "note": f"missing KPI artifact(s): {', '.join(missing_rel)}",
            "diffs": {},
        }

    k1 = _load_json(kpi1_path)
    k2 = _load_json(kpi2_path)
    ok_cmp, diffs = _compare_dicts_close(k1, k2)
    return {
        "PASS": bool(ok_cmp),
        "primary_failure_reason": "pass" if ok_cmp else "determinism_failed",
        "note": "phaseB KPI JSON equivalence across repeated baseline run",
        "diffs": diffs,
    }


def _first_failed_gate(gate_section: Dict[str, Any], *, startswith: str | None = None) -> str | None:
    for key, value in gate_section.items():
        if not isinstance(value, dict):
            continue
        if startswith is not None and not key.startswith(startswith):
            continue
        if value.get("ok") is False:
            return key
    return None


def _json_safe_triage_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _collect_failed_gates(gate_detail: Dict[str, Any]) -> list[Dict[str, Any]]:
    gate_a = gate_detail.get("A", {}) if isinstance(gate_detail, dict) else {}
    gate_b = gate_detail.get("B", {}) if isinstance(gate_detail, dict) else {}

    failed: list[Dict[str, Any]] = []
    for section_name, section in (("A", gate_a), ("B", gate_b)):
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if not isinstance(value, dict):
                continue
            if value.get("ok") is False:
                failed.append(
                    {
                        "section": section_name,
                        "gate": key,
                        "value": _json_safe_triage_value(value.get("value", None)),
                        "thr": _json_safe_triage_value(value.get("thr", None)),
                    }
                )
    return failed


def _classify_gate_failure(gate_detail: Dict[str, Any]) -> Tuple[str, str]:
    detail = _classify_gate_failure_detail(gate_detail)
    return str(detail["reason"]), str(detail["note"])


def _classify_gate_failure_detail(gate_detail: Dict[str, Any]) -> Dict[str, Any]:
    gate_a = gate_detail.get("A", {}) if isinstance(gate_detail, dict) else {}
    gate_b = gate_detail.get("B", {}) if isinstance(gate_detail, dict) else {}

    fail_a = _first_failed_gate(gate_a, startswith="A_")
    fail_b = _first_failed_gate(gate_b, startswith="B_")

    failed = _collect_failed_gates(gate_detail)
    primary = failed[0] if failed else {}
    secondary = [str(x.get("gate")) for x in failed[1:] if x.get("gate")]

    reason = "missing_artifact"
    if fail_a in {"A_shortfall_step_ratio", "A_E_short_over_E_trac"}:
        reason = "traction_shortfall"
    elif fail_a and ("resid" in fail_a or "overspeed" in fail_a):
        reason = "residual_limit_exceeded"
    elif fail_a:
        reason = "audit_A_failed"
    elif fail_b and "resid" in fail_b:
        reason = "residual_limit_exceeded"
    elif fail_b:
        reason = "audit_B_failed"

    if primary:
        note = (
            f"{primary['gate']} failed "
            f"(value={primary['value']}, thr={primary['thr']})"
        )
    else:
        note = "gate_detail.json had no explicit failed gate key"

    return {
        "reason": reason,
        "note": note,
        "primary_failed_gate": str(primary.get("gate", "")),
        "primary_failed_value": _json_safe_triage_value(primary.get("value", None)),
        "primary_failed_thr": _json_safe_triage_value(primary.get("thr", None)),
        "secondary_failed_gates": "|".join(secondary),
    }


def _expected_fail_detected(gate_detail: Dict[str, Any]) -> Tuple[bool, str]:
    a = gate_detail.get("A", {}) if isinstance(gate_detail, dict) else {}
    for key in ["A_shortfall_step_ratio", "A_E_short_over_E_trac"]:
        if isinstance(a.get(key), dict) and (a[key].get("ok") is False):
            return True, key
    return False, "none"


def _new_summary_row(
    *,
    variant: str,
    passed: bool,
    gating_mode: str,
    primary_failure_reason: str,
    note: str,
    blocking: bool,
    primary_failed_gate: str = "",
    primary_failed_value: Any = None,
    primary_failed_thr: Any = None,
    secondary_failed_gates: str = "",
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "PASS": bool(passed),
        "gating_mode": gating_mode,
        "primary_failure_reason": primary_failure_reason,
        "note": note,
        "blocking": bool(blocking),
        "primary_failed_gate": primary_failed_gate,
        "primary_failed_value": primary_failed_value,
        "primary_failed_thr": primary_failed_thr,
        "secondary_failed_gates": secondary_failed_gates,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wltc_csv", required=True, help="WLTC CSV path (e.g., WLTC3b.csv)")
    ap.add_argument("--out_dir", default="out_phaseB_suite", help="Output directory")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    wltc = load_wltc(args.wltc_csv)
    assert float(wltc["dt_s"].iloc[-1]) == 0.0

    common0 = CommonConfig()
    veh0 = VehicleConfig()
    batt0 = BatteryConfig()
    init0 = InitialState()
    env0 = EnvironmentConfig()

    results: list[Dict[str, Any]] = []

    def append_run(
        name: str,
        *run_args,
        gating_mode: str = "blocking",
        blocking: bool = True,
        **run_kwargs,
    ) -> Dict[str, Any]:
        try:
            result = run_once_B(name, *run_args, **run_kwargs)
        except Exception as e:
            row = _new_summary_row(
                variant=name,
                passed=False,
                gating_mode=gating_mode,
                primary_failure_reason="unexpected_exception",
                note=f"{type(e).__name__}: {e}",
                blocking=blocking,
            )
            results.append(row)
            return row

        gate_path = os.path.join(args.out_dir, name, "gate_detail.json")
        if not os.path.exists(gate_path):
            row = _new_summary_row(
                variant=name,
                passed=False,
                gating_mode=gating_mode,
                primary_failure_reason="missing_artifact",
                note="gate_detail.json not found",
                blocking=blocking,
            )
            results.append(row)
            return row

        gate_detail = _load_json(gate_path)
        if result["PASS"]:
            reason, note = "pass", "all gating checks passed"
            fail_detail = {
                "primary_failed_gate": "",
                "primary_failed_value": None,
                "primary_failed_thr": None,
                "secondary_failed_gates": "",
            }
        else:
            fail_detail = _classify_gate_failure_detail(gate_detail)
            reason, note = str(fail_detail["reason"]), str(fail_detail["note"])
        row = _new_summary_row(
            variant=name,
            passed=result["PASS"],
            gating_mode=gating_mode,
            primary_failure_reason=reason,
            note=note,
            blocking=blocking,
            primary_failed_gate=str(fail_detail["primary_failed_gate"]),
            primary_failed_value=fail_detail["primary_failed_value"],
            primary_failed_thr=fail_detail["primary_failed_thr"],
            secondary_failed_gates=str(fail_detail["secondary_failed_gates"]),
        )
        results.append(row)
        return row

    # --- Must-PASS set ---
    append_run("B00_baseline", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)

    engine_gate_v0 = {
        "relight_fuel_g": 0.30,
    }
    engine_gate_v1 = {
        "relight_fuel_g": 0.30,
        "min_on_s": 8.0,
        "min_off_s": 2.0,
        "override_start_power_W": 30_000.0,
    }

    append_run("B00b_supervisor_start_penalty", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0, engine_gate=engine_gate_v0)
    append_run("B00c_supervisor_min_on_off", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0, engine_gate=engine_gate_v1)

    append_run("B01_cold_Tamb-10C", wltc, common0, veh0, batt0, init0, replace(env0, Tamb_C=-10.0), args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)
    append_run("B02_hot_Tamb45C", wltc, common0, veh0, batt0, init0, replace(env0, Tamb_C=45.0), args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)

    append_run("B03_lowSOC_start", wltc, common0, veh0, batt0, replace(init0, soc0=0.35), env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)
    append_run("B04_highSOC_start", wltc, common0, veh0, batt0, replace(init0, soc0=0.75), env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)

    append_run("B05_high_road_load", wltc, common0, replace(veh0, mass_kg=veh0.mass_kg*1.2, Crr=veh0.Crr*1.3, CdA=veh0.CdA*1.2), batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)

    append_run("B07_grid_coarse_200rpm_10Nm", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=200.0, eng_tq_step=10.0)

    # --- Determinism compare (baseline repeated) ---
    r1 = append_run("B09_determinism_run1", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)
    r2 = append_run("B09_determinism_run2", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)

    det_cmp = _evaluate_determinism_compare(args.out_dir)

    vdir_cmp = os.path.join(args.out_dir, "B09_determinism_compare")
    _ensure_dir(vdir_cmp)
    with open(os.path.join(vdir_cmp, "compare_ok.json"), "w", encoding="utf-8") as f:
        json.dump({"PASS": bool(det_cmp["PASS"]), "diffs": det_cmp["diffs"]}, f, indent=2)

    results.append(_new_summary_row(
        variant="B09_determinism_compare",
        passed=bool(det_cmp["PASS"]),
        gating_mode="blocking",
        primary_failure_reason=str(det_cmp["primary_failure_reason"]),
        note=str(det_cmp["note"]),
        blocking=True,
    ))

    cmp_v0 = _make_baseline_compare(args.out_dir, "B00_baseline", "B00b_supervisor_start_penalty", "B00b_compare_vs_baseline.json")
    cmp_v1 = _make_baseline_compare(args.out_dir, "B00_baseline", "B00c_supervisor_min_on_off", "B00c_compare_vs_baseline.json")
    pass_cmp_v0 = bool(all(cmp_v0["criteria"].values()))
    pass_cmp_v1 = bool(all(cmp_v1["criteria"].values()))
    results.append(_new_summary_row(
        variant="B00b_compare_vs_baseline",
        passed=pass_cmp_v0,
        gating_mode="informational",
        primary_failure_reason="pass" if pass_cmp_v0 else "informational_compare_regression",
        note="non-gating baseline comparison row",
        blocking=False,
    ))
    results.append(_new_summary_row(
        variant="B00c_compare_vs_baseline",
        passed=pass_cmp_v1,
        gating_mode="informational",
        primary_failure_reason="pass" if pass_cmp_v1 else "informational_compare_regression",
        note="non-gating baseline comparison row",
        blocking=False,
    ))

    # --- Expected FAIL: force traction shortfall by shrinking MG2 limit ---
    veh_fail = replace(veh0, mg2_tq_max_Nm=max(20.0, veh0.mg2_tq_max_Nm * 0.25))
    append_run("F01_expected_fail_low_mg2_tq_max", wltc, common0, veh_fail, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0, gating_mode="expected_fail_probe", blocking=False)

    # Turn "expected fail" into a PASS condition if Phase A tracking gates fail (shortfall/E_short)
    gate_f_path = os.path.join(args.out_dir, "F01_expected_fail_low_mg2_tq_max", "gate_detail.json")
    if os.path.exists(gate_f_path):
        gate_f = _load_json(gate_f_path)
        exp_ok, exp_reason = _expected_fail_detected(gate_f)
    else:
        exp_ok, exp_reason = False, "missing_gate_detail"

    vdir_exp = os.path.join(args.out_dir, "F01_expected_fail_check")
    _ensure_dir(vdir_exp)
    with open(os.path.join(vdir_exp, "expected_fail_check.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "PASS": bool(exp_ok),
                "reason": "traction_shortfall" if exp_ok else "expected_fail_not_detected",
                "trigger_gate": exp_reason,
                "note": "PASS means expected fail was detected via Phase-A tracking gates.",
            },
            f,
            indent=2,
        )
    results.append(_new_summary_row(
        variant="F01_expected_fail_check",
        passed=bool(exp_ok),
        gating_mode="blocking_expected_fail",
        primary_failure_reason="pass" if exp_ok else "expected_fail_not_detected",
        note=f"trigger_gate={exp_reason}",
        blocking=True,
    ))

    # Summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "suite_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    overall_pass = bool(df[df["blocking"]]["PASS"].all())

    with open(os.path.join(args.out_dir, "suite_overall.json"), "w", encoding="utf-8") as f:
        json.dump({"OVERALL_PASS": overall_pass}, f, indent=2)

    print("\n=== Phase B Suite Summary ===")
    print(df.to_string(index=False))
    print(f"\nOVERALL_PASS = {overall_pass}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
