# run_phaseB_suite.py  (place OUTSIDE the ths_zero/ package)
from __future__ import annotations

import argparse
import json
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

    return {"variant": name, "PASS": bool(okA and okB)}


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

    results = []

    # --- Must-PASS set ---
    results.append(run_once_B("B00_baseline", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))

    results.append(run_once_B("B01_cold_Tamb-10C", wltc, common0, veh0, batt0, init0, replace(env0, Tamb_C=-10.0), args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))
    results.append(run_once_B("B02_hot_Tamb45C", wltc, common0, veh0, batt0, init0, replace(env0, Tamb_C=45.0), args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))

    results.append(run_once_B("B03_lowSOC_start", wltc, common0, veh0, batt0, replace(init0, soc0=0.35), env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))
    results.append(run_once_B("B04_highSOC_start", wltc, common0, veh0, batt0, replace(init0, soc0=0.75), env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))

    results.append(run_once_B("B05_high_road_load", wltc, common0, replace(veh0, mass_kg=veh0.mass_kg*1.2, Crr=veh0.Crr*1.3, CdA=veh0.CdA*1.2), batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))

    results.append(run_once_B("B07_grid_coarse_200rpm_10Nm", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=200.0, eng_tq_step=10.0))

    # --- Determinism compare (baseline repeated) ---
    r1 = run_once_B("B09_determinism_run1", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)
    r2 = run_once_B("B09_determinism_run2", wltc, common0, veh0, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0)

    # Compare KPI jsons
    def _load_json(p: str) -> Dict[str, Any]:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    k1 = _load_json(os.path.join(args.out_dir, "B09_determinism_run1", "kpis_phaseB.json"))
    k2 = _load_json(os.path.join(args.out_dir, "B09_determinism_run2", "kpis_phaseB.json"))
    ok_cmp, diffs = _compare_dicts_close(k1, k2)

    vdir_cmp = os.path.join(args.out_dir, "B09_determinism_compare")
    _ensure_dir(vdir_cmp)
    with open(os.path.join(vdir_cmp, "compare_ok.json"), "w", encoding="utf-8") as f:
        json.dump({"PASS": bool(ok_cmp), "diffs": diffs}, f, indent=2)

    results.append({"variant": "B09_determinism_compare", "PASS": bool(ok_cmp)})

    # --- Expected FAIL: force traction shortfall by shrinking MG2 limit ---
    veh_fail = replace(veh0, mg2_tq_max_Nm=max(20.0, veh0.mg2_tq_max_Nm * 0.25))
    results.append(run_once_B("F01_expected_fail_low_mg2_tq_max", wltc, common0, veh_fail, batt0, init0, env0, args.out_dir, eng_rpm_step=100.0, eng_tq_step=5.0))

    # Turn "expected fail" into a PASS condition if Phase A tracking gates fail (shortfall/E_short)
    gate_f = _load_json(os.path.join(args.out_dir, "F01_expected_fail_low_mg2_tq_max", "gate_detail.json"))
    a = gate_f.get("A", {})
    exp_ok = False
    # If either shortfall metric fails, we consider expected-fail detected.
    for key in ["A_shortfall_step_ratio", "A_E_short_over_E_trac"]:
        if isinstance(a.get(key), dict) and (a[key].get("ok") is False):
            exp_ok = True

    vdir_exp = os.path.join(args.out_dir, "F01_expected_fail_check")
    _ensure_dir(vdir_exp)
    with open(os.path.join(vdir_exp, "expected_fail_check.json"), "w", encoding="utf-8") as f:
        json.dump({"PASS": bool(exp_ok), "note": "PASS means expected fail was detected via Phase-A tracking gates."}, f, indent=2)
    results.append({"variant": "F01_expected_fail_check", "PASS": bool(exp_ok)})

    # Summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "suite_summary.csv"), index=False)

    overall_pass = bool(df[df["variant"].str.startswith("B")]["PASS"].all() and exp_ok and ok_cmp)

    with open(os.path.join(args.out_dir, "suite_overall.json"), "w", encoding="utf-8") as f:
        json.dump({"OVERALL_PASS": overall_pass}, f, indent=2)

    print("\n=== Phase B Suite Summary ===")
    print(df.to_string(index=False))
    print(f"\nOVERALL_PASS = {overall_pass}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
