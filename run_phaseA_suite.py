# run_phaseA_suite.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from typing import Callable, Dict, Any, Tuple

import numpy as np
import pandas as pd

from ths_zero.wltc import load_wltc
from ths_zero.configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig
from ths_zero.sim_grid_A import simulate_ths_grid_A
from ths_zero.audit import compute_audit_outputs


# -------------------------
# Gate thresholds (Phase A completion)
# -------------------------
GATES = {
    "psd_speed_resid_rms_rpm": 100.0,
    "psd_speed_resid_max_rpm": 300.0,
    "elec_power_resid_rms_W": 100.0,
    "elec_power_resid_max_W": 500.0,
    "soc_recon_resid_rms_pct": 0.05,
    "soc_recon_resid_max_pct": 0.2,
    "shortfall_step_ratio": 0.001,   # 0.1%
    "E_short_over_E_trac": 0.001,    # 0.1%
}

NUM_TOL = 1e-9


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dt_from_ts(ts: pd.DataFrame) -> np.ndarray:
    t = ts["t_s"].to_numpy(dtype=float)
    return np.gradient(t)


def eval_gates(ts: pd.DataFrame, kpis: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
      ok: overall PASS/FAIL
      detail: per-gate results
    """
    detail: Dict[str, Any] = {}

    # G0-ish: basic sanity
    required_cols = ["t_s", "dt_s", "P_wheel_req_W", "P_batt_act_W", "soc_pct", "shortfall_power_W"]
    missing = [c for c in required_cols if c not in ts.columns]
    detail["missing_required_cols"] = missing
    if missing:
        detail["PASS"] = False
        return False, detail

    finite_ok = True
    for c in required_cols:
        arr = ts[c].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            finite_ok = False
            detail[f"finite_{c}"] = False
        else:
            detail[f"finite_{c}"] = True

    # Constraint residual gates
    def _lt(key: str, thr: float) -> bool:
        v = float(kpis.get(key, float("nan")))
        ok = np.isfinite(v) and (v <= thr + NUM_TOL)
        detail[key] = {"value": v, "thr": thr, "ok": ok}
        return ok

    ok_res = True
    ok_res &= _lt("psd_speed_resid_rms_rpm", GATES["psd_speed_resid_rms_rpm"])
    ok_res &= _lt("psd_speed_resid_max_rpm", GATES["psd_speed_resid_max_rpm"])
    ok_res &= _lt("elec_power_resid_rms_W", GATES["elec_power_resid_rms_W"])
    ok_res &= _lt("elec_power_resid_max_W", GATES["elec_power_resid_max_W"])
    ok_res &= _lt("soc_recon_resid_rms_pct", GATES["soc_recon_resid_rms_pct"])
    ok_res &= _lt("soc_recon_resid_max_pct", GATES["soc_recon_resid_max_pct"])

    # Hard-limit gates (overspeed must be zero)
    mg1_os = int(kpis.get("count_flag_mg1_overspeed", -999))
    mg2_os = int(kpis.get("count_flag_mg2_overspeed", -999))
    detail["count_flag_mg1_overspeed"] = {"value": mg1_os, "thr": 0, "ok": (mg1_os == 0)}
    detail["count_flag_mg2_overspeed"] = {"value": mg2_os, "thr": 0, "ok": (mg2_os == 0)}
    ok_limits = (mg1_os == 0) and (mg2_os == 0)

    # Demand tracking: shortfall ratio + energy ratio
    dt = _dt_from_ts(ts)
    P_short = np.maximum(ts["shortfall_power_W"].to_numpy(dtype=float), 0.0)
    short_steps = float(np.mean(P_short > 1e-6))
    detail["shortfall_step_ratio"] = {"value": short_steps, "thr": GATES["shortfall_step_ratio"], "ok": (short_steps <= GATES["shortfall_step_ratio"] + NUM_TOL)}

    P_trac_pos = np.maximum(ts["P_wheel_req_W"].to_numpy(dtype=float), 0.0)
    E_short = float(np.sum(P_short * dt))       # J (since W*s)
    E_trac = float(np.sum(P_trac_pos * dt))     # J
    ratio = (E_short / E_trac) if E_trac > 1e-9 else 0.0
    detail["E_short_over_E_trac"] = {"value": ratio, "thr": GATES["E_short_over_E_trac"], "ok": (ratio <= GATES["E_short_over_E_trac"] + NUM_TOL)}

    ok_track = (short_steps <= GATES["shortfall_step_ratio"] + NUM_TOL) and (ratio <= GATES["E_short_over_E_trac"] + NUM_TOL)

    ok = finite_ok and ok_res and ok_limits and ok_track
    detail["PASS"] = bool(ok)
    return bool(ok), detail


def run_once(
    name: str,
    wltc: pd.DataFrame,
    common: CommonConfig,
    veh: VehicleConfig,
    batt: BatteryConfig,
    init: InitialState,
    env: EnvironmentConfig,
    out_dir: str,
    eng_rpm_step: float = 100.0,
    eng_tq_step: float = 5.0,
    weights=None,
) -> Dict[str, Any]:
    ts, cons = simulate_ths_grid_A(
        wltc=wltc,
        common=common,
        veh=veh,
        batt=batt,
        init=init,
        env=env,
        weights=weights,  # None => default StepWeights()
        eng_rpm_step=eng_rpm_step,
        eng_tq_step=eng_tq_step,
        soc_target=common.soc_target,
        soc_band=common.soc_band,
    )
    cons2, kpis, budgets = compute_audit_outputs(ts, cons)
    ok, gate_detail = eval_gates(ts, kpis)

    # Save artifacts
    vdir = os.path.join(out_dir, name)
    _ensure_dir(vdir)
    ts.to_csv(os.path.join(vdir, "timeseries.csv"), index=False)
    cons2.to_csv(os.path.join(vdir, "constraints.csv"), index=False)
    with open(os.path.join(vdir, "kpis.json"), "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)
    with open(os.path.join(vdir, "budgets.json"), "w", encoding="utf-8") as f:
        json.dump(budgets, f, ensure_ascii=False, indent=2)
    with open(os.path.join(vdir, "gate_detail.json"), "w", encoding="utf-8") as f:
        json.dump(gate_detail, f, ensure_ascii=False, indent=2)

    return {
        "name": name,
        "PASS": ok,
        "kpis": kpis,
        "budgets": budgets,
        "gate_detail": gate_detail,
    }


def kpi_close(a: Dict[str, Any], b: Dict[str, Any], rtol: float = 0.0, atol: float = 1e-9) -> Tuple[bool, Dict[str, float]]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    diffs: Dict[str, float] = {}
    ok = True
    for k in keys:
        va, vb = a[k], b[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if np.isfinite(va) and np.isfinite(vb):
                d = float(abs(float(va) - float(vb)))
                diffs[k] = d
                if not np.isclose(va, vb, rtol=rtol, atol=atol):
                    ok = False
    return ok, diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wltc_csv", type=str, default="WLTC3b.csv")
    ap.add_argument("--out_dir", type=str, default="out_phaseA_suite")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    wltc = load_wltc(args.wltc_csv)

    # Baseline configs
    common0 = CommonConfig(soc_target=0.55, soc_band=0.05)
    veh0 = VehicleConfig()
    batt0 = BatteryConfig()
    init0 = InitialState(soc0=0.55)
    env0 = EnvironmentConfig(Tamb_C=20.0)

    results = []

    # -------------------------
    # A00 Baseline
    # -------------------------
    results.append(run_once("A00_baseline", wltc, common0, veh0, batt0, init0, env0, args.out_dir,
                            eng_rpm_step=100.0, eng_tq_step=5.0, weights=None))

    # -------------------------
    # A01 Cold
    # -------------------------
    env = replace(env0, Tamb_C=-10.0)
    results.append(run_once("A01_cold_Tamb-10C", wltc, common0, veh0, batt0, init0, env, args.out_dir))

    # -------------------------
    # A02 Hot
    # -------------------------
    env = replace(env0, Tamb_C=45.0)
    results.append(run_once("A02_hot_Tamb45C", wltc, common0, veh0, batt0, init0, env, args.out_dir))

    # -------------------------
    # A03 Low SOC start
    # -------------------------
    init = replace(init0, soc0=batt0.soc_min + 0.02)
    results.append(run_once("A03_lowSOC_start", wltc, common0, veh0, batt0, init, env0, args.out_dir))

    # -------------------------
    # A04 High SOC start
    # -------------------------
    init = replace(init0, soc0=batt0.soc_max - 0.02)
    results.append(run_once("A04_highSOC_start", wltc, common0, veh0, batt0, init, env0, args.out_dir))

    # -------------------------
    # A05 High road load (mass +15%, Crr +20%, CdA +20%)
    # -------------------------
    veh = replace(
        veh0,
        mass_kg=veh0.mass_kg * 1.15,
        Crr=veh0.Crr * 1.20,
        CdA=veh0.CdA * 1.20,
    )
    results.append(run_once("A05_high_road_load", wltc, common0, veh, batt0, init0, env0, args.out_dir))

    # -------------------------
    # A07 Grid robustness: coarse (200rpm/10Nm)
    # -------------------------
    results.append(run_once("A07_grid_coarse_200rpm_10Nm", wltc, common0, veh0, batt0, init0, env0, args.out_dir,
                            eng_rpm_step=200.0, eng_tq_step=10.0))

    # -------------------------
    # A09 Determinism: run baseline twice and compare kpis
    # -------------------------
    r1 = run_once("A09_determinism_run1", wltc, common0, veh0, batt0, init0, env0, args.out_dir)
    r2 = run_once("A09_determinism_run2", wltc, common0, veh0, batt0, init0, env0, args.out_dir)
    det_ok, diffs = kpi_close(r1["kpis"], r2["kpis"], rtol=0.0, atol=1e-9)
    det_report = {
        "name": "A09_determinism_compare",
        "PASS": bool(det_ok),
        "max_abs_diff": float(max(diffs.values())) if diffs else 0.0,
        "diffs": diffs,
    }
    with open(os.path.join(args.out_dir, "A09_determinism_compare.json"), "w", encoding="utf-8") as f:
        json.dump(det_report, f, ensure_ascii=False, indent=2)
    results.append(det_report)

    # -------------------------
    # F01 EXPECT-FAIL: reduce MG2 torque limit to force shortfall
    # -------------------------
    veh_fail = replace(veh0, mg2_tq_max_Nm=100.0)
    r_fail = run_once("F01_expected_fail_low_mg2_tq_max", wltc, common0, veh_fail, batt0, init0, env0, args.out_dir)
    # This test PASSES if it FAILS the Phase-A gates (i.e., audit catches it)
    expected_fail_pass = (r_fail["PASS"] is False)
    f01_report = {
        "name": "F01_expected_fail_check",
        "PASS": bool(expected_fail_pass),
        "phaseA_gate_PASS": bool(r_fail["PASS"]),
    }
    with open(os.path.join(args.out_dir, "F01_expected_fail_check.json"), "w", encoding="utf-8") as f:
        json.dump(f01_report, f, ensure_ascii=False, indent=2)
    results.append(f01_report)

    # -------------------------
    # Print summary + overall PASS
    # -------------------------
    summary_rows = []
    overall_ok = True

    # A00..A07 must be PASS
    must_pass_prefix = ("A00_", "A01_", "A02_", "A03_", "A04_", "A05_", "A07_")
    for r in results:
        name = r["name"]
        passed = bool(r["PASS"])
        summary_rows.append({"variant": name, "PASS": passed})
        if name.startswith(must_pass_prefix) and not passed:
            overall_ok = False
        if name == "A09_determinism_compare" and not passed:
            overall_ok = False
        if name == "F01_expected_fail_check" and not passed:
            overall_ok = False

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    print("\n=== Phase A Suite Summary ===")
    print(summary.to_string(index=False))
    print(f"\nOVERALL_PASS = {overall_ok}")

    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())