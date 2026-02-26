# run_phaseB.py  (place OUTSIDE the ths_zero/ package)
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import argparse

import pandas as pd

# ---- Phase A (firewall OUTSIDE, allowed here) ----
from ths_zero.step_A import solve_step_A, StepInputs, StepWeights
from ths_zero.sim_grid_A import (
    constant_eta,
    constant_bsfc,
    eng_tq_max_flat,
    eng_drag_min_simple,
)

# ---- Phase B (package code) ----
from ths_zero.sim_grid_B import simulate_ths_grid_B, default_output_filenames
from ths_zero.step_B import StepInputsB
from ths_zero.battery_B import BatteryConfigB
from ths_zero.losses_B import SimConfigB
from ths_zero.audit_B import compute_audit_outputs_B
from ths_zero.derived_B import compute_phaseB_derived_fuel_integrals, validate_phaseB_derived_fuel_integrals
from ths_zero.engine_maps import fake_bsfc_g_per_kwh, make_bsfc_map_from_grid_csv

# ---- shared inputs ----
from ths_zero.wltc import load_wltc
from ths_zero.configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig


def solve_step_A_adapter(x, **kwargs):
    """
    Adapter so sim_grid_B can use its default StepInputsB builder while calling Phase A solver.

    Firewall rule:
      - This adapter lives outside ths_zero/PhaseB modules, so importing step_A is OK.
    """
    if isinstance(x, StepInputsB):
        xA = StepInputs(**asdict(x))  # StepInputsB mirrors StepInputs 1:1
    else:
        xA = x
    return solve_step_A(xA, **kwargs)

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run THS Zero Simulator Phase B")
    ap.add_argument("--wltc_csv", default="WLTC3b.csv", help="WLTC CSV path (default: WLTC3b.csv)")
    ap.add_argument("--outdir", default="out_phaseB", help="Output directory (default: out_phaseB)")
    ap.add_argument(
        "--bsfc_csv",
        default="",
        help="Optional BSFC grid CSV (eng_rpm × tq_Nm). If not set, uses fake_bsfc bowl.",
    )
    ap.add_argument(
        "--use_bsfc_in_phaseA",
        action="store_true",
        help="If set, pass the same BSFC map into Phase A solver objective (changes point selection).",
    )
    return ap

def main():
    args = build_argparser().parse_args()
    # ---- paths ----
    wltc_csv = Path(args.wltc_csv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load WLTC ----
    wltc_df = load_wltc(str(wltc_csv))
    print("WLTC rows:", len(wltc_df))
    print("t_s tail:", wltc_df["t_s"].tail(3).tolist())
    print("dt_s tail:", wltc_df["dt_s"].tail(3).tolist())
    assert wltc_df["dt_s"].iloc[-1] == 0.0

    # ---- configs (use defaults or load from json via configs.load_or_default if you have) ----
    common = CommonConfig()
    veh = VehicleConfig()
    batt = BatteryConfig()
    init = InitialState()
    env = EnvironmentConfig()

    # ---- Phase A solver kwargs ----
    weights = StepWeights()

    solver_kwargs = dict(
        weights=weights,
        bsfc_map=constant_bsfc,          # Phase A proxy fuel term (still OK)
        eng_tq_max_map=eng_tq_max_flat,
        eta_mg1_map=constant_eta,
        eta_mg2_map=constant_eta,
        eng_drag_min_map=eng_drag_min_simple,
        eng_rpm_step=100.0,
        eng_tq_step=5.0,
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

    # Important:
    # Phase B fuel accounting uses this bsfc_map (eng_rpm, eng_tq)->g/kWh.
    # You can pass the same callable as Phase A, or a real BSFC map interpolator.
    if str(args.bsfc_csv).strip():
        bsfc_map_B = make_bsfc_map_from_grid_csv(str(args.bsfc_csv))
        print(f"[BSFC] Phase B using grid CSV: {args.bsfc_csv}")
    else:
        bsfc_map_B = fake_bsfc_g_per_kwh
        print("[BSFC] Phase B using fake_bsfc_g_per_kwh (bowl)")

    # Optional: make Phase A objective use the same BSFC map (changes chosen engine points)
    if args.use_bsfc_in_phaseA and str(args.bsfc_csv).strip():
        solver_kwargs["bsfc_map"] = bsfc_map_B
        print("[BSFC] Phase A objective uses the same BSFC map (use_bsfc_in_phaseA=ON)")

    # ---- run Phase B ----
    ts, cons = simulate_ths_grid_B(
        wltc_df,
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
    cons2, kpis, budgets = compute_audit_outputs_B(ts, cons)

    # ---- save ----
    ts_name, cons_name = default_output_filenames()
    ts_path = out_dir / ts_name
    cons_path = out_dir / cons_name

    ts.to_csv(ts_path, index=False)
    cons.to_csv(cons_path, index=False)

    stamp = ts_name.removeprefix("timeseries_phaseB_").removesuffix(".csv")
    
    # ---- Phase B.1b (derived) ----
    # Option A: DO NOT modify the canonical 85-col timeseries.
    # Create a derived view and optionally save as a separate file.
    ts_derived = compute_phaseB_derived_fuel_integrals(
        ts,
        fuel_density_kg_per_L=None,  # set e.g. 0.745 if you want L/Lph
        copy=True,
    )
    validate_phaseB_derived_fuel_integrals(ts_derived)

    ts_derived_path = out_dir / f"timeseries_phaseB_derived_{stamp}.csv"
    ts_derived.to_csv(ts_derived_path, index=False)

    with open(out_dir / f"kpis_phaseB_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)

    with open(out_dir / f"budgets_phaseB_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(budgets, f, indent=2)

    cons2.to_csv(out_dir / f"constraints_phaseB_with_audit_{stamp}.csv", index=False)

    audit_cons_path = out_dir / f"constraints_phaseB_with_audit_{stamp}.csv"

    print("✅ Phase B CSV generated:")
    print(f"  timeseries:        {ts_path}  ({len(ts)} rows × {len(ts.columns)} cols)")
    print(f"  timeseries(derived): {ts_derived_path} ({len(ts_derived)} rows × {len(ts_derived.columns)} cols)")
    print(f"  constraints(raw):  {cons_path} ({len(cons)} rows × {len(cons.columns)} cols)")
    print(f"  constraints(audit):{audit_cons_path} ({len(cons2)} rows × {len(cons2.columns)} cols)")
    print(f"  kpis:              {out_dir / f'kpis_phaseB_{stamp}.json'}")
    print(f"  budgets:           {out_dir / f'budgets_phaseB_{stamp}.json'}")


if __name__ == "__main__":
    main()