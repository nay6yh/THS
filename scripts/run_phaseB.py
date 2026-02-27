# scripts/run_phaseB.py
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# === Resolve repo root & make src-layout imports work (fallback) ===
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]  # scripts/ -> repo root
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

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
from ths_zero.derived_B import (
    compute_phaseB_derived_fuel_integrals,
    validate_phaseB_derived_fuel_integrals,
)
from ths_zero.engine_maps import fake_bsfc_g_per_kwh, make_bsfc_map_from_grid_csv

# ---- shared inputs ----
from ths_zero.wltc import load_wltc
from ths_zero.configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    cwd_pp = Path.cwd() / pp
    if cwd_pp.exists():
        return cwd_pp
    repo_pp = REPO_ROOT / pp
    return repo_pp


def _slug(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            keep.append(ch)
        else:
            keep.append("_")
    # collapse
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _try_get(obj: Any, names: list[str]) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def _to_jsonable(obj: Any) -> Any:
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def _try_git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def solve_step_A_adapter(x, **kwargs):
    if isinstance(x, StepInputsB):
        xA = StepInputs(**asdict(x))  # StepInputsB mirrors StepInputs 1:1
    else:
        xA = x
    return solve_step_A(xA, **kwargs)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run THS Zero Simulator Phase B (run-folder layout)")

    ap.add_argument("--wltc_csv", default="data/cycles/WLTC3b.csv")
    ap.add_argument("--out_root", default="artifacts/out_phaseB", help="Base output root (run folders created under here)")

    ap.add_argument(
        "--case",
        default="",
        help="Optional run label used in folder name (e.g., yaris_wltc3b_20degC)",
    )

    ap.add_argument("--bsfc_csv", default="", help="Optional BSFC grid CSV. If not set, uses fake_bsfc bowl.")
    ap.add_argument("--use_bsfc_in_phaseA", action="store_true")

    ap.add_argument("--copy_inputs", action="store_true", help="Copy WLTC/BSFC into inputs/ for reproducibility")
    ap.add_argument("--run_viz", action="store_true", help="Run visualization after simulation into Viz/")
    ap.add_argument("--viz_dpi", type=int, default=160)

    return ap


def main():
    args = build_argparser().parse_args()

    # ---- resolve inputs ----
    wltc_csv = _resolve_path(args.wltc_csv)
    out_root = _resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not wltc_csv.exists():
        raise FileNotFoundError(
            f"WLTC CSV not found: {wltc_csv}\n"
            f"  - CWD:       {Path.cwd()}\n"
            f"  - REPO_ROOT: {REPO_ROOT}\n"
        )

    # ---- configs ----
    common = CommonConfig()
    veh = VehicleConfig()
    batt = BatteryConfig()
    init = InitialState()
    env = EnvironmentConfig()

    # ---- Phase A solver kwargs ----
    weights = StepWeights()
    solver_kwargs = dict(
        weights=weights,
        bsfc_map=constant_bsfc,
        eng_tq_max_map=eng_tq_max_flat,
        eta_mg1_map=constant_eta,
        eta_mg2_map=constant_eta,
        eng_drag_min_map=eng_drag_min_simple,
        eng_rpm_step=100.0,
        eng_tq_step=5.0,
        soc_target=getattr(common, "soc_target", 0.5),
        soc_band=getattr(common, "soc_band", 0.02),
    )

    # ---- Phase B configs ----
    batt_cfg_B = BatteryConfigB(
        model="B0",
        eta_charge=getattr(batt, "eta_charge", 0.95),
        eta_discharge=getattr(batt, "eta_discharge", 0.95),
    )
    sim_cfg_B = SimConfigB(inv_loss_mode="embedded_in_mg_eta")

    # ---- BSFC map ----
    bsfc_tag = "bsfc_fake"
    if str(args.bsfc_csv).strip():
        bsfc_csv_path = _resolve_path(args.bsfc_csv)
        if not bsfc_csv_path.exists():
            raise FileNotFoundError(f"BSFC CSV not found: {bsfc_csv_path}")
        bsfc_map_B = make_bsfc_map_from_grid_csv(str(bsfc_csv_path))
        bsfc_tag = _slug(bsfc_csv_path.stem)
        print(f"[BSFC] Phase B using grid CSV: {bsfc_csv_path}")
    else:
        bsfc_map_B = fake_bsfc_g_per_kwh
        print("[BSFC] Phase B using fake_bsfc_g_per_kwh (bowl)")

    if args.use_bsfc_in_phaseA and str(args.bsfc_csv).strip():
        solver_kwargs["bsfc_map"] = bsfc_map_B
        print("[BSFC] Phase A objective uses the same BSFC map (use_bsfc_in_phaseA=ON)")

    engine_gate = dict(
        start_fuel_g=0.30,
        min_on_s=8.0,
        min_off_s=2.0,
        override_start_power_W=30000.0,
        w_fuel_scale_off=5.0,
        w_fuel_scale_on=0.5,
        w_charge_scale_on=5.0,
        k_short=10.0,
    )

    # ---- build run_id (time + conditions) ----
    now = datetime.now()
    ts_tag = now.strftime("%Y-%m-%d_%H%M%S")
    cycle_tag = _slug(wltc_csv.stem.lower())

    Tamb = _try_get(env, ["Tamb_C", "tamb_C", "ambient_C", "Tamb"])
    tamb_tag = f"Tamb{float(Tamb):.0f}C" if isinstance(Tamb, (int, float)) else "TambNA"

    soc0 = _try_get(init, ["soc_init", "soc0", "soc_init_pct", "soc_pct0"])
    soc_tag = f"SOC{float(soc0):.3f}" if isinstance(soc0, (int, float)) else ""

    case_tag = _slug(args.case) if str(args.case).strip() else ""
    name_parts = [ts_tag]
    if case_tag:
        name_parts.append(case_tag)
    else:
        name_parts.append(cycle_tag)
    name_parts.append(tamb_tag)
    name_parts.append(bsfc_tag)
    if soc_tag:
        name_parts.append(soc_tag)

    run_id = "_".join([p for p in name_parts if p])
    run_dir = out_root / run_id

    # subfolders (match your reference)
    d_inputs = run_dir / "inputs"
    d_configs = run_dir / "configs"
    d_signals = run_dir / "signals"
    d_audit = run_dir / "audit"
    d_viz = run_dir / "Viz"
    for d in [d_inputs, d_configs, d_signals, d_audit, d_viz]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- snapshot inputs ----
    copied = {}
    if args.copy_inputs:
        shutil.copy2(wltc_csv, d_inputs / wltc_csv.name)
        copied["wltc_csv"] = str((d_inputs / wltc_csv.name).resolve())
        if str(args.bsfc_csv).strip():
            shutil.copy2(bsfc_csv_path, d_inputs / bsfc_csv_path.name)
            copied["bsfc_csv"] = str((d_inputs / bsfc_csv_path.name).resolve())

    # ---- save configs ----
    configs_payload = dict(
        common=_to_jsonable(common),
        veh=_to_jsonable(veh),
        batt=_to_jsonable(batt),
        init=_to_jsonable(init),
        env=_to_jsonable(env),
        batt_cfg_B=_to_jsonable(batt_cfg_B),
        sim_cfg_B=_to_jsonable(sim_cfg_B),
        engine_gate=_to_jsonable(engine_gate),
        solver_kwargs=_to_jsonable({k: v for k, v in solver_kwargs.items() if k not in ("bsfc_map", "eta_mg1_map", "eta_mg2_map", "eng_tq_max_map", "eng_drag_min_map")}),
        bsfc_tag=bsfc_tag,
        wltc_csv=str(wltc_csv),
    )
    with open(d_configs / "configs.json", "w", encoding="utf-8") as f:
        json.dump(configs_payload, f, indent=2, ensure_ascii=False)

    # ---- run Phase B ----
    wltc_df = load_wltc(str(wltc_csv))
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
        engine_gate=engine_gate,
    )
    cons2, kpis, budgets = compute_audit_outputs_B(ts, cons)

    # ---- save signals ----
    ts_name, cons_name = default_output_filenames()
    ts_path = d_signals / ts_name
    cons_path = d_signals / cons_name
    ts.to_csv(ts_path, index=False)
    cons.to_csv(cons_path, index=False)

    stamp = ts_name.removeprefix("timeseries_phaseB_").removesuffix(".csv")

    ts_derived = compute_phaseB_derived_fuel_integrals(ts, fuel_density_kg_per_L=None, copy=True)
    validate_phaseB_derived_fuel_integrals(ts_derived)
    ts_derived_path = d_signals / f"timeseries_phaseB_derived_{stamp}.csv"
    ts_derived.to_csv(ts_derived_path, index=False)

    audit_cons_path = d_signals / f"constraints_phaseB_with_audit_{stamp}.csv"
    cons2.to_csv(audit_cons_path, index=False)

    # ---- save audit ----
    with open(d_audit / f"kpis_phaseB_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)
    with open(d_audit / f"budgets_phaseB_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(budgets, f, indent=2, ensure_ascii=False)

    # ---- manifest.json ----
    manifest = dict(
        run_id=run_id,
        created_at_local=now.isoformat(timespec="seconds"),
        python=sys.version,
        platform=platform.platform(),
        git_head=_try_git_head(),
        args=vars(args),
        folders=dict(
            run_dir=str(run_dir.resolve()),
            inputs=str(d_inputs.resolve()),
            configs=str(d_configs.resolve()),
            signals=str(d_signals.resolve()),
            audit=str(d_audit.resolve()),
            Viz=str(d_viz.resolve()),
        ),
        files=dict(
            timeseries=str(ts_path.resolve()),
            timeseries_derived=str(ts_derived_path.resolve()),
            constraints=str(cons_path.resolve()),
            constraints_with_audit=str(audit_cons_path.resolve()),
        ),
        copied_inputs=copied,
    )
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n✅ Phase B run saved")
    print(f"  run_dir: {run_dir.resolve()}")
    print(f"  signals: {d_signals.resolve()}")
    print(f"  audit:   {d_audit.resolve()}")

    # ---- optional: run viz into Viz/ ----
    if args.run_viz:
        cand = [
            REPO_ROOT / "scripts" / "viz_ths_dashboards_phaseB.py",
            REPO_ROOT / "viz_ths_dashboards_phaseB.py",
        ]
        viz_py = next((p for p in cand if p.exists()), None)
        if viz_py is None:
            print("[WARN] viz_ths_dashboards_phaseB.py not found; skip viz.")
        else:
            cmd = [
                sys.executable,
                str(viz_py),
                "--timeseries",
                str(ts_path),
                "--outdir",
                str(d_viz),
                "--dpi",
                str(args.viz_dpi),
            ]
            print("\n▶ Running viz:", " ".join(cmd))
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
            print(f"[OK] Viz saved to: {d_viz.resolve()}")


if __name__ == "__main__":
    main()