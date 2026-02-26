from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ths_zero.configs import (
    CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig,
    load_or_default, dump_config,
)
from ths_zero.wltc import load_wltc
# from ths_zero.sim import simulate_ths   # 旧
from ths_zero.sim_grid_A import simulate_ths_grid_A  # 新

from ths_zero.audit import compute_audit_outputs
from ths_zero.run_io import (
    ensure_dir, write_json, write_df_parquet_or_csv,
    copy_input, sha256_file, sha256_json, build_manifest, default_schema
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wltc", type=str, required=True, help="WLTC CSV (Phase, Total_time_s, Speed_kmh)")
    ap.add_argument("--out_root", type=str, default="runs", help="runs output root dir")

    ap.add_argument("--common", type=str, default=None, help="common config json")
    ap.add_argument("--vehicle", type=str, default=None, help="vehicle config json")
    ap.add_argument("--battery", type=str, default=None, help="battery config json")
    ap.add_argument("--initial", type=str, default=None, help="initial state json")
    ap.add_argument("--env", type=str, default=None, help="environment config json")
    ap.add_argument("--tag", type=str, default="ths_wltc", help="run tag string")
    args = ap.parse_args()

    # load configs (or defaults)
    common = load_or_default(args.common, CommonConfig())
    vehicle = load_or_default(args.vehicle, VehicleConfig())
    battery = load_or_default(args.battery, BatteryConfig())
    initial = load_or_default(args.initial, InitialState())
    env = load_or_default(args.env, EnvironmentConfig())

    # run folder
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{ts}_{args.tag}"
    run_dir = Path(args.out_root) / run_id

    # create structure
    inputs_dir = run_dir / "inputs"
    cfg_dir = run_dir / "configs"
    sig_dir = run_dir / "signals"
    audit_dir = run_dir / "audit"
    ensure_dir(inputs_dir); ensure_dir(cfg_dir); ensure_dir(sig_dir); ensure_dir(audit_dir)

    # copy inputs
    wltc_src = Path(args.wltc)
    wltc_dst = inputs_dir / "wltc.csv"
    copy_input(wltc_src, wltc_dst)

    # dump configs used
    dump_config(str(cfg_dir / "common.json"), common)
    dump_config(str(cfg_dir / "vehicle.json"), vehicle)
    dump_config(str(cfg_dir / "battery.json"), battery)
    dump_config(str(cfg_dir / "initial.json"), initial)
    dump_config(str(cfg_dir / "environment.json"), env)

    # simulate
    wltc = load_wltc(str(wltc_dst))
    # ★ここ：旧シミュレーション呼び出しを置換
    timeseries, constraints = simulate_ths_grid_A(wltc, common, vehicle, battery, initial, env)
    constraints2, kpis, budgets = compute_audit_outputs(timeseries, constraints)


    # write outputs
    p_ts = write_df_parquet_or_csv(timeseries, sig_dir / "timeseries")
    p_cons = write_df_parquet_or_csv(constraints2, audit_dir / "constraints")
    write_json(sig_dir / "timeseries_schema.json", default_schema())
    write_json(audit_dir / "kpis.json", kpis)
    write_json(audit_dir / "budgets.json", budgets)

    # manifest
    inputs_meta = {
        "wltc.csv": {"path": str(wltc_dst), "sha256": sha256_file(wltc_dst)},
    }
    cfg_meta = {
        "common": {"sha256": sha256_json(common.__dict__)},
        "vehicle": {"sha256": sha256_json(vehicle.__dict__)},
        "battery": {"sha256": sha256_json(battery.__dict__)},
        "initial": {"sha256": sha256_json(initial.__dict__)},
        "environment": {"sha256": sha256_json(env.__dict__)},
    }
    outputs = {
        "signals/timeseries": str(p_ts),
        "signals/timeseries_schema": str(sig_dir / "timeseries_schema.json"),
        "audit/constraints": str(p_cons),
        "audit/kpis": str(audit_dir / "kpis.json"),
        "audit/budgets": str(audit_dir / "budgets.json"),
    }
    manifest = build_manifest(run_id=run_id, inputs=inputs_meta, configs=cfg_meta, outputs=outputs)
    write_json(run_dir / "manifest.json", manifest)

    print("[OK] run saved:", run_dir.resolve())
    print(" - timeseries:", p_ts.name)
    print(" - constraints:", p_cons.name)
    print(" - kpis:", (audit_dir / "kpis.json").name)


if __name__ == "__main__":
    main()