# analyze_phaseB_suite.py
import json
import os
import glob
import argparse
import pandas as pd


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_gate_dict(prefix: str, d: dict):
    for k, v in d.items():
        if isinstance(v, dict) and ("ok" in v) and (v["ok"] is False):
            print(f"  - FAIL {prefix}: {k}  value={v.get('value')}  thr={v.get('thr')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_phaseB_suite", help="suite output dir")
    args = ap.parse_args()

    root = args.root
    summary_csv = os.path.join(root, "suite_summary.csv")
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        print("\n=== Phase B Suite Summary ===")
        print(df.to_string(index=False))
        overall_p = os.path.join(root, "suite_overall.json")
        if os.path.exists(overall_p):
            print(f"\nOVERALL_PASS = {load_json(overall_p).get('OVERALL_PASS')}")
    else:
        print(f"[WARN] summary CSV not found: {summary_csv}")

    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(vdir):
            continue
        gate_p = os.path.join(vdir, "gate_detail.json")
        if not os.path.exists(gate_p):
            continue

        gate = load_json(gate_p)
        if gate.get("PASS", False):
            continue

        name = os.path.basename(vdir)
        print(f"\n=== {name} : FAIL ===")

        if "A" in gate:
            _print_gate_dict("PhaseA", gate["A"])
        if "B" in gate:
            _print_gate_dict("PhaseB", gate["B"])

        # quick KPI hints
        kA_p = os.path.join(vdir, "kpis_phaseA.json")
        kB_p = os.path.join(vdir, "kpis_phaseB.json")
        if os.path.exists(kA_p):
            kA = load_json(kA_p)
            print("  [Phase A KPIs]")
            print(f"    psd_speed_resid_max_rpm={kA.get('psd_speed_resid_max_rpm')}, elec_power_resid_max_W={kA.get('elec_power_resid_max_W')}")
            print(f"    count_flag_mg1_overspeed={kA.get('count_flag_mg1_overspeed')}, count_flag_mg2_overspeed={kA.get('count_flag_mg2_overspeed')}, count_flag_batt_sat={kA.get('count_flag_batt_sat')}")
            print(f"    E_short_over_E_trac={kA.get('E_short_over_E_trac')}, shortfall_step_ratio={kA.get('shortfall_step_ratio')}")
        if os.path.exists(kB_p):
            kB = load_json(kB_p)
            print("  [Phase B KPIs]")
            print(f"    TTW_eff={kB.get('TTW_eff')}, regen_utilization={kB.get('regen_utilization')}, EV_share_time={kB.get('EV_share_time')}")
            print(f"    fuel_balance_resid_max_W={kB.get('fuel_balance_resid_max_W')}, bus_balance_resid_max_W={kB.get('bus_balance_resid_max_W')}, soc_recon_resid_max_abs_pct={kB.get('soc_recon_resid_max_abs_pct')}")


if __name__ == "__main__":
    main()
