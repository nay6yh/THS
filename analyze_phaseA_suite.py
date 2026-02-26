# analyze_phaseA_suite.py
import json, os, glob

ROOT = "out_phaseA_suite5"

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

for vdir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(vdir):
        continue
    gate_p = os.path.join(vdir, "gate_detail.json")
    kpi_p  = os.path.join(vdir, "kpis.json")
    if not (os.path.exists(gate_p) and os.path.exists(kpi_p)):
        continue

    gate = load_json(gate_p)
    kpis = load_json(kpi_p)
    name = os.path.basename(vdir)
    if gate.get("PASS", False):
        continue

    print(f"\n=== {name} : FAIL ===")

    # 1) gate_detail で ok=False の項目を表示
    for k, v in gate.items():
        if isinstance(v, dict) and ("ok" in v) and (v["ok"] is False):
            print(f"  - FAIL gate: {k}  value={v.get('value')}  thr={v.get('thr')}")

    # 2) 参考：サチュレーション/ショートfallの代表値も表示
    print(f"  count_flag_batt_sat={kpis.get('count_flag_batt_sat')}, "
          f"count_flag_eng_sat={kpis.get('count_flag_eng_sat')}, "
          f"count_flag_mg2_sat={kpis.get('count_flag_mg2_sat')}")
    print(f"  elec_power_resid_max_W={kpis.get('elec_power_resid_max_W')}, "
          f"wheel_power_resid_max_W={kpis.get('wheel_power_resid_max_W')}")