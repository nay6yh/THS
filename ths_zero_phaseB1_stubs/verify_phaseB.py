# verify_phaseB.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ths_zero.schema_B import PHASE_B_OUTPUT_COLUMN_ORDER, validate_column_order

EPS = 1e-9

def main():
    csv_path = sorted(Path("out_phaseB").glob("timeseries_phaseB_*.csv"))[-1]
    df = pd.read_csv(csv_path)

    # strict column order check (85 cols)
    validate_column_order(df, strict=True)
    print("✅ Column order validation PASSED (strict=True)")

    # basic energies (J -> MJ)
    dt = df["dt_s"].to_numpy(float)

    P_fuel = df["P_fuel_W"].to_numpy(float)
    P_wheel_deliv = df["P_wheel_deliv_W_dbg"].to_numpy(float)

    E_fuel_MJ = (P_fuel * dt).sum() / 1e6
    E_wheel_pos_MJ = (P_wheel_deliv.clip(min=0) * dt).sum() / 1e6
    E_wheel_neg_MJ = ((-P_wheel_deliv).clip(min=0) * dt).sum() / 1e6

    # regen split
    P_regen_to_batt = df["P_batt_chg_from_regen_W"].to_numpy(float)
    E_regen_to_batt_MJ = (P_regen_to_batt * dt).sum() / 1e6

    TTW_eff = E_wheel_pos_MJ / max(E_fuel_MJ, EPS)
    regen_util = E_regen_to_batt_MJ / max(E_wheel_neg_MJ, EPS)

    print("\n📊 Quick KPIs")
    print(f"  E_fuel_MJ           = {E_fuel_MJ:.3f}")
    print(f"  E_wheel_pos_MJ      = {E_wheel_pos_MJ:.3f}")
    print(f"  E_wheel_neg_MJ      = {E_wheel_neg_MJ:.3f}")
    print(f"  TTW_eff             = {TTW_eff:.4f}")
    print(f"  E_regen_to_batt_MJ  = {E_regen_to_batt_MJ:.3f}")
    print(f"  regen_utilization   = {regen_util:.4f}  (should be 0..1)")

    print("\n🔎 sim_phase unique:", df["sim_phase"].unique())

if __name__ == "__main__":
    main()