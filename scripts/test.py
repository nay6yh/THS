import pandas as pd

df = pd.read_csv(r"out_phaseB\timeseries_phaseB_20260224_224846.csv")

print("rows:", len(df), "cols:", len(df.columns))
print("t_s tail:", df["t_s"].tail(3).tolist())
print("dt_s tail:", df["dt_s"].tail(3).tolist())

# Local balances
fuel_resid = (df["P_fuel_W"] - (df["P_eng_mech_W"] + df["loss_engine_W"])).abs()
bus_resid  = (df["P_batt_act_W"] - (df["P_mg1_elec_W"] + df["P_mg2_elec_W"] + df["P_aux_W"])).abs()

print("fuel_resid max [W]:", fuel_resid.max())
print("bus_resid  max [W]:", bus_resid.max())