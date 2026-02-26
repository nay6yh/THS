import pandas as pd
ts = pd.read_csv("out_phaseA_suite5/A01_cold_Tamb-10C/timeseries.csv")
m = ts["shortfall_power_W"] > 1e-6

pin = (ts["P_batt_act_W"] >= ts["lim_batt_discharge_W"] - 1e-3)
print("shortfall_ratio:", m.mean())
print("pinned_while_shortfall:", (pin[m]).mean())