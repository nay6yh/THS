import pandas as pd
import numpy as np

ts = pd.read_csv("out_phaseB/timeseries_phaseB_derived_20260226_001215.csv")  # 自分のファイル名に

rpm_thr = 100.0
spin = ts["eng_rpm"].to_numpy(float) > rpm_thr

# combustionは mdot_fuel_gps が最も確実（無ければ fuel_cut==0）
if "mdot_fuel_gps" in ts.columns:
    combust = ts["mdot_fuel_gps"].to_numpy(float) > 0.0
else:
    combust = (ts["fuel_cut"].to_numpy(int) == 0)

spin_start = int(np.sum((~spin[:-1]) & (spin[1:])))
combust_start = int(np.sum((~combust[:-1]) & (combust[1:])))
true_start = int(np.sum((~spin[:-1]) & (spin[1:]) & combust[1:]))

print("spin_start:", spin_start)
print("combust_start:", combust_start)
print("true_start:", true_start)

import pandas as pd, numpy as np

# ts = pd.read_csv("timeseries_phaseB_derived_XXXX.csv")  # 対象ラン
combust = (ts["mdot_fuel_gps"].to_numpy(float) > 0.0)   # or fuel_cut==0
dt = ts["dt_s"].to_numpy(float)

# combust OFF 連続区間長（秒）
off = ~combust
lens = []
cur = 0.0
for k in range(len(off)):
    if off[k]:
        cur += dt[k]
    else:
        if cur > 0:
            lens.append(cur)
        cur = 0.0
if cur > 0:
    lens.append(cur)

print("num_off_segments =", len(lens))
print("min/median/max off_s =", np.min(lens), np.median(lens), np.max(lens))
print("count off<5s =", np.sum(np.array(lens) < 5.0))