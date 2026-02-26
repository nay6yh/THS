# THS Zero Simulator — Phase A Spec (Physically Consistent + Grid Solver A)

## 0. Scope

Phase A is the **baseline physically-consistent THS simulator** that:

* satisfies PSD kinematic constraint (speed constraint)
* respects component limits (rpm/torque/battery power)
* updates SOC by **power integration with clipping**
* produces a **timeseries.csv** that supports dashboards ①〜④（＋⑤の比較入力にもなる）

**Not in Phase A (explicitly out of scope):**

* fuel / BSFC / TTW efficiency
* explicit engine/MG/battery loss accounting closeable to strict energy balance
* thermal dynamics of battery / cabin

---

## 1. Inputs

### 1.1 Drive cycle

* WLTC (time series)

  * `t_s`, `veh_spd_mps` (or km/h converted), `veh_acc_mps2`, `phase`
* Sampling: fixed `dt_s` or derived from `t_s`

### 1.2 Configs

**Common (THS base)**

* PSD constants: `alpha`, `beta`
* sign conventions (battery +discharge / -charge, etc.)
* objective weights (A solver)

**Vehicle**

* limits: `eng_rpm_min/max`, `mg1_rpm_max`, `mg2_rpm_max`
* torque limits: `eng_tq_max_map(rpm)` (or constant max), `mg1_tq_max_Nm`, `mg2_tq_max_Nm`
* driveline efficiency: `driveline_eff`
* vehicle dynamics params: mass, CdA, Cr, wheel radius, etc. (used to compute wheel demand)

**Battery**

* usable energy constants:

  * `E_usable_Wh`
  * `soc_min`, `soc_max` → `Emin_Wh`, `Emax_Wh`
* per-step power limits (may depend on temperature in a simple way):

  * `P_dis_max_W`, `P_chg_max_W`

**Initial**

* `soc0` (0..1) or `E_batt0_Wh`
* `Tbatt0_C` (constant in Phase A unless explicitly varied)

**Environment**

* `Tamb_C` (constant per run in Phase A)
* `rho_air_kgpm3` derived from Tamb (optional simplification)
* auxiliary loads:

  * `P_aux_W` (can include HVAC proxy)

---

## 2. Outputs (primary artifacts)

Phase A produces one main output consumed by dashboards:

### 2.1 `timeseries.csv`

Per-step signals including:

* time/labels: `t_s`, `dt_s`, `phase`, `mode`
* vehicle dynamics: `veh_spd_mps`, `veh_acc_mps2`
* wheel/ring:

  * `wheel_omega_radps`, `ring_omega_radps`, `ring_rpm`
  * `P_wheel_req_W`, `T_ring_req_Nm`
  * delivered: `P_wheel_deliv_W_dbg`, `T_ring_deliv_Nm`
* powertrain states:

  * `eng_rpm`, `eng_tq_Nm`
  * `mg1_rpm`, `mg1_tq_Nm`
  * `mg2_rpm`, `mg2_tq_Nm`
* electrical powers (sign-consistent):

  * `P_mg1_elec_W`, `P_mg2_elec_W`
  * `P_batt_req_W`, `P_batt_act_W`
* braking & slack:

  * `P_brake_fric_W` (>=0)
  * `shortfall_tq_Nm` / `shortfall_power_W`
  * `excess_tq_Nm` (over-delivery) and `J_over` (penalty)
* SOC:

  * `E_batt_Wh`, `soc_pct`
  * batt limits: `lim_batt_discharge_W`, `lim_batt_charge_W`
* flags (for audit/atlas):

  * `flag_batt_sat`, `flag_eng_sat`, `flag_mg1_sat`, `flag_mg2_sat`
  * `flag_mg1_overspeed`, `flag_mg2_overspeed`
* audit residuals (debug visibility):

  * `resid_psd_speed_rpm`
  * `resid_elec_power_W`
  * `resid_ring_torque_Nm_dbg`
  * `resid_wheel_power_W_dbg`

### 2.2 `kpis.json` / `audit outputs`

* RMS/max residuals
* SOC start/end/delta
* counts of flags

(Phase Aでは “燃料・効率” は含めない)

---

## 3. Physical model (Phase A core)

### 3.1 PSD kinematics (speed constraint)

Assume PSD constraint holds by construction:

[
\omega_{eng} = \alpha , \omega_{mg1} + \beta , \omega_{ring}
]

Implementation uses rpm form:

* `mg1_rpm = (eng_rpm - beta*ring_rpm)/alpha`
* This makes `resid_psd_speed_rpm ≈ 0` by design (numerical only)

### 3.2 PSD quasi-static torque relation (simplified THS A-model)

Use the simplified static torque relation:

* `mg1_tq = -alpha * eng_tq`
* ring-side torque contribution from PSD: `T_psd_to_ring = beta * eng_tq`

This is an intentional Phase A simplification (no detailed planetary torque split dynamics).

### 3.3 Ring demand and MG2 torque

Given vehicle dynamics, compute wheel power/torque demand, map to ring:

* `P_wheel_req_W` from resistive forces (aero/roll/inertial) + speed
* `T_ring_req_Nm` derived via ring speed (and driveline mapping)

Then:

* `mg2_tq_cmd = T_ring_req - T_psd_to_ring`
* `mg2_tq_act = clip(mg2_tq_cmd, ±mg2_tq_max)`

### 3.4 Electrical power conversion (sign-consistent)

`mech_to_elec(P_mech, eta)` with consistent loss sign:

* motoring: `P_elec = P_mech/eta`
* generating: `P_elec = P_mech*eta`

Battery request:
[
P_{batt,req} = P_{mg1,elec} + P_{mg2,elec} + P_{aux}
]
Battery actual:
[
P_{batt,act} = clip(P_{batt,req}, -P_{chg,max}, +P_{dis,max})
]

### 3.5 Braking (regen + friction) — explicit handling

If `P_wheel_req_W < 0`:

* enforce charge acceptance:

  * reduce regen (MG2 torque) when `P_batt_req` exceeds `-P_chg_max`
* leftover braking is converted to friction:

  * `P_brake_fric_W >= 0`

### 3.6 SOC update (no inference, clip-consistent)

SOC is updated only by integrating `P_batt_act`:

[
E_{k+1} = clip(E_k - P_{batt,act},dt/3600,; E_{min},E_{max})
]
[
SOC_{k+1} = E_{k+1}/E_{usable}
]

**Critical Phase A rule:**
`E_usable_Wh`, `Emin_Wh`, `Emax_Wh` are passed in and **never inferred** from data.

---

## 4. Control / Solver (A案)

### 4.1 Mode set

* If braking: `Regen`, `FrictionBrake`
* If traction: `EV`, `HybridDrive`
* If SOC low: add `Charge`

### 4.2 A solver (grid search with local search)

For each step:

* EV / braking: evaluate a single candidate (fast)
* HybridDrive/Charge:

  * local search around previous `(eng_rpm, eng_tq)` first
  * if no feasible candidate found → fallback to global coarse grid

Feasibility checks:

* mg1 speed limit, mg2 speed limit
* mg1 torque limit
* engine torque bounds (0..Tmax) for powered modes
* fuel-cut mode torque bounds (drag_min..0)

### 4.3 Objective components (Phase A)

Typical components:

* `J_short` : traction shortfall penalty
* `J_over`  : over-delivery torque penalty (excess ring torque)
* `J_soc`   : SOC tracking (banded around target)
* `J_fric`  : friction dissipation penalty
* `J_spin`  : high-speed spin penalty (eng/mg1 speed)
* `J_smooth`: smoothness vs previous step
* `J_charge`: charge tracking (when in Charge mode)

**Phase A note:** fuel term is either absent or treated as a proxy (no BSFC).

---

## 5. Residuals and their meaning (Phase A)

### 5.1 `resid_psd_speed_rpm`

* should be ~0 because kinematics are constructed to satisfy PSD equation

### 5.2 `resid_elec_power_W`

[
resid_elec = P_{batt,act} - (P_{mg1,elec}+P_{mg2,elec}+P_{aux})
]

* should be ~0 unless a bug exists (Phase A aims for exact)

### 5.3 `resid_ring_torque_Nm_dbg`

[
resid_{ring} = T_{ring,req} - ( \beta T_{eng} + T_{mg2} ) - shortfall
]

* should be close to 0 (shortfall is explicit slack)

### 5.4 `resid_wheel_power_W_dbg`

[
resid_{wheel} = P_{wheel,req} - P_{wheel,deliv} + P_{brake,fric}
]

* should be close to 0 (friction is explicit term)

---

## 6. Phase A “Done” criteria (frozen)

Phase A is considered correct when:

* PSD speed residuals are ~0
* wheel power residuals are small (RMS small, max bounded)
* ring torque residuals small (except when explicit slack is used)
* SOC reconstruction residual is ~0 (clip-consistent re-integration)
* dashboards ①〜④ read correctly and anomalies are traceable

---

## 7. Known limitations (intended in Phase A)

* No fuel consumption / BSFC / TTW
* No explicit loss accounting → Sankey is explanatory
* Temperature mostly constant; Tbatt dynamics not modeled
* Engine/motor maps are simplified; operating points may be discretized by solver grid
* “Charge” mode is heuristic (proxy) until Phase B introduces fuel/loss objective