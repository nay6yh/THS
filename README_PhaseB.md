# THS Zero Simulator — Phase B Spec (Fuel + Loss + Control Objective Upgrade)

## 0. Scope

Phase B upgrades Phase A’s “physically consistent but explanatory” THS simulation into a **fuel/loss-aware** and **control-realistic** simulator.

### Phase A (current)

* PSD kinematics/torque constraints satisfied
* Battery SOC updated by integrated `P_batt_act`
* A-solver (grid search) produces plausible mode/points
* Sankey/energy plots are **explanatory** (not closeable balance)

### Phase B (this spec)

* Add **fuel model (BSFC)** and **loss accounting** (engine/MG/inverter/battery)
* Upgrade objective to reflect **real control incentives**
* Make energy flow closer to a **quasi-balance** (reduce “Other/unknown”)

---

## 1. Inputs (Phase B additions)

### 1.1 Engine

* `BSFC_map(eng_rpm, eng_tq_Nm) -> g/kWh`
* `eng_tq_max_map(eng_rpm) -> Nm`
* Optional (recommended):

  * `eng_drag_min_map(eng_rpm) -> Nm` (fuel-cut drag torque)

### 1.2 MG / Inverter

* Already used:

  * `eta_mg1_map(rpm, tq)`, `eta_mg2_map(rpm, tq)`
* Optional split:

  * `eta_inv1_map`, `eta_inv2_map` (if inverter losses separated)

### 1.3 Battery electrical model

Minimum viable (choose one):

* **Model B0 (constant efficiency)**

  * `eta_dis`, `eta_chg`
* **Model B1 (Voc/Rint)** (recommended)

  * `Voc(SOC, Tbatt)`
  * `Rint(SOC, Tbatt)`
  * `I = P / V` (iterative or approximate)
  * `P_loss_batt = I^2 * Rint`

Limits (temperature dependent):

* `P_dis_max_W(Tbatt)`
* `P_chg_max_W(Tbatt)`

### 1.4 Thermal / HVAC

* `Tamb_C` is an input (already in timeseries)
* HVAC load model (MVP):

  * `P_hvac_W = f(Tamb_C, target_cabin_C, vehicle_class)`
  * Can be simple piecewise or linear for now
* Optional Tbatt dynamics (recommended later):

  * `Tbatt[k+1] = Tbatt[k] + dt/tau * (Tamb - Tbatt) + k_loss*(loss_batt+loss_inv)`

---

## 2. Outputs: timeseries columns to add

### 2.1 Fuel / Engine energy (must-have)

* `mdot_fuel_gps` [g/s]
* `P_fuel_W` [W]

  * `P_fuel_W = mdot_fuel_gps * LHV_Jpg` (LHV ~ 43e3 J/g as a default constant)
* `loss_engine_W` [W]

  * `loss_engine_W = max(P_fuel_W - P_eng_mech_W, 0)`

### 2.2 Electrical losses (must-have)

* `loss_mg1_W`, `loss_mg2_W` from mech↔elec conversion
* If inverter is separated:

  * `loss_inv1_W`, `loss_inv2_W`
* Battery loss:

  * `loss_batt_W` (from B0/B1 model)

### 2.3 Split charging source (strongly recommended)

To avoid regen metrics > 1:

* `P_batt_chg_from_regen_W`
* `P_batt_chg_from_engine_W` (proxy)
  And their integrated energies:
* `E_regen_to_batt_MJ`
* `E_engine_to_batt_MJ`

---

## 3. Accounting & Balance KPIs (frozen checks)

### 3.1 Local balance checks (per-step diagnostics)

* Engine:

  * `P_fuel_W ≈ P_eng_mech_W + loss_engine_W`
* Electrical bus (conceptual):

  * `P_batt_act_W ≈ (P_mg1_elec_W + P_mg2_elec_W + P_aux_W) + loss_batt_W + (optional inv losses)`
  * (Exact depends on how losses are placed; the goal is “explainable residuals”)

### 3.2 Global energy checks (over WLTC)

* `E_fuel_MJ`, `E_eng_mech_MJ`, `E_engine_loss_MJ`
* `E_mg_loss_MJ`, `E_batt_loss_MJ`, `E_inv_loss_MJ` (if applicable)
* “Other” energy in Sankey should approach **~0** (or be attributable)

### 3.3 Regen metrics (must be 0..1)

* `regen_utilization = E_regen_to_batt_MJ / max(E_wheel_neg_MJ, eps)`  ∈ [0,1]
* `friction_share = E_fric_MJ / max(E_wheel_neg_MJ, eps)`

### 3.4 Efficiency metrics (new)

* `TTW_eff = E_wheel_pos_MJ / max(E_fuel_MJ, eps)` (Tank-to-Wheel)
* `EV_share_time` = fraction of time with `fuel_cut==1`
* `EV_share_energy` = fraction of wheel traction energy supplied by battery discharge (proxy)

### 3.5 Constraint metrics (extended)

* `count_batt_sat_charge`, `count_batt_sat_discharge` (directional)
* `count_friction_due_to_batt_limit` (batt_sat==1 and P_brake_fric>0)

---

## 4. Control / Optimization Objective (Phase B solver target)

Priority order (frozen philosophy):

1. **Wheel demand tracking**

   * traction: minimize `shortfall_tq` (hard)
   * braking: meet braking demand using regen + friction (hard)
2. **Fuel minimization**

   * minimize `mdot_fuel_gps` or `P_fuel_W`
3. **SOC regulation**

   * maintain SOC within band; penalize SOC drift
4. **Loss minimization**

   * penalize `loss_mg* + loss_batt (+ loss_inv)`
5. **Constraint avoidance + smoothness**

   * penalize saturations (eng_sat, batt_sat)
   * penalize large step changes (rpm/tq jerk)

Notes:

* This objective should make operating points “naturally” realistic (no arbitrary artifacts).
* Keep Phase A A-solver as fallback until Phase B solver is stable.

---

## 5. Visualization updates enabled by Phase B

### 5.1 Sankey (upgrade path)

* Add a true source: `Fuel_energy`

  * `Fuel -> Engine_mech + Engine_loss`
* Add loss sinks:

  * `MG/INV loss`, `Battery loss`, `Engine loss`
* Keep “explanatory” note until residuals are consistently low.

### 5.2 KPI tables

Add:

* `E_fuel_MJ`, `TTW_eff`
* `E_engine_loss_MJ`, `E_mg_loss_MJ`, `E_batt_loss_MJ`
* `E_regen_to_batt_MJ`, `E_engine_to_batt_MJ`

---

## 6. Implementation order (recommended)

1. **Fuel (BSFC) only**

   * add `mdot_fuel_gps`, `P_fuel_W`, `E_fuel_MJ`
2. **MG conversion loss logging**

   * record `loss_mg1_W`, `loss_mg2_W`
3. **Battery loss model**

   * start with B0 (eta), then upgrade to B1 (Voc/Rint)
4. **Split charging origin**

   * `regen_to_batt` vs `engine_to_batt`
5. **Phase B solver (optimization)**

   * introduce continuous objective and compare vs A-solver

---

## 7. Done criteria (Phase B freeze)

* Sankey “Other” becomes **explainable loss** (not mysterious gap)
* `regen_utilization` always in **[0,1]**
* `TTW_eff` available and stable across Tamb comparisons
* Constraint events can be explained in terms of **fuel/loss/limits**
