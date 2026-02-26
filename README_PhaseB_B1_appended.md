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

---

## 8. Phase B.1 Plan

### 8.1 Goal

- **JP:** B.1 は「燃料＋損失の会計を“閉じる方向に寄せる”最小セット」を導入し、**Sankey の “Other/unknown” を減らし、TTW/regen KPI を安定化**させる。
- **EN:** In B.1, we introduce the minimal “fuel + loss accounting” set to **shrink Sankey’s “Other/unknown”** and stabilize **TTW / regen KPIs**.

### 8.2 Done criteria (B.1 freeze conditions)

B.1 は **No-NaN / 範囲制約（Range）**を最優先で固定し、"極端値" はまず **WARN** として検知します。

#### HARD (must pass)

- **No-NaN**: `mdot_fuel_gps`, `P_fuel_W`, `loss_engine_W`, `loss_mg1_W`, `loss_mg2_W`, `loss_batt_W`, `TTW_eff`, `regen_utilization` が全タイムステップで NaN にならない。
- **Range**: `regen_utilization ∈ [0, 1]` を満たす。
- **Range**: `TTW_eff ∈ (0, 1)` を満たす（0/1 ちょうどは epsilon で回避）。

#### WARN (non-blocking, investigate)

- **TTW_eff sanity band**: 例として `TTW_eff ∈ [0.05, 0.60]` を外れた場合は WARN として検知（B.2 で HARD 昇格候補）。
- **Loss non-negative**: `loss_engine_W`, `loss_mg1_W`, `loss_mg2_W`, `loss_batt_W ≥ -eps` を WARN として検知（丸め等で微小負値が出る場合はクリップ or 原因調査）。

### 8.3 Implementation scope

#### 8.3.1 In-scope for B.1 (must-have)

1. **Fuel (BSFC) 導入**: `mdot_fuel_gps`, `P_fuel_W`, `loss_engine_W` を追加。
2. **MG 損失ログ**: 既存の効率マップから `loss_mg1_W`, `loss_mg2_W` を算出（機械↔電気変換差分）。
3. **Battery 損失モデル B0**: 定数効率 `eta_dis/eta_chg` に基づく `loss_batt_W` を追加。
4. **充電起源の分離**: `P_batt_chg_from_regen_W` と `P_batt_chg_from_engine_W` を分離し、`regen_utilization > 1` を防止。

#### 8.3.2 Constants (explicit)

- **Fuel LHV**: `LHV_Jpg` を単一の定数として定義（例: `LHV_Jpg = 43_000.0` [J/g]）。
  - 設定場所は `constants.py` / `config.yaml` のいずれかに固定し、KPI に `LHV_Jpg_used` を出力して追跡可能にする。

#### 8.3.3 Out-of-scope for B.1 (deferred to B.2+)

- Battery の **Voc/Rint (B1)** モデル
- Tbatt ダイナミクス
- 最適化ソルバ（連続目的関数）

### 8.4 Test plan & gates

#### 8.4.1 Test variants

- B.0 の全バリアント（`B00`〜`B09`, `F01_expected_fail_*`）を **変更せずに継続**し、non-regression を保証する。
- `F01` は引き続き expected-fail として扱い、`F01_expected_fail_check` が PASS なら suite PASS とする。

#### 8.4.2 New gates for B.1

- **No-NaN gates (HARD)**: fuel/loss/KPI 列が全タイムステップで定義される。
- **Range gates (HARD)**: `regen_utilization ∈ [0,1]`, `TTW_eff ∈ (0,1)`。
- **Balance checks (Loose) (WARN→段階強化)**:
  - 局所: `P_fuel_W ≈ P_eng_mech_W + loss_engine_W` の符号と桁が破綻しない（まずは粗い許容誤差）。
  - グローバル: `E_fuel_MJ ≈ E_eng_mech_MJ + E_engine_loss_MJ` の相対誤差が許容内（まずは大枠）。
- **Loss non-negative (WARN)**: `loss_* ≥ -eps` を検知して異常を早期発見。
- **Clamp rule (HARD + WARN count)**: `regen_utilization` は [0,1] に必ずクリップし、クリップ発生回数を WARN としてレポート（デバッグ容易化）。

### 8.5 Implementation order

1. **Fuel (BSFC) only**: まず `B00` を回し、HARD No-NaN を満たすことを確認。
2. **MG Losses**: `loss_mg1/loss_mg2` を追加し、WARN loss non-negative を確認。
3. **Battery Loss (B0)**: charge/discharge で `eta_chg/eta_dis` が正しく適用されることを確認。
4. **Split charging origin**: `regen_utilization ∈ [0,1]`（HARD）と、クリップ回数（WARN）を確認。
5. **Full suite run**: 全バリアント実行し、`B09_determinism_compare` の決定性が維持されていることを最終確認。

### 8.6 Deliverables (outputs & visualization)

- **KPI tables**: `E_fuel_MJ`, `TTW_eff`, `E_engine_loss_MJ`, `E_mg_loss_MJ`, `E_batt_loss_MJ`, `E_regen_to_batt_MJ`, `E_engine_to_batt_MJ`, `LHV_Jpg_used` を追加。
- **Sankey**: `Fuel → Engine_mech + Engine_loss` を追加し、`Engine Loss`, `MG Loss`, `Battery Loss` を損失 sink として追加することで "Other" を縮小・説明可能化する。
