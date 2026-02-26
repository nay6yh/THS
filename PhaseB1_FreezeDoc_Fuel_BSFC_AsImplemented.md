## Phase B.1 Design Freeze Doc（Fuel / BSFC）— As Implemented

## Phase B.1 設計凍結ドキュメント（Fuel / BSFC）— 現行実装準拠

### 0. Status / ステータス

* JP: **APPROVED / FROZEN**（本ドキュメントの記載事項は Phase B.1 の契約として凍結する）
  EN: **APPROVED / FROZEN** (this document freezes the Phase B.1 contract)

### 1. Scope / スコープ

* JP: Phase B.1 は、BSFC に基づいて **燃料流量・燃料パワー・エンジン熱損失**を一貫計算し、Phase B timeseries へ **append-only** で出力する。
  EN: Phase B.1 consistently computes **fuel mass flow, fuel power, engine thermal loss** from BSFC and appends them to Phase B timeseries (append-only).
* JP: **Phase A の列名・列順を変更しない**（Phase B は拡張列を末尾に追加するのみ）。
  EN: **Do not change Phase A column names/order**; Phase B only appends extension columns.

### 2. Out of Scope / 対象外

* JP: 過渡燃料補正、暖機詳細、始動ペナルティ、アイドル燃料モデルの精密化は本フェーズ対象外。
  EN: Transient fuel corrections, warm-up dynamics, start penalties, detailed idle-fuel modeling are out of scope for B.1.

### 3. Interfaces / インターフェース（凍結）

#### 3.1 BSFC callable signature / BSFC 関数シグネチャ

* JP: `bsfc_map(eng_rpm, eng_tq_Nm) -> g/kWh` を満たす callable を DI（Dependency Injection）で受け取る。
  EN: Receive DI callable satisfying `bsfc_map(eng_rpm, eng_tq_Nm) -> g/kWh`.

#### 3.2 Fuel accounting core API / Fuel 会計コア API

* JP: `fuel_B.calc_fuel_account(*, eng_rpm: float, eng_tq_Nm: float, bsfc_map: BsfcMap, lhv_J_per_g: float = 43e3) -> FuelAccount`
  EN: `fuel_B.calc_fuel_account(*, eng_rpm: float, eng_tq_Nm: float, bsfc_map: BsfcMap, lhv_J_per_g: float = 43e3) -> FuelAccount`

#### 3.3 Phase B entrypoint / Phase B 実行エントリポイント

* JP: `sim_grid_B.simulate_ths_grid_B(..., bsfc_map: BsfcMap, lhv_J_per_g: float = 43_000.0, solve_step_base_fn=...) -> (timeseries_df, constraints_df)`
  EN: `sim_grid_B.simulate_ths_grid_B(..., bsfc_map: BsfcMap, lhv_J_per_g: float = 43_000.0, solve_step_base_fn=...) -> (timeseries_df, constraints_df)`

---

### 4. Frozen sign conventions / 符号規約（最重要・凍結）

#### 4.1 Mechanical power and fuel-on rule / 機械出力と fuel_on 規約

* JP: `P_mech_W = ω(eng_rpm) * eng_tq_Nm` を計算し、**`P_mech_W > 0` のときのみ燃料を消費する（fuel_on）**。
  EN: Compute `P_mech_W = ω(eng_rpm) * eng_tq_Nm`; **consume fuel only when `P_mech_W > 0`**.

#### 4.2 Fuel-cut contract / Fuel-cut 契約

* JP: `P_mech_W <= 0` の場合、**`mdot_fuel_gps = 0`, `P_fuel_W = 0`, `loss_engine_W = 0`** とする。
  EN: If `P_mech_W <= 0`, enforce **`mdot_fuel_gps = 0`, `P_fuel_W = 0`, `loss_engine_W = 0`**.

---

### 5. Computation definitions / 計算定義（凍結）

* JP: `P_kW = P_mech_W / 1000`
  EN: `P_kW = P_mech_W / 1000`
* JP: `mdot_fuel_gps = bsfc_g_per_kWh * P_kW / 3600`（fuel_on のとき）
  EN: `mdot_fuel_gps = bsfc_g_per_kWh * P_kW / 3600` (when fuel_on)
* JP: `P_fuel_W = mdot_fuel_gps * lhv_J_per_g`
  EN: `P_fuel_W = mdot_fuel_gps * lhv_J_per_g`
* JP: `loss_engine_W = P_fuel_W - P_mech_W`（fuel_on のとき、**常に ≥0**）
  EN: `loss_engine_W = P_fuel_W - P_mech_W` (when fuel_on, **must be ≥0**)

---

### 6. Frozen outputs / 出力（列名・列順・列数凍結）

#### 6.1 Phase A base timeseries contract / Phase A ベース契約

* JP: Phase A timeseries の列順は `schema_B.PHASE_A_TIMESERIES_COLUMN_ORDER`（**71列**）で凍結。
  EN: Phase A timeseries order is frozen by `schema_B.PHASE_A_TIMESERIES_COLUMN_ORDER` (**71 columns**).

#### 6.2 Phase B extension columns / Phase B 拡張列（append-only）

* JP: Phase B 拡張列は `schema_B.PHASE_B_EXT_COLUMNS`（**14列**）で凍結し、Phase A の末尾に append する。
  EN: Phase B extension columns are frozen by `schema_B.PHASE_B_EXT_COLUMNS` (**14 columns**) and appended after Phase A.

**PHASE_B_EXT_COLUMNS (Frozen, 14 cols)**

1. `P_batt_chem_W`
2. `loss_batt_W`
3. `I_batt_A`
4. `mdot_fuel_gps`
5. `P_fuel_W`
6. `loss_engine_W`
7. `loss_mg1_W`
8. `loss_mg2_W`
9. `loss_inv1_W`
10. `loss_inv2_W`
11. `loss_inv_W`
12. `P_batt_chg_from_regen_W`
13. `P_batt_chg_from_engine_W`
14. `sim_phase`

* JP: Phase B 出力列順は `schema_B.PHASE_B_OUTPUT_COLUMN_ORDER = 71 + 14 = 85列` を凍結。
  EN: Phase B output order is frozen by `schema_B.PHASE_B_OUTPUT_COLUMN_ORDER = 71 + 14 = 85 columns`.
* JP: `sim_phase` は **`"B"` 固定**（`schema_B.SIM_PHASE_VALUE`）
  EN: `sim_phase` is fixed to **`"B"`** (`schema_B.SIM_PHASE_VALUE`)

---

### 7. Invariants / 不変条件（凍結）

* JP: `mdot_fuel_gps >= 0`, `P_fuel_W >= 0`, `loss_engine_W >= 0` を満たすこと。
  EN: Must satisfy `mdot_fuel_gps >= 0`, `P_fuel_W >= 0`, `loss_engine_W >= 0`.
* JP: `P_fuel_W < P_mech_W` を検出した場合は **入力不正（BSFC/LHV）として例外**。
  EN: If `P_fuel_W < P_mech_W`, raise exception as **invalid BSFC/LHV input**.

---

### 8. Call graph / 呼び出し関係（凍結）

* JP: Phase B は「Import Firewall」を守り、Phase A solver は **外部アダプタ経由で注入**する。
  EN: Phase B respects the “Import Firewall”; Phase A solver is **injected via external adapter**.

**Execution flow (frozen)**

* `run_phaseB.py`
  → `sim_grid_B.simulate_ths_grid_B(..., solve_step_base_fn=solve_step_A_adapter, bsfc_map=...)`
  → (per step) base solve → `step_B.enrich_with_phase_b_extensions(base=..., bsfc_map=..., ...)`
  → `flatten_B.flatten_phase_b_extensions(solB)`
  → append to row dict following `schema_B.PHASE_B_OUTPUT_COLUMN_ORDER`
  → outputs: `(timeseries_df, constraints_df)`
  → `audit_B.compute_audit_outputs_B(ts, cons)` for KPIs/budgets

---

### 9. Acceptance / 受け入れ（監査KPI）

* JP: `audit_B.compute_audit_outputs_B()` は Fuel の整合を residual として監査する：
  `fuel_balance_resid_max_W = max(|P_fuel_W - (P_eng_mech_W + loss_engine_W)|)`
  EN: `audit_B.compute_audit_outputs_B()` audits fuel balance via residual:
  `fuel_balance_resid_max_W = max(|P_fuel_W - (P_eng_mech_W + loss_engine_W)|)`
* JP: B.1 の KPI には以下が含まれる（代表）：
  `TTW_eff`, `P_fuel_avg_W`, `E_fuel_MJ`, `fuel_balance_resid_max_W`, `bus_balance_resid_max_W` 等
  EN: B.1 KPIs include (examples):
  `TTW_eff`, `P_fuel_avg_W`, `E_fuel_MJ`, `fuel_balance_resid_max_W`, `bus_balance_resid_max_W`, etc.

---

### 10. Change control / 変更管理

* JP: 本 Freeze Doc の項目（符号・API・列名/列順/列数・fuel-cut規約）を変更する場合、**Phase B schema のバージョン更新**と、下流（可視化・監査・テスト）の同時レビューを必須とする。
  EN: Any changes to this frozen contract require **schema version bump** and coordinated downstream reviews (viz/audit/tests).
