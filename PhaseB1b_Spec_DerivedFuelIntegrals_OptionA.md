## Phase B.1b Spec（Derived Fuel Integrals）— Option A

## Phase B.1b 仕様（燃料積分などの派生列）— Option A（契約維持）

### 0. Status / ステータス

* JP: **APPROVED（設計確定）**：Option A を採用し、**85列 Frozen contract は変更しない**。
  EN: **APPROVED**: adopt Option A; **do not modify the frozen 85-column contract**.

---

## 1. Decision / 設計判断（Option A）

* JP: 派生列（積分・累積など）は **timeseries 本体（85列）に追加しない**。
  EN: Derived columns (integrals/cumulatives) are **NOT added to the canonical 85-column timeseries**.
* JP: 派生列は、(A1) ダッシュボード等で都度計算する、または (A2) **派生CSV/Parquet**として別出力する。
  EN: Derived columns are computed either (A1) on-the-fly in dashboards, or (A2) exported as **separate derived CSV/Parquet**.

---

## 2. Inputs / 入力（前提）

* JP: 入力は Phase B timeseries（85列）で、最低限 `dt_s`, `mdot_fuel_gps`, `P_fuel_W`, `t_s` を要求する。
  EN: Input is Phase B timeseries (85 cols), requiring at minimum `dt_s`, `mdot_fuel_gps`, `P_fuel_W`, `t_s`.
* JP: `dt_s` は末尾行で 0 を許容（WLTC終端）。積分は `Σ(x * dt)` で定義し、終端行は増分ゼロとなる。
  EN: `dt_s` may be 0 at the last row; integration is `Σ(x * dt)` so terminal row adds zero.

---

## 3. Derived columns / 派生列（列名・定義式・単位）

> JP: ここで定義する列名は「派生データの契約」です（ただし本体85列とは別契約）。
> EN: These names define the derived-data contract (separate from the canonical 85-col contract).

### 3.1 Required derived columns / 必須派生列

| Column          | Unit | Definition (JP)         | Definition (EN)         |
| --------------- | ---: | ----------------------- | ----------------------- |
| `fuel_step_g`   |    g | `mdot_fuel_gps * dt_s`  | `mdot_fuel_gps * dt_s`  |
| `fuel_cum_g`    |    g | `cumsum(fuel_step_g)`   | `cumsum(fuel_step_g)`   |
| `E_fuel_step_J` |    J | `P_fuel_W * dt_s`       | `P_fuel_W * dt_s`       |
| `E_fuel_cum_J`  |    J | `cumsum(E_fuel_step_J)` | `cumsum(E_fuel_step_J)` |

### 3.2 Optional derived columns / 任意派生列（必要時のみ）

* JP: 体積換算を扱う場合に限り、燃料密度 `fuel_density_kg_per_L` を外部設定として渡す。
  EN: Only if volumetric conversion is needed, pass `fuel_density_kg_per_L` externally.

| Column          | Unit | Definition (JP)                                       | Definition (EN)                                       |
| --------------- | ---: | ----------------------------------------------------- | ----------------------------------------------------- |
| `fuel_cum_kg`   |   kg | `fuel_cum_g / 1000`                                   | `fuel_cum_g / 1000`                                   |
| `fuel_cum_L`    |    L | `fuel_cum_kg / fuel_density_kg_per_L`                 | `fuel_cum_kg / fuel_density_kg_per_L`                 |
| `fuel_flow_Lph` |  L/h | `(mdot_fuel_gps/1000) / fuel_density_kg_per_L * 3600` | `(mdot_fuel_gps/1000) / fuel_density_kg_per_L * 3600` |

---

## 4. Output forms / 出力形態（A1 / A2）

### A1) On-the-fly derived view / ダッシュボード等で都度計算

* JP: 可視化・解析側で `ts` を受け取り、派生列を一時的に付与する（保存しない）。
  EN: Dashboard/analysis computes derived columns transiently (no file output).
* JP: メリット：出力ファイル増加なし、Frozen contract 不変。
  EN: Pros: no new files; frozen contract remains untouched.

### A2) Separate derived file / 派生ファイルとして別出力（推奨オプション）

* JP: `timeseries_phaseB_derived_{stamp}.(csv|parquet)` を追加で出力する。
  EN: Export `timeseries_phaseB_derived_{stamp}.(csv|parquet)` as an additional artifact.
* JP: `{stamp}` は既存の `timeseries_phaseB_{stamp}.csv` と同一のスタンプを用いる。
  EN: `{stamp}` matches the canonical `timeseries_phaseB_{stamp}.csv` stamp.

**Recommended derived file content / 推奨する派生ファイルの中身**

* JP: 解析の利便性を優先し、**元85列 + 派生列を末尾に追加**した DataFrame を保存する。
  EN: For convenience, save **original 85 columns + derived columns appended at the end**.
* JP: 派生ファイルの列順：
  `schema_B.PHASE_B_OUTPUT_COLUMN_ORDER + DERIVED_COLUMNS`
  EN: Derived file column order:
  `schema_B.PHASE_B_OUTPUT_COLUMN_ORDER + DERIVED_COLUMNS`

---

## 5. Determinism / 決定論性

* JP: 派生列は `ts` と定数設定のみから計算され、乱数や外部状態を持たないため **決定論的**。
  EN: Derived columns are **deterministic**, computed only from `ts` and constant configs.

---

## 6. Acceptance criteria / 受け入れ基準

### 6.1 Numerical sanity / 数値健全性

* JP: `fuel_step_g`, `fuel_cum_g`, `E_fuel_step_J`, `E_fuel_cum_J` に NaN/inf が無いこと。
  EN: No NaN/inf in `fuel_step_g`, `fuel_cum_g`, `E_fuel_step_J`, `E_fuel_cum_J`.

### 6.2 Monotonicity / 単調性

* JP: `fuel_cum_g` と `E_fuel_cum_J` は **単調非減少**であること。
  EN: `fuel_cum_g` and `E_fuel_cum_J` must be **monotonic non-decreasing**.

### 6.3 Contract separation / 契約分離

* JP: canonical timeseries（85列）の列名・列順・列数は一切変更しないこと。
  EN: Do not modify canonical 85-column timeseries in any way (names/order/count).

---

## 7. Implementation notes (design-only) / 実装メモ（設計のみ）

> JP: ここは設計メモ。実装は Phase B.1b の作業として別PRで行う。
> EN: Design notes only; implementation will be in a separate PR for Phase B.1b.

* JP: 推奨として `derived_B.py`（新規）に以下の純関数を置く：
  EN: Recommended to add pure function in new `derived_B.py`:

**Proposed API (design target)**

* `compute_phaseB_derived_fuel_integrals(ts: pd.DataFrame, *, fuel_density_kg_per_L: float | None = None) -> pd.DataFrame`

* JP: 出力保存は既存 I/O（`run_io.write_df_parquet_or_csv`）の流儀に合わせる。
  EN: Saving should reuse existing I/O helper (`run_io.write_df_parquet_or_csv`) conventions.