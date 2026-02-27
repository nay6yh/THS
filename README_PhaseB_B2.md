## THS Zero Simulator — Phase B.2 Spec

### BSFC Map Upgrade + Engine Supervisor (Start Penalty / Min ON-OFF / Fuel Weight Tuning)

---

## 0. Purpose / 目的

### JP

Phase B.2 は、Phase B.1（Fuel/Loss 会計の凍結）をベースに、以下を達成する：

1. **BSFC を “定数” から “2Dマップ（rpm×tq）”へ**置換し、運転点依存の燃料会計を実現する
2. エンジンの ON/OFF 挙動を **実車らしく**する（チャタリング抑制）
3. その上で **燃費最適化のチューニング**を可能にする（重み調整）

### EN

Phase B.2 builds on Phase B.1 (frozen fuel/loss accounting) and delivers:

1. Replace constant BSFC with a **2D BSFC map (rpm×tq)**
2. Add a Phase-B-only **Engine Supervisor** for realistic ON/OFF behavior
3. Enable stable fuel-economy tuning via **weight adjustment**

---

## 1. Scope / スコープ

### 1.1 In-scope (B.2)

* **B.2b — BSFC 2D map (rpm×tq)**

  * Grid CSV → interpolation callable `bsfc_map(eng_rpm, eng_tq_Nm)->g/kWh`
  * OOB policy: clamp（範囲外は端値）＋ invalid cell fill
* **B.2 Supervisor — Engine ON/OFF realism layer (Phase B only)**

  * (1) Engine start penalty
  * (2) Minimum ON / Minimum OFF time (hysteresis)
  * (3) Fuel weight tuning (run_phaseB.py で weights を変えるだけ。Phase Aコードは改造しない)

### 1.2 Out-of-scope (B.2)

* 実車の完全な排ガス/触媒/暖機モデルの同定（B.3+）
* エンジン再始動の詳細ダイナミクス（MG1/クラッチ等の過渡モデル）
* Phase A solver 内部の大改造（Phase B-only 方針）

---

## 2. Design Principles / 設計原則（重要）

### JP

* **Phase A を汚さない**：Phase B で使うが、Phase Aの実装変更は最小（可能ならゼロ）。
* **Import firewall を守る**：Phase B モジュールは Phase A を import しない。
* **後処理で eng_rpm/eng_tq を書き換えない**：PSD/MG/Bus整合が壊れるため禁止。
* 代わりに、Phase B では **複数の“整合した候補 StepResult”を作り、選択する**。

### EN

* Keep Phase A untouched (ideally zero code changes).
* Respect import firewall (Phase B must not import Phase A).
* Do not patch `eng_rpm/eng_tq` post-solve (breaks physical consistency).
* Instead, Phase B generates multiple consistent candidates and selects one.

---

## 3. B.2b — BSFC Map (rpm×tq) / 仕様

### 3.1 Input format (grid CSV)

* Header: `eng_rpm, <tq0>, <tq1>, ...`
* Cells: `bsfc_g_per_kWh`
* Invalid/unreachable cells: use `>= invalid_threshold` (default 900) → treated as invalid and filled

Example:

```
eng_rpm,0,10,20,...,120
1000,999,320,280,...,245
...
```

### 3.2 Map behavior

* Interpolation: bilinear
* OOB policy: clamp to axis limits
* NaN fill: row/col forward-fill & backward-fill, fallback to `fill_value`

### 3.3 Deliverable

* `make_bsfc_map_from_grid_csv(path)->BsfcMap2D` returning callable:

  * `bsfc_map(eng_rpm, eng_tq_Nm) -> g/kWh`

---

## 4. Engine Supervisor (Phase B-only) / 仕様

### 4.1 Why / なぜ必要か

B.2で観測された “不自然さ” の主因：

* 1秒ステップでの **ON/OFFチャタリング**
* 離散グリッド探索由来の **運転点ジャンプ**
* スイッチングコスト（始動燃料/NVH）が無いことによる **bang-bang運転**

### 4.2 Priority order (real-vehicle-like)

1. **Hard protection** (SOC limits / hardware safety)
2. **Drivability / demand satisfaction**
3. **Emissions / warm-up** (deferred, but considered later)
4. **SOC regulation (charge sustaining)**
5. **Fuel economy optimization**

### 4.3 State (mutable, carried across steps)

* `fuel_on_prev: bool`
* `on_timer_s: float`
* `off_timer_s: float`
* `count_eng_start: int`
* `count_eng_stop: int`
* `start_fuel_total_g: float`

### 4.4 Parameters (frozen initial values for B.2)

Initial recommended defaults (can be tuned later):

* `start_fuel_g = 0.30` g/start
* `min_on_s = 8.0` s
* `min_off_s = 2.0` s
* Overrides:

  * allow start if `P_wheel_req_W > 30kW`
  * allow start if `soc < soc_target - soc_band`
  * allow stop if `soc > soc_target + soc_band`
* Note: **min_on is stronger**, min_off is weaker (aligns with real-world intuition)

### 4.5 Candidate generation strategy (Phase B-only)

Engine Supervisor must select from physically consistent candidates.
We generate up to 3 candidates per step by calling the injected solver with different *bias* weights:

* Candidate FREE: baseline solver_kwargs
* Candidate OFF-PREF: increase fuel penalty (EV bias)
* Candidate ON-PREF: reduce fuel penalty + strengthen SOC-charge tracking (engine bias)

Important:

* Do **not** compare candidates using solver’s own `J_total` because weights differ.
* Use a **common gate score** (below).

### 4.6 Gate constraints (hysteresis + overrides)

* If `fuel_on_prev==1` and `on_timer_s < min_on_s`: **stop forbidden**
* If `fuel_on_prev==0` and `off_timer_s < min_off_s`: **start forbidden**
* Override forbidden-start if:

  * high demand (`P_wheel_req_W > override_start_power_W`) OR
  * SOC trending low (`soc < soc_target - soc_band`)
* SOC hard bounds (if available) override all hysteresis rules.

### 4.7 Common gate score (minimize)

For fair comparison across candidates:

* `fuel_g_step = mdot_fuel_gps * dt`
* `start_penalty_g = start_fuel_g` only if OFF→ON transition
* optional: `shortfall_penalty = k_short * shortfall_power_W * dt`

Minimal score used in B.2:

```
score = fuel_g_step + start_penalty_g + k_short * shortfall_power_W * dt
```

### 4.8 State update (after selecting final candidate)

* If OFF→ON: increment `count_eng_start`, add `start_fuel_g` to `start_fuel_total_g`
* If ON→OFF: increment `count_eng_stop`
* Update timers:

  * ON: `on_timer_s += dt`, `off_timer_s = 0`
  * OFF: `off_timer_s += dt`, `on_timer_s = 0`

---

## 5. Fuel weight tuning (B.2 “(3)”) / 燃料重み微調整

### JP

燃料項の重み変更は **Phase Aコードを改造せず**、`run_phaseB.py` 側で `weights.w_fuel` を調整して行う。
Engine Supervisor導入後の方が、挙動が安定し、重み調整の効果が素直に現れる。

### EN

Tune fuel term weight via `run_phaseB.py` (solver_kwargs weights), without modifying Phase A code.
After the supervisor, tuning becomes stable and predictable.

---

## 6. Acceptance Criteria / 受け入れ基準

### 6.1 Behavior realism

* Engine start/stop chattering decreases:

  * `count_eng_start` reduces significantly vs baseline
* Minimum dwell time is respected:

  * no OFF within `min_on_s`
  * no ON within `min_off_s` (unless override triggers)

### 6.2 Accounting invariants

* Frozen Phase B.1 residual checks remain valid:

  * `fuel_balance_resid_max_W` stays near numeric epsilon
  * `bus_balance_resid_max_W` not degraded
  * `regen_utilization ∈ [0,1]`

### 6.3 KPI improvements (expected trend)

* `fuel_g_per_km` improves or remains stable after supervisor introduction
* `bsfc_implied_mean_g_per_kWh` moves toward plausible range
* `eta_eng_mean` improves or remains plausible

---

## 7. Deliverables / 成果物

* Code:

  * `engine_maps.py`: BSFC grid map loader + `BsfcMap2D`
  * `engine_gate_B.py`: Engine Supervisor (Phase B-only)
  * `sim_grid_B.py`: insert gate between base solve and enrichment
* Outputs:

  * Existing outputs unchanged (canonical 85 columns preserved)
  * New audit KPIs (append-only):

    * `count_eng_start`, `count_eng_stop`, `start_fuel_total_g`
* Plots:

  * BSFC heatmap + operating points overlay
  * Density diff plots (B vs C) for diagnosis

---

## 8. Implementation Plan / 実装手順（B.2）

1. **B.2b**: BSFC map callable from CSV (done)
2. **Supervisor v0**:

   * start penalty only
3. Add **min_on/min_off**
4. Validate: chattering reduced + accounting residuals preserved
5. **Fuel weight tuning** (run_phaseB.py)
6. Compare: fuel_g_per_km, starts/stops, density plots