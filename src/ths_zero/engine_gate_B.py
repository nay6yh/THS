from __future__ import annotations

"""
Phase B Engine Supervisor (firewall compliant).

Goal:
- Add realistic engine ON/OFF behavior without modifying Phase A code:
  (1) relight penalty (combustion OFF->ON)
  (2) minimum combustion ON / OFF time (hysteresis)
- MUST NOT import step_A.py / sim_grid_A.py.

Key clarification (based on your measurements):
- "spin"   = engine is rotating (eng_rpm > rpm_spin_thr)
- "combust"= fuel is being injected / combustion ON (fuel_cut == 0 or mdot_fuel_gps > 0)

In your current behavior:
- spin_start is small, true_start is 0, but combust_start is large.
=> We must suppress combust toggling (fuel_cut chatter), not "spin starts".

Design constraints:
- Do NOT patch eng_rpm/eng_tq post-solve (breaks PSD/MG/DC-bus consistency).
- Instead: evaluate multiple physically-consistent StepResult candidates and select one.

Candidate generation:
- FREE: baseline solver_kwargs
- OFF-PREF: increase fuel weight to bias combustion OFF
- ON-PREF:  decrease fuel weight and increase charge tracking to bias combustion ON

Selection:
- Enforce combustion hysteresis rules + conservative overrides
- Use common gate score (grams equivalent) to compare across candidates

Return value:
- (best_candidate, relight_event)
  relight_event is used by sim_grid_B to add "event fuel" if desired.
"""

from dataclasses import dataclass, replace, is_dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .fuel_B import calc_fuel_account


# =========================
# Params
# =========================
@dataclass(frozen=True)
class EngineGateParams:
    # ---- State definitions ----
    # Spin: engine physically rotating
    rpm_spin_thr: float = 100.0

    # (1) Penalties [g/event]
    # - relight_fuel_g: Combust OFF->ON (fuel_cut 1->0), even if already spinning
    # - start_spin_fuel_g: Spin OFF->ON (eng_rpm crosses threshold) (often rare in current model)
    relight_fuel_g: float = 0.15
    start_spin_fuel_g: float = 0.50

    # (2) Hysteresis on COMBUST (fuel_cut), not on spin
    min_combust_on_s: float = 12.0
    min_combust_off_s: float = 5.0

    # Overrides (realistic priority)
    override_start_power_W: float = 50_000.0  # allow relight if wheel demand is high

    # Conservative SOC override to allow relight:
    # allow if soc < soc_target - soc_low_override_mult*soc_band for >= soc_low_dwell_s
    soc_low_override_mult: float = 2.5
    soc_low_dwell_s: float = 5.0

    # Candidate bias scales
    w_fuel_scale_off: float = 5.0      # OFF-PREF: increase fuel term
    w_fuel_scale_on: float = 0.5       # ON-PREF: decrease fuel term
    w_charge_scale_on: float = 5.0     # ON-PREF: increase charge tracking term

    # shortfall penalty coefficient (dimensionless multiplier on eq-fuel[g])
    k_short: float = 10.0

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "EngineGateParams":
        """
        Backward compatible loader.

        If old keys are provided:
          - start_fuel_g      -> relight_fuel_g
          - min_on_s          -> min_combust_on_s
          - min_off_s         -> min_combust_off_s
        """
        base = EngineGateParams()
        if not d:
            return base

        relight = d.get("relight_fuel_g", d.get("start_fuel_g", base.relight_fuel_g))
        min_on = d.get("min_combust_on_s", d.get("min_on_s", base.min_combust_on_s))
        min_off = d.get("min_combust_off_s", d.get("min_off_s", base.min_combust_off_s))

        return EngineGateParams(
            rpm_spin_thr=float(d.get("rpm_spin_thr", base.rpm_spin_thr)),
            relight_fuel_g=float(relight),
            start_spin_fuel_g=float(d.get("start_spin_fuel_g", base.start_spin_fuel_g)),
            min_combust_on_s=float(min_on),
            min_combust_off_s=float(min_off),
            override_start_power_W=float(d.get("override_start_power_W", base.override_start_power_W)),
            soc_low_override_mult=float(d.get("soc_low_override_mult", base.soc_low_override_mult)),
            soc_low_dwell_s=float(d.get("soc_low_dwell_s", base.soc_low_dwell_s)),
            w_fuel_scale_off=float(d.get("w_fuel_scale_off", base.w_fuel_scale_off)),
            w_fuel_scale_on=float(d.get("w_fuel_scale_on", base.w_fuel_scale_on)),
            w_charge_scale_on=float(d.get("w_charge_scale_on", base.w_charge_scale_on)),
            k_short=float(d.get("k_short", base.k_short)),
        )


# =========================
# Helpers
# =========================
def _combust_on(base: Any) -> int:
    """
    Combustion ON if fuel is injected.
    Primary: fuel_cut flag (frozen semantics) fuel_cut=1 means no fuel (combust OFF).
    """
    if hasattr(base, "fuel_cut"):
        return int(int(getattr(base, "fuel_cut")) == 0)

    # Fallback: positive engine mech work often implies combustion on
    eng_rpm = float(getattr(base, "eng_rpm", 0.0) or 0.0)
    eng_tq = float(getattr(base, "eng_tq_Nm", 0.0) or 0.0)
    return int(eng_rpm > 0.0 and eng_tq > 0.0)


def _spin_on(base: Any, rpm_thr: float) -> int:
    eng_rpm = float(getattr(base, "eng_rpm", 0.0) or 0.0)
    return int(eng_rpm > float(rpm_thr))


def _scale_weights(weights: Any, *, w_fuel_mult: float = 1.0, w_charge_mult: float = 1.0) -> Any:
    """
    Scale StepWeights without importing its type.
    Works if weights is a dataclass (StepWeights is a dataclass in Phase A).
    """
    if weights is None:
        return None

    if is_dataclass(weights):
        kw: Dict[str, Any] = {}
        if hasattr(weights, "w_fuel"):
            kw["w_fuel"] = float(getattr(weights, "w_fuel")) * float(w_fuel_mult)
        if hasattr(weights, "w_charge_track"):
            kw["w_charge_track"] = float(getattr(weights, "w_charge_track")) * float(w_charge_mult)
        return replace(weights, **kw) if kw else weights

    # Best-effort: if not dataclass, return as-is
    return weights


# =========================
# Main supervisor
# =========================
def apply_engine_supervisor_B(
    *,
    x: Any,
    base_free: Any,
    solve_step_base_fn: Callable[..., Any],
    solver_kwargs: Dict[str, Any],
    bsfc_map: Any,
    lhv_J_per_g: float,
    gate_state: Dict[str, Any],
    params: EngineGateParams,
) -> Tuple[Any, bool]:
    """
    Phase-B-only engine supervisor.

    Returns:
      (best_candidate, relight_event)

    relight_event = combust OFF->ON (fuel_cut 1->0).
    This can be used by sim_grid_B to add an event fuel penalty to P_fuel/loss_engine.
    """
    dt = float(getattr(x, "dt_s", 1.0) or 1.0)
    P_req = float(getattr(x, "P_wheel_req_W", 0.0) or 0.0)
    soc = float(getattr(x, "soc", float("nan")))

    soc_target = float(solver_kwargs.get("soc_target", 0.55))
    soc_band = float(solver_kwargs.get("soc_band", 0.05))

    traction_step = (P_req >= 0.0)

    # --- Track COMBUST hysteresis explicitly (not spin) ---
    prev_combust = int(gate_state.get("combust_on_prev", gate_state.get("fuel_on_prev", 0)))
    combust_on_timer = float(gate_state.get("combust_on_timer_s", gate_state.get("on_timer_s", 0.0)))
    combust_off_timer = float(gate_state.get("combust_off_timer_s", gate_state.get("off_timer_s", 1e9)))

    # SOC low dwell (for conservative override)
    soc_low_timer = float(gate_state.get("soc_low_timer_s", 0.0))

    # --- allow combust transitions (hysteresis + conservative overrides) ---
    allow_relight = True   # combust OFF->ON
    allow_cut = True       # combust ON->OFF

    # braking: generally do NOT allow relight unless SOC is deeply low
    if not traction_step:
        allow_relight = bool(soc == soc and soc < (soc_target - params.soc_low_override_mult * soc_band))
        allow_cut = True

    # min combust ON: keep combust ON until min satisfied (strict; no soft-high exception)
    if prev_combust == 1 and combust_on_timer < (params.min_combust_on_s - 1e-9):
        allow_cut = False

    # min combust OFF: keep combust OFF until min satisfied unless override triggers
    if prev_combust == 0 and combust_off_timer < (params.min_combust_off_s - 1e-9):
        allow_relight = False

        # conservative SOC override: require deeper low SOC and dwell
        soc_low = (soc == soc) and (soc < (soc_target - params.soc_low_override_mult * soc_band))
        if soc_low:
            soc_low_timer = soc_low_timer + dt
        else:
            soc_low_timer = 0.0

        if (P_req > params.override_start_power_W) or (soc_low_timer >= params.soc_low_dwell_s):
            allow_relight = True
    else:
        # still update soc_low_timer for logging / continuity
        soc_low = (soc == soc) and (soc < (soc_target - params.soc_low_override_mult * soc_band))
        soc_low_timer = soc_low_timer + dt if soc_low else 0.0

    # Candidate set
    cands: list[tuple[str, Any]] = [("FREE", base_free)]
    weights0 = solver_kwargs.get("weights", None)

    def _solve_with_scaled_weights(name: str, w_fuel_mult: float, w_charge_mult: float) -> Any:
        kw = dict(solver_kwargs)
        kw["weights"] = _scale_weights(weights0, w_fuel_mult=w_fuel_mult, w_charge_mult=w_charge_mult)
        return solve_step_base_fn(x, **kw)

    free_combust = _combust_on(base_free)

    # IMPORTANT:
    # If FREE proposes a COMBUST transition, ALWAYS generate the opposite-biased candidate for comparison.
    # This is required for relight penalty to actually reduce combust toggling.
    if prev_combust == 0 and free_combust == 1:
        cands.append(("OFFPREF", _solve_with_scaled_weights("OFFPREF", params.w_fuel_scale_off, 1.0)))
    if prev_combust == 1 and free_combust == 0:
        cands.append(("ONPREF", _solve_with_scaled_weights("ONPREF", params.w_fuel_scale_on, params.w_charge_scale_on)))

    # --- validity check against combust hysteresis (with overrides already applied) ---
    def _is_valid(cand: Any) -> bool:
        cur_combust = _combust_on(cand)
        if prev_combust == 0 and cur_combust == 1 and not allow_relight:
            return False
        if prev_combust == 1 and cur_combust == 0 and not allow_cut:
            return False
        return True

    # --- common gate score in "grams equivalent" ---
    def _score_g(cand: Any) -> float:
        cur_combust = _combust_on(cand)

        # Fuel grams for this step (from operating point + bsfc_map)
        fuel_g = 0.0
        if cur_combust == 1 and dt > 0.0:
            eng_rpm = float(getattr(cand, "eng_rpm", 0.0) or 0.0)
            eng_tq = float(getattr(cand, "eng_tq_Nm", 0.0) or 0.0)
            fa = calc_fuel_account(
                eng_rpm=eng_rpm,
                eng_tq_Nm=eng_tq,
                bsfc_map=bsfc_map,
                lhv_J_per_g=lhv_J_per_g,
            )
            fuel_g = float(fa.mdot_fuel_gps) * dt

        # Relight penalty: combust OFF->ON
        relight_pen = float(params.relight_fuel_g) if (prev_combust == 0 and cur_combust == 1 and dt > 0.0) else 0.0

        # Shortfall penalty: convert missing energy [J] to eq fuel grams and scale
        shortfall_W = float(getattr(cand, "shortfall_power_W", 0.0) or 0.0)
        shortfall_W = max(shortfall_W, 0.0)
        short_eq_g = (shortfall_W * dt) / max(float(lhv_J_per_g), 1e-9)

        return float(fuel_g + relight_pen + params.k_short * short_eq_g)

    # choose best among valid; fallback to best among all if none valid
    valid = [(n, c) for (n, c) in cands if _is_valid(c)]
    pool = valid if valid else cands

    best_name, best = min(pool, key=lambda nc: _score_g(nc[1]))

    # Events and states
    cur_combust = _combust_on(best)
    cur_spin = _spin_on(best, params.rpm_spin_thr)
    prev_spin = int(gate_state.get("spin_on_prev", cur_spin))

    relight_event = (prev_combust == 0 and cur_combust == 1)
    spin_start_event = (prev_spin == 0 and cur_spin == 1)

    # update gate state
    gate_state["soc_low_timer_s"] = float(soc_low_timer)
    gate_state["spin_on_prev"] = int(cur_spin)
    gate_state["combust_on_prev"] = int(cur_combust)

    # timers on combust
    if cur_combust == 1:
        gate_state["combust_on_timer_s"] = float((combust_on_timer + dt) if prev_combust == 1 else dt)
        gate_state["combust_off_timer_s"] = 0.0
    else:
        gate_state["combust_off_timer_s"] = float((combust_off_timer + dt) if prev_combust == 0 else dt)
        gate_state["combust_on_timer_s"] = 0.0

    # Backward compatible keys (optional)
    gate_state["fuel_on_prev"] = int(cur_combust)
    gate_state["on_timer_s"] = float(gate_state["combust_on_timer_s"])
    gate_state["off_timer_s"] = float(gate_state["combust_off_timer_s"])

    # Counters
    if relight_event:
        gate_state["count_combust_start"] = int(gate_state.get("count_combust_start", 0)) + 1
        gate_state["relight_fuel_total_g"] = float(gate_state.get("relight_fuel_total_g", 0.0)) + float(params.relight_fuel_g)

        # keep old keys too (historical)
        gate_state["count_eng_start"] = int(gate_state.get("count_eng_start", 0)) + 1
        gate_state["start_fuel_total_g"] = float(gate_state.get("start_fuel_total_g", 0.0)) + float(params.relight_fuel_g)

    if prev_combust == 1 and cur_combust == 0:
        gate_state["count_combust_stop"] = int(gate_state.get("count_combust_stop", 0)) + 1
        gate_state["count_eng_stop"] = int(gate_state.get("count_eng_stop", 0)) + 1

    if spin_start_event:
        gate_state["count_spin_start"] = int(gate_state.get("count_spin_start", 0)) + 1
        gate_state["start_spin_fuel_total_g"] = float(gate_state.get("start_spin_fuel_total_g", 0.0)) + float(params.start_spin_fuel_g)

    gate_state["last_choice"] = str(best_name)

    # Return relight_event as start_event for sim_grid_B hook (adds event fuel penalty)
    return best, bool(relight_event)