from __future__ import annotations

"""Phase B simulation runner (WLTC loop) — firewall compliant.

FROZEN OUTPUT CONTRACT:
- Preserve Phase A *true* timeseries columns (71) in the exact order defined in schema_B.
- Append Phase B extension columns (14) in the exact order defined in schema_B.

Import firewall (IMMUTABLE):
- MUST NOT import step_A.py, sim_grid_A.py, or audit.py at runtime.

Design:
- Orchestration-only.
- Uses dependency injection for Phase A step solver and (optionally) the front-end builders.

Recommended integration pattern:
- In project-level code (outside Phase B package), import solve_step_A and inject it as
  solve_step_base_fn (or via an adapter).
- Inject build_step_input_fn and build_base_row_fn if you want an exact replica of
  sim_grid_A's inputs/rows.

This module provides a best-effort default derived-quantity calculator and a default
71-column row builder to help catch ordering issues early.
"""

import inspect
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .battery_B import BatteryConfigB
    from .losses_B import SimConfigB
    from .step_B import BsfcMap
    # These are allowed imports (read-only) by the firewall spec.
    from .configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig


def default_output_filenames(timestamp: str | None = None) -> tuple[str, str]:
    """Return frozen filename pattern for Phase B outputs."""
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return (
        f"timeseries_phaseB_{ts}.csv",
        f"constraints_phaseB_{ts}.csv",
    )


# Builder signatures (recommended)
BuildStepInputFn = Callable[..., Any]
BuildBaseRowFn = Callable[..., Dict[str, Any]]


def simulate_ths_grid_B(
    wltc: "pd.DataFrame",
    *,
    # Optional Phase A configs (used by default derived/row builders)
    common: Optional["CommonConfig"] = None,
    veh: Optional["VehicleConfig"] = None,
    batt: Optional["BatteryConfig"] = None,
    init: Optional["InitialState"] = None,
    env: Optional["EnvironmentConfig"] = None,

    # Phase B configs/maps
    batt_cfg: "BatteryConfigB",
    sim_cfg: "SimConfigB",
    bsfc_map: "BsfcMap",
    lhv_J_per_g: float = 43_000.0,

    # Dependency injection (firewall compliant)
    solve_step_base_fn: Callable[..., Any] | None = None,  # REQUIRED
    build_step_input_fn: BuildStepInputFn | None = None,    # optional
    build_base_row_fn: BuildBaseRowFn | None = None,        # optional

    # Pass-through kwargs to base solver
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Run Phase B simulation over WLTC (firewall compliant).

    Args:
        wltc:
            WLTC time series as a pandas DataFrame. sim_grid_A convention expects:
              - t_s, dt_s, veh_spd_mps, veh_acc_mps2, phase
            Default builders will try to fall back if names differ.

        common/veh/batt/init/env:
            Optional Phase A config objects. If provided, default derived calculations
            will match sim_grid_A closely.

        batt_cfg:
            Phase B battery model config.
        sim_cfg:
            Phase B inverter-loss config.
        bsfc_map:
            Callable mapping (eng_rpm, eng_tq_Nm) -> BSFC [g/kWh].
        lhv_J_per_g:
            Fuel LHV [J/g], default 43e3.

        solve_step_base_fn:
            REQUIRED. Inject a Phase A step solver (or adapter) from outside this module.
            Signature should accept a step input object + any solver kwargs and return
            a StepResult-like object (duck-typed).

        build_step_input_fn:
            Optional. If provided, it will be called with a filtered set of kwargs from:
              idx, wltc_row, derived, state, prev_t, common, veh, batt, env
            You may implement either the new signature (with derived) or an older one.

        build_base_row_fn:
            Optional. If provided, it will be called with a filtered set of kwargs from:
              wltc_row, derived, base, state, common, veh, batt, env
            You may implement either the new signature (with derived) or an older one.

        solver_kwargs:
            Optional dict of kwargs forwarded into solve_step_base_fn.

    Returns:
        (timeseries_df, constraints_df)

    Raises:
        NotImplementedError: if solve_step_base_fn is None.
        ValueError: if frozen column ordering validation fails.
    """
    if solve_step_base_fn is None:
        raise NotImplementedError(
            "simulate_ths_grid_B requires a Phase A solver function. "
            "Pass solve_step_base_fn=<adapter to solve_step_A> from outside the Phase B package."
        )

    import pandas as pd  # heavy import inside function by design

    from .schema_B import (
        PHASE_A_TIMESERIES_COLUMN_ORDER,
        PHASE_B_OUTPUT_COLUMN_ORDER,
        validate_column_order,
    )
    from .step_B import StepInputsB, enrich_with_phase_b_extensions
    from .flatten_B import flatten_phase_b_extensions

    solver_kwargs = dict(solver_kwargs or {})

    # Default builders
    build_step_input_fn = build_step_input_fn or _default_build_step_input
    build_base_row_fn = build_base_row_fn or _default_build_base_row_71col

    # state init (minimal)
    soc0 = float(getattr(init, 'soc0', 0.55)) if init is not None else 0.55
    E_usable_Wh = float(getattr(batt, 'E_usable_Wh', float('nan'))) if batt is not None else float('nan')
    E_batt_Wh = soc0 * E_usable_Wh if E_usable_Wh == E_usable_Wh else float('nan')

    state: Dict[str, Any] = {
        'soc': soc0,
        'E_batt_Wh': E_batt_Wh,
        'prev_eng_rpm': None,
        'prev_eng_tq_Nm': None,
    }

    rows: list[Dict[str, Any]] = []

    prev_t: float | None = None

    # Use records to avoid pandas per-row overhead
    for idx, wltc_row in enumerate(wltc.to_dict(orient='records')):
        derived = _compute_derived_quantities(
            wltc_row=wltc_row,
            state=state,
            common=common,
            veh=veh,
            batt=batt,
            env=env,
            prev_t=prev_t,
        )

        # Build step input (duck-typed)
        x = _call_with_filtered_kwargs(
            build_step_input_fn,
            idx=idx,
            wltc_row=wltc_row,
            derived=derived,
            state=state,
            prev_t=prev_t,
            common=common,
            veh=veh,
            batt=batt,
            env=env,
        )
        if x is None:
            raise ValueError('build_step_input_fn returned None')

        # Encourage (but do not require) StepInputsB
        if isinstance(x, StepInputsB):
            pass

        # Solve Phase A (injected)
        base = solve_step_base_fn(x, **solver_kwargs)

        # Enrich with Phase B
        sol_b = enrich_with_phase_b_extensions(
            base=base,
            bsfc_map=bsfc_map,
            batt_cfg=batt_cfg,
            sim_cfg=sim_cfg,
            lhv_J_per_g=lhv_J_per_g,
        )

        # Build base row dict (71 cols)
        out = _call_with_filtered_kwargs(
            build_base_row_fn,
            wltc_row=wltc_row,
            derived=derived,
            base=base,
            state=state,
            common=common,
            veh=veh,
            batt=batt,
            env=env,
        )
        if out is None:
            raise ValueError('build_base_row_fn returned None')

        # Ensure all Phase A keys exist
        for k in PHASE_A_TIMESERIES_COLUMN_ORDER:
            if k not in out:
                out[k] = float('nan')

        # Append Phase B extension columns (14)
        out.update(flatten_phase_b_extensions(sol_b))

        # ------------------------------------------------------------------
        # Phase B corrections (v10):
        #  1) Charging split robustness: avoid counting non-braking charging as "regen".
        #  2) SOC/state update uses chemical power (P_batt_chem_W), not terminal power.
        # ------------------------------------------------------------------

        # (1) Recompute split with braking context if available (wheel power/mode)
        try:
            from .accounting_B import split_charging_origin

            pw = out.get('P_wheel_deliv_W_dbg', None)
            # normalize NaN -> None
            if isinstance(pw, float) and pw != pw:
                pw = None

            P_regen, P_eng = split_charging_origin(
                P_batt_term_W=float(out.get('P_batt_act_W', float('nan'))),
                P_mg1_bus_W=float(out.get('P_mg1_elec_W', float('nan'))),
                P_mg2_bus_W=float(out.get('P_mg2_elec_W', float('nan'))),
                P_wheel_deliv_W_dbg=pw,
                mode=out.get('mode', None),
            )
            out['P_batt_chg_from_regen_W'] = float(P_regen)
            out['P_batt_chg_from_engine_W'] = float(P_eng)
        except Exception:
            # keep previously computed split if anything is missing
            pass

        # (2) SOC/state update using chemical power (forward Euler with clipping)
        chem_updated = False
        try:
            dt_s = float(out.get('dt_s', derived.get('dt_s', 1.0)))
            P_chem_W = float(out.get('P_batt_chem_W', float('nan')))

            # battery bounds (prefer full-row values; fall back to derived)
            E_usable_Wh = float(out.get('batt_E_usable_Wh', derived.get('E_usable_Wh', float('nan'))))
            Emin_Wh = float(out.get('batt_Emin_Wh', derived.get('Emin_Wh', float('nan'))))
            Emax_Wh = float(out.get('batt_Emax_Wh', derived.get('Emax_Wh', float('nan'))))

            E_curr_Wh = float(state.get('E_batt_Wh', float('nan')))

            if (E_curr_Wh == E_curr_Wh) and (P_chem_W == P_chem_W) and (E_usable_Wh == E_usable_Wh) and dt_s >= 0.0:
                dE_Wh = (P_chem_W * dt_s) / 3600.0
                E_next_Wh = E_curr_Wh - dE_Wh
                # clip if bounds are available
                if Emin_Wh == Emin_Wh:
                    E_next_Wh = max(E_next_Wh, Emin_Wh)
                if Emax_Wh == Emax_Wh:
                    E_next_Wh = min(E_next_Wh, Emax_Wh)

                soc_next = E_next_Wh / max(E_usable_Wh, 1e-9)
                soc_next = max(0.0, min(1.0, soc_next))

                # overwrite output + state (chem-consistent)
                out['E_batt_Wh'] = float(E_next_Wh)
                out['soc_pct'] = float(soc_next * 100.0)

                state['E_batt_Wh'] = float(E_next_Wh)
                state['soc'] = float(soc_next)
                chem_updated = True
        except Exception:
            # fall back to Phase A state update logic below
            pass

        rows.append(out)

        # Update state fallback (Phase A) if chem update did not run
        if not chem_updated:
            if hasattr(base, 'soc_next'):
                state['soc'] = float(getattr(base, 'soc_next'))
            if hasattr(base, 'E_batt_next_Wh'):
                state['E_batt_Wh'] = float(getattr(base, 'E_batt_next_Wh'))

        # prev engine for smoothing
        if hasattr(base, 'eng_rpm'):
            state['prev_eng_rpm'] = float(getattr(base, 'eng_rpm'))
        if hasattr(base, 'eng_tq_Nm'):
            state['prev_eng_tq_Nm'] = float(getattr(base, 'eng_tq_Nm'))
        # time
        t_s = wltc_row.get('t_s', wltc_row.get('time_s'))
        prev_t = float(t_s) if t_s is not None else prev_t

    ts = pd.DataFrame(rows, columns=PHASE_B_OUTPUT_COLUMN_ORDER)
    validate_column_order(ts, strict=True)

    # Phase B.0: constraints dataframe left empty (audit_B will own checks later)
    cons = pd.DataFrame()

    return ts, cons


# -----------------------------
# Helpers
# -----------------------------

def _call_with_filtered_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call fn with kwargs filtered to the parameters it accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    params = sig.parameters
    accepted = {
        k: v for k, v in kwargs.items()
        if (k in params) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    }
    return fn(**accepted)


def _default_build_step_input(
    *,
    idx: int,
    wltc_row: Dict[str, Any],
    derived: Dict[str, Any],
    state: Dict[str, Any],
    prev_t: float | None,
    **_ignored: Any,
) -> Any:
    """Default StepInputsB builder.

    This is a conservative builder intended for testing and early plumbing.
    For exact sim_grid_A behavior, inject a builder that mirrors sim_grid_A.
    """
    from .step_B import StepInputsB

    t_s = float(wltc_row.get('t_s', idx))
    if 'dt_s' in wltc_row and wltc_row['dt_s'] is not None:
        dt_s = float(wltc_row['dt_s'])
    elif prev_t is not None:
        dt_s = max(float(t_s - prev_t), 0.0)
    else:
        dt_s = 1.0

    # Use derived if available
    ring_rpm = float(derived.get('ring_rpm', wltc_row.get('ring_rpm', 0.0)))
    ring_omega = float(derived.get('ring_omega_radps', wltc_row.get('ring_omega_radps', 0.0)))
    T_ring_req = float(derived.get('T_ring_req_Nm', wltc_row.get('T_ring_req_Nm', 0.0)))
    P_wheel_req = float(derived.get('P_wheel_req_W', wltc_row.get('P_wheel_req_W', 0.0)))

    return StepInputsB(
        t_s=t_s,
        dt_s=dt_s,
        phase=str(wltc_row.get('phase', '')),

        ring_rpm=ring_rpm,
        ring_omega_radps=ring_omega,
        T_ring_req_Nm=T_ring_req,
        P_wheel_req_W=P_wheel_req,

        soc=float(state.get('soc', 0.55)),
        E_batt_Wh=float(state.get('E_batt_Wh', float('nan'))),

        E_usable_Wh=float(derived.get('E_usable_Wh', float('nan'))),
        Emin_Wh=float(derived.get('Emin_Wh', float('nan'))),
        Emax_Wh=float(derived.get('Emax_Wh', float('nan'))),

        P_aux_W=float(derived.get('P_aux_W', 0.0)),
        Tamb_C=float(derived.get('Tamb_C', float('nan'))),
        Tbatt_C=float(derived.get('Tbatt_C', float('nan'))),

        P_dis_max_W=float(derived.get('P_dis_max_W', float('nan'))),
        P_chg_max_W=float(derived.get('P_chg_max_W', float('nan'))),

        mg1_rpm_max=float(derived.get('mg1_rpm_max', float('nan'))),
        mg2_rpm_max=float(derived.get('mg2_rpm_max', float('nan'))),
        mg1_tq_max_Nm=float(derived.get('mg1_tq_max_Nm', float('nan'))),
        mg2_tq_max_Nm=float(derived.get('mg2_tq_max_Nm', float('nan'))),
        eng_rpm_min=float(derived.get('eng_rpm_min', 0.0)),
        eng_rpm_max=float(derived.get('eng_rpm_max', float('nan'))),

        alpha=float(derived.get('alpha', float('nan'))),
        beta=float(derived.get('beta', float('nan'))),

        prev_eng_rpm=state.get('prev_eng_rpm'),
        prev_eng_tq_Nm=state.get('prev_eng_tq_Nm'),
    )


def _compute_derived_quantities(
    *,
    wltc_row: Dict[str, Any],
    state: Dict[str, Any],
    common: Any,
    veh: Any,
    batt: Any,
    env: Any,
    prev_t: float | None,
) -> Dict[str, Any]:
    """Compute derived quantities (vehicle demand, PSD kinematics, env loads).

    If configs are provided, this mirrors sim_grid_A computations as closely as possible.
    If configs are missing, returns a partial dict with best-effort fields.
    """
    import numpy as np
    from .environment import air_density_kgpm3, hvac_power_W, battery_temp_C, batt_power_limits_W

    # WLTC signals (support both naming styles)
    v_mps = float(wltc_row.get('veh_spd_mps', wltc_row.get('v_mps', 0.0)))
    a_mps2 = float(wltc_row.get('veh_acc_mps2', wltc_row.get('a_mps2', 0.0)))

    # dt
    if 'dt_s' in wltc_row and wltc_row['dt_s'] is not None:
        dt_s = float(wltc_row['dt_s'])
    else:
        t_s = float(wltc_row.get('t_s', 0.0))
        dt_s = max(t_s - float(prev_t), 0.0) if prev_t is not None else 1.0

    # Environment
    Tamb_C = float(getattr(env, 'Tamb_C', wltc_row.get('Tamb_C', float('nan')))) if env is not None else float(wltc_row.get('Tamb_C', float('nan')))
    p_amb = float(getattr(env, 'p_amb_Pa', 101325.0)) if env is not None else 101325.0

    rho = air_density_kgpm3(Tamb_C, p_amb, common) if Tamb_C == Tamb_C else float('nan')
    P_hvac = hvac_power_W(Tamb_C, env) if env is not None and Tamb_C == Tamb_C else float('nan')

    P_aux_base = float(getattr(batt, 'P_aux_base_W', 0.0)) if batt is not None else 0.0
    P_aux = float(P_aux_base + (P_hvac if P_hvac == P_hvac else 0.0))

    Tbatt_C = battery_temp_C(Tamb_C, env) if env is not None and Tamb_C == Tamb_C else float('nan')
    P_dis_max, P_chg_max = batt_power_limits_W(Tbatt_C, batt) if batt is not None and Tbatt_C == Tbatt_C else (float('nan'), float('nan'))

    # Vehicle demand
    if veh is not None and common is not None and rho == rho:
        g = float(getattr(common, 'g', 9.80665))
        mass = float(getattr(veh, 'mass_kg', 0.0))
        Crr = float(getattr(veh, 'Crr', 0.0))
        CdA = float(getattr(veh, 'CdA', 0.0))
        r_tire = float(getattr(veh, 'tire_radius_m', 1.0))

        F_roll = mass * g * Crr
        F_aero = 0.5 * rho * CdA * (v_mps ** 2)
        F_iner = mass * a_mps2
        F_total = F_roll + F_aero + F_iner

        T_wheel_req = F_total * r_tire
        P_wheel_req = F_total * v_mps

        wheel_omega = v_mps / max(r_tire, 1e-9)
        final_drive = float(getattr(veh, 'final_drive', 1.0))
        ring_omega = wheel_omega * final_drive
        ring_rpm = ring_omega * 60.0 / (2.0 * np.pi)

        driveline_eff = float(getattr(veh, 'driveline_eff', 1.0))
        T_ring_req = T_wheel_req / max(final_drive * driveline_eff, 1e-9)
    else:
        F_roll = F_aero = F_iner = F_total = float('nan')
        T_wheel_req = P_wheel_req = float('nan')
        wheel_omega = ring_omega = ring_rpm = float('nan')
        T_ring_req = float('nan')
        driveline_eff = float('nan')

    # PSD constants
    if veh is not None:
        Zs = float(getattr(veh, 'Zs', float('nan')))
        Zr = float(getattr(veh, 'Zr', float('nan')))
        if Zs == Zs and Zr == Zr and (Zs + Zr) != 0:
            alpha = Zs / (Zs + Zr)
            beta = Zr / (Zs + Zr)
        else:
            alpha = beta = float('nan')
    else:
        alpha = beta = float('nan')

    # Battery energy bounds
    if batt is not None:
        E_usable_Wh = float(getattr(batt, 'E_usable_Wh', float('nan')))
        soc_min = float(getattr(batt, 'soc_min', float('nan')))
        soc_max = float(getattr(batt, 'soc_max', float('nan')))
        Emin_Wh = soc_min * E_usable_Wh if (soc_min == soc_min and E_usable_Wh == E_usable_Wh) else float('nan')
        Emax_Wh = soc_max * E_usable_Wh if (soc_max == soc_max and E_usable_Wh == E_usable_Wh) else float('nan')
    else:
        E_usable_Wh = Emin_Wh = Emax_Wh = float('nan')

    # Component limits (from veh)
    mg1_rpm_max = float(getattr(veh, 'mg1_rpm_max', float('nan'))) if veh is not None else float('nan')
    mg2_rpm_max = float(getattr(veh, 'mg2_rpm_max', float('nan'))) if veh is not None else float('nan')
    mg1_tq_max_Nm = float(getattr(veh, 'mg1_tq_max_Nm', float('nan'))) if veh is not None else float('nan')
    mg2_tq_max_Nm = float(getattr(veh, 'mg2_tq_max_Nm', float('nan'))) if veh is not None else float('nan')
    eng_rpm_min = float(getattr(veh, 'eng_rpm_min', float('nan'))) if veh is not None else float('nan')
    eng_rpm_max = float(getattr(veh, 'eng_rpm_max', float('nan'))) if veh is not None else float('nan')

    return {
        'dt_s': dt_s,
        'veh_spd_mps': v_mps,
        'veh_acc_mps2': a_mps2,
        'Tamb_C': Tamb_C,
        'rho_air_kgpm3': float(rho),
        'Tbatt_C': float(Tbatt_C),
        'P_hvac_W': float(P_hvac),
        'P_aux_W': float(P_aux),
        'P_dis_max_W': float(P_dis_max),
        'P_chg_max_W': float(P_chg_max),

        'F_roll_N': float(F_roll),
        'F_aero_N': float(F_aero),
        'F_iner_N': float(F_iner),
        'F_total_N': float(F_total),

        'T_wheel_req_Nm': float(T_wheel_req),
        'P_wheel_req_W': float(P_wheel_req),

        'wheel_omega_radps': float(wheel_omega),
        'ring_omega_radps': float(ring_omega),
        'ring_rpm': float(ring_rpm),
        'T_ring_req_Nm': float(T_ring_req),
        'driveline_eff': float(driveline_eff),

        'alpha': float(alpha),
        'beta': float(beta),

        'E_usable_Wh': float(E_usable_Wh),
        'Emin_Wh': float(Emin_Wh),
        'Emax_Wh': float(Emax_Wh),

        'mg1_rpm_max': mg1_rpm_max,
        'mg2_rpm_max': mg2_rpm_max,
        'mg1_tq_max_Nm': mg1_tq_max_Nm,
        'mg2_tq_max_Nm': mg2_tq_max_Nm,
        'eng_rpm_min': eng_rpm_min,
        'eng_rpm_max': eng_rpm_max,
    }


def _default_build_base_row_71col(
    *,
    wltc_row: Dict[str, Any],
    derived: Dict[str, Any],
    base: Any,
    state: Dict[str, Any],
    veh: Any = None,
    batt: Any = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Default 71-column Phase A row builder (best-effort).

    This aims to reproduce sim_grid_A.py row dict keys. Values are filled from:
      - wltc_row (speed/acc/time/phase)
      - derived (vehicle/env computations)
      - base (Phase A solver outputs)

    Missing values are left as NaN.
    """
    from .schema_B import PHASE_A_TIMESERIES_COLUMN_ORDER

    def g(obj: Any, name: str, default: Any = float('nan')) -> Any:
        return getattr(obj, name, default)

    out: Dict[str, Any] = {k: float('nan') for k in PHASE_A_TIMESERIES_COLUMN_ORDER}

    # WLTC/meta
    out['phase'] = wltc_row.get('phase', wltc_row.get('Phase', ''))
    out['t_s'] = float(wltc_row.get('t_s', out['t_s']))
    out['dt_s'] = float(wltc_row.get('dt_s', derived.get('dt_s', out['dt_s'])))
    out['veh_spd_mps'] = float(wltc_row.get('veh_spd_mps', wltc_row.get('v_mps', derived.get('veh_spd_mps', out['veh_spd_mps']))))
    out['veh_acc_mps2'] = float(wltc_row.get('veh_acc_mps2', wltc_row.get('a_mps2', derived.get('veh_acc_mps2', out['veh_acc_mps2']))))

    # Env
    out['Tamb_C'] = derived.get('Tamb_C', out['Tamb_C'])
    out['rho_air_kgpm3'] = derived.get('rho_air_kgpm3', out['rho_air_kgpm3'])
    out['Tbatt_C'] = derived.get('Tbatt_C', out['Tbatt_C'])

    # Forces
    out['F_roll_N'] = derived.get('F_roll_N', out['F_roll_N'])
    out['F_aero_N'] = derived.get('F_aero_N', out['F_aero_N'])
    out['F_iner_N'] = derived.get('F_iner_N', out['F_iner_N'])
    out['F_total_N'] = derived.get('F_total_N', out['F_total_N'])

    # Kinematics
    out['wheel_omega_radps'] = derived.get('wheel_omega_radps', out['wheel_omega_radps'])
    out['ring_omega_radps'] = derived.get('ring_omega_radps', out['ring_omega_radps'])
    out['ring_rpm'] = derived.get('ring_rpm', out['ring_rpm'])

    # Wheel/ring demand
    out['T_wheel_req_Nm'] = derived.get('T_wheel_req_Nm', out['T_wheel_req_Nm'])
    out['P_wheel_req_W'] = derived.get('P_wheel_req_W', out['P_wheel_req_W'])
    out['T_ring_req_Nm'] = derived.get('T_ring_req_Nm', out['T_ring_req_Nm'])

    # Solver outputs
    out['mode'] = g(base, 'mode')
    out['fuel_cut'] = int(g(base, 'fuel_cut', 0))

    out['eng_rpm'] = g(base, 'eng_rpm')
    out['eng_tq_Nm'] = g(base, 'eng_tq_Nm')
    out['mg1_rpm'] = g(base, 'mg1_rpm')
    out['mg1_tq_Nm'] = g(base, 'mg1_tq_Nm')
    out['mg2_rpm'] = g(base, 'mg2_rpm')
    out['mg2_tq_Nm'] = g(base, 'mg2_tq_Nm')

    out['P_eng_mech_W'] = g(base, 'P_eng_mech_W')
    out['P_mg1_mech_W'] = g(base, 'P_mg1_mech_W')
    out['P_mg2_mech_W'] = g(base, 'P_mg2_mech_W')
    out['P_mg1_elec_W'] = g(base, 'P_mg1_elec_W')
    out['P_mg2_elec_W'] = g(base, 'P_mg2_elec_W')

    out['P_hvac_W'] = derived.get('P_hvac_W', out['P_hvac_W'])
    out['P_aux_W'] = g(base, 'P_aux_W', derived.get('P_aux_W', out['P_aux_W']))

    out['P_batt_req_W'] = g(base, 'P_batt_req_W')
    out['P_batt_act_W'] = g(base, 'P_batt_act_W')

    out['P_brake_fric_W'] = g(base, 'P_brake_fric_W')
    out['shortfall_tq_Nm'] = g(base, 'shortfall_tq_Nm')
    out['shortfall_power_W'] = g(base, 'shortfall_power_W')

    out['soc_pct'] = float(g(base, 'soc_next', state.get('soc', float('nan'))) * 100.0) if hasattr(base, 'soc_next') else float('nan')
    out['E_batt_Wh'] = g(base, 'E_batt_next_Wh', state.get('E_batt_Wh', float('nan')))

    out['lim_batt_discharge_W'] = derived.get('P_dis_max_W', out['lim_batt_discharge_W'])
    out['lim_batt_charge_W'] = derived.get('P_chg_max_W', out['lim_batt_charge_W'])

    # Objective
    out['J_total'] = g(base, 'J_total')
    out['J_fuel'] = g(base, 'J_fuel')
    out['J_soc'] = g(base, 'J_soc')
    out['J_fric'] = g(base, 'J_fric')
    out['J_short'] = g(base, 'J_short')
    out['J_spin'] = g(base, 'J_spin')
    out['J_smooth'] = g(base, 'J_smooth')
    out['J_charge'] = g(base, 'J_charge')

    out['excess_tq_Nm'] = g(base, 'excess_tq_Nm')
    out['J_over'] = g(base, 'J_over')

    # Grid stats
    stats = getattr(base, 'stats', None)
    out['n_grid_total'] = getattr(stats, 'n_total', float('nan')) if stats is not None else float('nan')
    out['n_grid_kept'] = getattr(stats, 'n_kept', float('nan')) if stats is not None else float('nan')

    # Battery constants
    if batt is not None:
        E_usable = float(getattr(batt, 'E_usable_Wh', float('nan')))
        out['batt_E_usable_Wh'] = E_usable
        out['batt_Emin_Wh'] = float(getattr(batt, 'soc_min', float('nan')) * E_usable) if E_usable == E_usable else float('nan')
        out['batt_Emax_Wh'] = float(getattr(batt, 'soc_max', float('nan')) * E_usable) if E_usable == E_usable else float('nan')

    # Flags
    if veh is not None:
        eng_tq_max = float(getattr(veh, 'eng_tq_max_Nm', float('nan')))
        mg1_tq_max = float(getattr(veh, 'mg1_tq_max_Nm', float('nan')))
        mg2_tq_max = float(getattr(veh, 'mg2_tq_max_Nm', float('nan')))
        mg1_rpm_max = float(getattr(veh, 'mg1_rpm_max', float('nan')))
        mg2_rpm_max = float(getattr(veh, 'mg2_rpm_max', float('nan')))

        out['flag_eng_sat'] = int((out['fuel_cut'] == 0) and abs(float(out['eng_tq_Nm'])) >= (eng_tq_max - 1e-9)) if eng_tq_max == eng_tq_max else 0
        out['flag_mg1_sat'] = int(abs(float(out['mg1_tq_Nm'])) >= (mg1_tq_max - 1e-9)) if mg1_tq_max == mg1_tq_max else 0
        out['flag_mg2_sat'] = int(abs(float(out['mg2_tq_Nm'])) >= (mg2_tq_max - 1e-9)) if mg2_tq_max == mg2_tq_max else 0
        out['flag_mg1_overspeed'] = int(abs(float(out['mg1_rpm'])) > (mg1_rpm_max + 1e-6)) if mg1_rpm_max == mg1_rpm_max else 0
        out['flag_mg2_overspeed'] = int(abs(float(out['mg2_rpm'])) > (mg2_rpm_max + 1e-6)) if mg2_rpm_max == mg2_rpm_max else 0

    out['flag_batt_sat'] = int(abs(float(out['P_batt_req_W']) - float(out['P_batt_act_W'])) > 1e-6) if (out['P_batt_req_W'] == out['P_batt_req_W'] and out['P_batt_act_W'] == out['P_batt_act_W']) else 0

    # Torque balance visibility
    beta = float(derived.get('beta', float('nan')))
    if beta == beta and out['eng_tq_Nm'] == out['eng_tq_Nm'] and out['mg2_tq_Nm'] == out['mg2_tq_Nm']:
        T_ring_deliv = beta * float(out['eng_tq_Nm']) + float(out['mg2_tq_Nm'])
        out['T_ring_deliv_Nm'] = float(T_ring_deliv)
        if out['T_ring_req_Nm'] == out['T_ring_req_Nm'] and out['shortfall_tq_Nm'] == out['shortfall_tq_Nm']:
            out['resid_ring_torque_Nm_dbg'] = float(float(out['T_ring_req_Nm']) - T_ring_deliv - float(out['shortfall_tq_Nm']))
    # wheel delivered
    if veh is not None and out['T_ring_deliv_Nm'] == out['T_ring_deliv_Nm'] and out['ring_omega_radps'] == out['ring_omega_radps']:
        driveline_eff = float(getattr(veh, 'driveline_eff', 1.0))
        P_wheel_deliv = float(out['T_ring_deliv_Nm']) * float(out['ring_omega_radps']) * driveline_eff
        out['P_wheel_deliv_W_dbg'] = float(P_wheel_deliv)
        if out['P_wheel_req_W'] == out['P_wheel_req_W'] and out['P_brake_fric_W'] == out['P_brake_fric_W']:
            out['resid_wheel_power_W_dbg'] = float(float(out['P_wheel_req_W']) - P_wheel_deliv + float(out['P_brake_fric_W']))

    # Debug mirrors
    out['wheel_omega_radps_dbg'] = out['wheel_omega_radps']
    out['ring_omega_radps_dbg'] = out['ring_omega_radps']
    out['T_wheel_req_Nm_dbg'] = out['T_wheel_req_Nm']
    out['T_ring_req_Nm_dbg'] = out['T_ring_req_Nm']

    return out
