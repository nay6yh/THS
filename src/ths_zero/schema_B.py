from __future__ import annotations

"""Phase B column schema + ordering (FROZEN).

This module freezes the *true* Phase A timeseries contract and defines the
Phase B append columns. The intent is strict dashboard compatibility:

- Phase A timeseries output from sim_grid_A.py is the source of truth.
- Phase B outputs must preserve Phase A columns (same names + same order),
  and then append Phase B extension columns (same names + same order).

Import rules (IMMUTABLE):
- This module is a leaf dependency (no imports from other project modules).
- Phase A modules MUST NEVER import Phase B modules.

Battery sign convention (IMMUTABLE):
- P_batt_act_W > 0  -> discharge (battery -> bus)
- P_batt_act_W < 0  -> charge    (bus -> battery)

Notes on duplicates:
- The Phase A timeseries does *not* contain I_batt_A.
  In Phase B it is introduced as an extension column.
"""

from typing import Any, Dict, List

SIM_PHASE_VALUE: str = "B"

# ---------------------------------------------------------------------------
# Phase A TRUE timeseries column order (FROZEN)
# Source of truth: sim_grid_A.py rows.append({...}) key order.
# ---------------------------------------------------------------------------

PHASE_A_TIMESERIES_COLUMN_ORDER: List[str] = [
    'phase',
    't_s',
    'dt_s',
    'veh_spd_mps',
    'veh_acc_mps2',
    'Tamb_C',
    'rho_air_kgpm3',
    'Tbatt_C',
    'F_roll_N',
    'F_aero_N',
    'F_iner_N',
    'F_total_N',
    'wheel_omega_radps',
    'ring_omega_radps',
    'ring_rpm',
    'T_wheel_req_Nm',
    'P_wheel_req_W',
    'T_ring_req_Nm',
    'mode',
    'fuel_cut',
    'eng_rpm',
    'eng_tq_Nm',
    'mg1_rpm',
    'mg1_tq_Nm',
    'mg2_rpm',
    'mg2_tq_Nm',
    'P_eng_mech_W',
    'P_mg1_mech_W',
    'P_mg2_mech_W',
    'P_mg1_elec_W',
    'P_mg2_elec_W',
    'P_hvac_W',
    'P_aux_W',
    'P_batt_req_W',
    'P_batt_act_W',
    'P_brake_fric_W',
    'shortfall_tq_Nm',
    'shortfall_power_W',
    'soc_pct',
    'E_batt_Wh',
    'lim_batt_discharge_W',
    'lim_batt_charge_W',
    'J_total',
    'J_fuel',
    'J_soc',
    'J_fric',
    'J_short',
    'J_spin',
    'J_smooth',
    'J_charge',
    'excess_tq_Nm',
    'J_over',
    'n_grid_total',
    'n_grid_kept',
    'batt_E_usable_Wh',
    'batt_Emin_Wh',
    'batt_Emax_Wh',
    'flag_eng_sat',
    'flag_mg1_sat',
    'flag_mg2_sat',
    'flag_batt_sat',
    'flag_mg1_overspeed',
    'flag_mg2_overspeed',
    'T_ring_deliv_Nm',
    'resid_ring_torque_Nm_dbg',
    'P_wheel_deliv_W_dbg',
    'resid_wheel_power_W_dbg',
    'wheel_omega_radps_dbg',
    'ring_omega_radps_dbg',
    'T_wheel_req_Nm_dbg',
    'T_ring_req_Nm_dbg',
]

# Optional: a small canonical view subset (useful for dashboards).
# This is NOT used for strict output validation.
PHASE_A_CANONICAL_COLUMN_ORDER: List[str] = [
    't_s',
    'veh_spd_mps',
    'veh_acc_mps2',
    'P_wheel_req_W',
    'eng_rpm',
    'eng_tq_Nm',
    'P_eng_mech_W',
    'mg1_rpm',
    'mg1_tq_Nm',
    'P_mg1_mech_W',
    'P_mg1_elec_W',
    'mg2_rpm',
    'mg2_tq_Nm',
    'P_mg2_mech_W',
    'P_mg2_elec_W',
    'P_batt_act_W',
    'soc_pct',
    'J_total',
    'P_brake_fric_W',
    'shortfall_power_W',
]

# ---------------------------------------------------------------------------
# Phase B extension columns (FROZEN append order)
# ---------------------------------------------------------------------------

PHASE_B_EXT_COLUMNS: List[str] = [
    # Battery chemistry (new)
    'P_batt_chem_W',
    'loss_batt_W',
    'I_batt_A',

    # Fuel
    'mdot_fuel_gps',
    'P_fuel_W',
    'loss_engine_W',

    # Losses
    'loss_mg1_W',
    'loss_mg2_W',
    'loss_inv1_W',
    'loss_inv2_W',
    'loss_inv_W',

    # Charging split
    'P_batt_chg_from_regen_W',
    'P_batt_chg_from_engine_W',

    # Metadata
    'sim_phase',
]

# Full Phase B output ordering
PHASE_B_OUTPUT_COLUMN_ORDER: List[str] = PHASE_A_TIMESERIES_COLUMN_ORDER + PHASE_B_EXT_COLUMNS

# Canonical lightweight dashboard output (optional).
# This keeps Phase A canonical subset + Phase B extensions (including sim_phase).
PHASE_B_CANONICAL_OUTPUT_ORDER: List[str] = PHASE_A_CANONICAL_COLUMN_ORDER + PHASE_B_EXT_COLUMNS


def extract_canonical_view(ts: Any, *, strict: bool = True) -> Any:
    """Extract a lightweight canonical view for dashboards.

    Args:
        ts: A DataFrame-like object (typically pandas.DataFrame) containing Phase B output.
        strict: If True, raise ValueError when any required column is missing.

    Returns:
        A copy/slice of ts with columns = PHASE_B_CANONICAL_OUTPUT_ORDER.

    Notes:
        - This function does not change the frozen full output contract (85 cols).
        - Intended for dashboards that do not need the full 71-col Phase A detail.
    """
    cols = PHASE_B_CANONICAL_OUTPUT_ORDER
    if strict:
        missing = [c for c in cols if c not in getattr(ts, "columns", [])]
        if missing:
            raise ValueError(f"timeseries missing required canonical columns: {missing}")
    # pandas-style slicing
    try:
        return ts[cols].copy()
    except Exception:
        # Fallback: return best-effort subset if strict=False
        if strict:
            raise
        avail = [c for c in cols if c in getattr(ts, "columns", [])]
        return ts[avail].copy()

def build_schema_B() -> Dict[str, Any]:
    """Return a minimal schema dictionary for Phase B outputs (JSON-serializable)."""
    return {
        "phase": "B",
        "frozen": True,
        "sim_phase": SIM_PHASE_VALUE,
        "phase_a_timeseries_column_order": PHASE_A_TIMESERIES_COLUMN_ORDER,
        "phase_a_canonical_column_order": PHASE_A_CANONICAL_COLUMN_ORDER,
        "phase_b_extension_columns": PHASE_B_EXT_COLUMNS,
        "column_order_output": PHASE_B_OUTPUT_COLUMN_ORDER,
    }


def validate_column_order(df_or_columns: Any, *, strict: bool = False) -> None:
    """Validate column order against the frozen Phase B template.

    Args:
        df_or_columns:
            Either:
              - an object with a `.columns` attribute (e.g., pandas/polars DataFrame), or
              - an explicit list/tuple of column names.
        strict:
            False (default): allow prefix match.
            True: require exact match to PHASE_B_OUTPUT_COLUMN_ORDER.

    Raises:
        TypeError: if df_or_columns is not DataFrame-like and not a list/tuple.
        ValueError: if the order does not match the frozen template.
    """
    if isinstance(df_or_columns, (list, tuple)):
        actual = list(df_or_columns)
    else:
        cols = getattr(df_or_columns, 'columns', None)
        if cols is None:
            raise TypeError('validate_column_order expects a DataFrame-like object or list of columns')
        actual = list(cols)

    expected = PHASE_B_OUTPUT_COLUMN_ORDER

    if strict:
        if actual != expected:
            raise ValueError(
                f"Column order mismatch (strict)!\nExpected: {expected}\nGot: {actual}"
            )
        return

    # prefix match
    if actual != expected[:len(actual)]:
        raise ValueError(
            f"Column order mismatch (prefix)!\nExpected prefix: {expected[:len(actual)]}\nGot: {actual}"
        )
