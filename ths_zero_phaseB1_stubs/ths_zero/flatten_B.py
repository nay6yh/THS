from __future__ import annotations

"""Phase B flattening utilities.

Purpose:
- Convert StepResultB into a flat dict suitable for appending to a Phase A row dict.

FROZEN OUTPUT:
- Column names must match schema_B.PHASE_B_EXT_COLUMNS (append order).
- Values must satisfy frozen constraints:
  - losses are >= 0
  - split powers are >= 0
  - sim_phase == "B"

Note:
- Phase A timeseries output does not contain I_batt_A; in Phase B it is introduced
  as an extension column.
"""

from typing import Dict, Any

from .schema_B import PHASE_B_EXT_COLUMNS, SIM_PHASE_VALUE
from .step_B import StepResultB


def flatten_phase_b_extensions(sol: StepResultB) -> Dict[str, Any]:
    """Return a dict of Phase B extension columns (append-only)."""

    d: Dict[str, Any] = {
        # Battery chemistry
        'P_batt_chem_W': sol.batt.P_chem_W,
        'loss_batt_W': sol.batt.loss_batt_W,
        'I_batt_A': sol.batt.I_batt_A,

        # Fuel
        'mdot_fuel_gps': sol.fuel.mdot_fuel_gps,
        'P_fuel_W': sol.fuel.P_fuel_W,
        'loss_engine_W': sol.fuel.loss_engine_W,

        # Losses
        'loss_mg1_W': sol.losses.loss_mg1_W,
        'loss_mg2_W': sol.losses.loss_mg2_W,
        'loss_inv1_W': sol.losses.loss_inv1_W,
        'loss_inv2_W': sol.losses.loss_inv2_W,
        'loss_inv_W': sol.losses.loss_inv_W,

        # Split
        'P_batt_chg_from_regen_W': sol.split.P_batt_chg_from_regen_W,
        'P_batt_chg_from_engine_W': sol.split.P_batt_chg_from_engine_W,

        # Metadata
        'sim_phase': SIM_PHASE_VALUE,
    }

    missing = [c for c in PHASE_B_EXT_COLUMNS if c not in d]
    extra = [k for k in d.keys() if k not in PHASE_B_EXT_COLUMNS]
    if missing:
        raise KeyError(f"Missing Phase B extension cols: {missing}")
    if extra:
        raise KeyError(f"Unexpected Phase B extension cols: {extra}")

    return d
