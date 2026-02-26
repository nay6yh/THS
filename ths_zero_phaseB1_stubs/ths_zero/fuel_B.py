from __future__ import annotations

"""Phase B fuel accounting (BSFC -> fuel power -> engine thermal loss).

FROZEN CONTRACTS:
- mdot_fuel_gps >= 0
- P_fuel_W >= 0
- loss_engine_W >= 0
- If engine mechanical power <= 0 (fuel-cut / no positive work): mdot=0, P_fuel=0, loss_engine=0

NOTE:
- This module is Level 1: pure functions, no solver logic.
- No imports from Phase A solver modules.
"""

from dataclasses import dataclass
from typing import Callable

from .units_B import assert_nonneg, rpm_to_radps


BsfcMap = Callable[[float, float], float]  # (eng_rpm, eng_tq_Nm) -> g/kWh


LHV_GASOLINE_J_PER_G: float = 43e3  # default constant, can be overridden via config later


@dataclass(frozen=True)
class FuelAccount:
    """Fuel consumption and engine thermal loss."""
    mdot_fuel_gps: float   # [g/s]
    P_fuel_W: float        # [W]
    loss_engine_W: float   # [W] >= 0


def calc_fuel_account(
    *,
    eng_rpm: float,
    eng_tq_Nm: float,
    bsfc_map: BsfcMap,
    lhv_J_per_g: float = LHV_GASOLINE_J_PER_G,
) -> FuelAccount:
    """Compute fuel and engine loss from BSFC.

    Frozen sign convention:
      - Engine torque/power positive corresponds to positive mechanical output.

    Args:
        eng_rpm: Engine speed [rpm].
        eng_tq_Nm: Engine torque [Nm].
        bsfc_map: Function returning BSFC [g/kWh].
        lhv_J_per_g: Fuel lower heating value [J/g].

    Returns:
        FuelAccount with mdot_fuel_gps, P_fuel_W, loss_engine_W.

    Raises:
        ValueError: If lhv_J_per_g <= 0.
    """

    # Input validation (frozen expectations)
    if eng_rpm < 0:
        raise ValueError(f"eng_rpm must be >= 0, got {eng_rpm}")
    if lhv_J_per_g <= 0:
        raise ValueError(f"lhv_J_per_g must be > 0, got {lhv_J_per_g}")

    omega = rpm_to_radps(eng_rpm)
    P_mech_W = omega * float(eng_tq_Nm)

    # Fuel-cut / no positive mechanical work
    if P_mech_W <= 0.0:
        return FuelAccount(mdot_fuel_gps=0.0, P_fuel_W=0.0, loss_engine_W=0.0)

    # BSFC lookup
    try:
        bsfc_gpkWh = float(bsfc_map(float(eng_rpm), float(eng_tq_Nm)))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bsfc_map lookup failed for rpm={eng_rpm}, tq={eng_tq_Nm}") from e

    if not (bsfc_gpkWh > 0.0):
        raise ValueError(f"bsfc_map must return > 0 g/kWh, got {bsfc_gpkWh}")

    # mdot [g/s] = BSFC [g/kWh] * P[kW] / 3600
    P_kW = P_mech_W / 1000.0
    mdot_gps = bsfc_gpkWh * P_kW / 3600.0
    assert_nonneg(mdot_gps, name="mdot_fuel_gps")

    P_fuel_W = mdot_gps * float(lhv_J_per_g)
    assert_nonneg(P_fuel_W, name="P_fuel_W")

    loss_engine_W = P_fuel_W - P_mech_W
    # Physically, BSFC implies P_fuel >= P_mech. If violated, treat as invalid input.
    if loss_engine_W < -1e-9:
        raise ValueError(
            "Invalid BSFC/LHV caused P_fuel < P_mech (negative engine loss): "
            f"P_fuel={P_fuel_W}, P_mech={P_mech_W}, loss={loss_engine_W}"
        )
    loss_engine_W = max(loss_engine_W, 0.0)
    assert_nonneg(loss_engine_W, name="loss_engine_W")

    return FuelAccount(mdot_fuel_gps=mdot_gps, P_fuel_W=P_fuel_W, loss_engine_W=loss_engine_W)
