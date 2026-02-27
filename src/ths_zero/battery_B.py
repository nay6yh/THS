from __future__ import annotations

"""Phase B battery models.

FROZEN CONTRACTS:
- Terminal power sign (IMMUTABLE):
    P_term_W > 0  discharge (battery -> bus)
    P_term_W < 0  charge    (bus -> battery)

- Loss is always non-negative:
    loss_batt_W >= 0

- Chemical power relation (CORRECTED & FROZEN):
    P_chem_W has the SAME sign as P_term_W.
    Loss is always dissipated as heat (>=0), so the chemical power magnitude is:
      |P_chem| = |P_term| + loss during discharge
      |P_chem| = |P_term| - loss during charge

  With the frozen sign convention (+ discharge, - charge), the convenient identity is:
      P_chem_W = P_term_W + loss_batt_W

  (Because adding a positive loss makes P_chem larger: more positive for discharge,
   and less negative for charge.)

- Phase B.0 implements B0 only (efficiency-based).
- B1 exists as a stub with the same interface.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class BatteryConfigB:
    """Phase B battery model selection & parameters (FROZEN)."""

    model: Literal["B0", "B1"] = "B0"

    # B0 params
    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    # B1 params (optional)
    voc_table: Optional[Callable[[float], float]] = None  # SOC -> V
    rint_table: Optional[Callable[[float], float]] = None # SOC -> ohm

    def __post_init__(self) -> None:
        # Basic validation for frozen efficiency bounds
        if not (0.0 < self.eta_charge <= 1.0):
            raise ValueError(f"eta_charge must be in (0,1], got {self.eta_charge}")
        if not (0.0 < self.eta_discharge <= 1.0):
            raise ValueError(f"eta_discharge must be in (0,1], got {self.eta_discharge}")


@dataclass(frozen=True)
class BattChemResult:
    """Battery accounting outputs for Phase B."""

    P_chem_W: float       # chemical power (for SOC integration)
    P_term_W: float       # terminal power (bus-level), equals P_batt_act_W
    loss_batt_W: float    # >=0
    I_batt_A: float = 0.0 # B1 only (0 for B0)


def battery_B0(*, P_term_W: float, cfg: BatteryConfigB) -> BattChemResult:
    """Efficiency-based model (Phase B.0).

    Args:
        P_term_W: Terminal power [W]. + discharge, - charge.
        cfg: BatteryConfigB (uses eta_charge / eta_discharge).

    Returns:
        BattChemResult with corrected chemical power relation and non-negative loss.

    Raises:
        ValueError: If efficiency is outside (0,1].
    """

    # cfg validation is handled in BatteryConfigB.__post_init__
    P_term_W = float(P_term_W)

    if P_term_W > 0.0:
        # Discharge: chemical depletion is larger than terminal delivery
        P_chem_W = P_term_W / float(cfg.eta_discharge)
        loss_batt_W = P_chem_W - P_term_W
    elif P_term_W < 0.0:
        # Charge: chemical storage is smaller in magnitude than terminal absorption
        P_chem_W = P_term_W * float(cfg.eta_charge)
        loss_batt_W = abs(P_term_W) - abs(P_chem_W)
    else:
        P_chem_W = 0.0
        loss_batt_W = 0.0

    if loss_batt_W < -1e-9:
        raise ValueError(f"Computed negative battery loss (invalid efficiencies?), loss={loss_batt_W}")
    loss_batt_W = max(loss_batt_W, 0.0)

    # Invariant: P_chem and P_term have the same sign (or both zero)
    if P_term_W != 0.0 and (P_chem_W == 0.0 or (P_chem_W > 0) != (P_term_W > 0)):
        raise ValueError(
            f"P_chem_W must have same sign as P_term_W. P_term={P_term_W}, P_chem={P_chem_W}"
        )

    return BattChemResult(P_chem_W=P_chem_W, P_term_W=P_term_W, loss_batt_W=loss_batt_W, I_batt_A=0.0)


def battery_B1(*, P_term_W: float, soc: float, cfg: BatteryConfigB) -> BattChemResult:
    """Voc-Rint ECM model (stub in Phase B.1).

    Args:
        P_term_W: Terminal power [W]. + discharge, - charge.
        soc: State of charge [0..1].
        cfg: BatteryConfigB with voc_table and rint_table.

    Returns:
        BattChemResult.

    Raises:
        NotImplementedError: Until Phase B.x.
    """
    raise NotImplementedError("Phase B.1 - B1 model pending")
