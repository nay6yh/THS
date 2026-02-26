from __future__ import annotations

"""Phase B loss bookkeeping for MGs and inverters.

FROZEN CONTRACTS:
- All loss terms are non-negative (>=0).
- Inverter losses are split per MG (loss_inv1_W, loss_inv2_W) in the schema,
  but Phase B.0 sets them to 0 when inv_loss_mode == "embedded_in_mg_eta".

Import firewall:
- MUST NOT import Phase A solver modules.
"""

from dataclasses import dataclass
from typing import Literal

from .units_B import assert_nonneg


@dataclass(frozen=True)
class SimConfigB:
    """Phase B sim flags (frozen)."""
    inv_loss_mode: Literal["embedded_in_mg_eta", "separate"] = "embedded_in_mg_eta"


@dataclass(frozen=True)
class LossAccount:
    """Frozen schema - supports future refinement without breaking changes."""

    # MG mechanical<->electrical conversion losses
    loss_mg1_W: float
    loss_mg2_W: float

    # Inverter losses (per-MG)
    loss_inv1_W: float
    loss_inv2_W: float

    # Engine thermal loss (duplicated also in FuelAccount by design)
    loss_engine_W: float

    @property
    def loss_inv_W(self) -> float:
        return float(self.loss_inv1_W + self.loss_inv2_W)


def mg_loss_from_mech_and_bus(*, P_mech_W: float, P_bus_elec_W: float) -> float:
    """Compute MG conversion loss from mechanical and bus electrical powers.

    Assumes Phase A / global sign conventions:
      - P_mech_W > 0 : motoring (elec -> mech) so bus power should be >= P_mech
      - P_mech_W < 0 : generating (mech -> elec) so |bus power| <= |mech power|

    Returns:
        loss_W >= 0
    """
    # direction-aware loss on magnitudes
    if P_mech_W >= 0:
        loss = P_bus_elec_W - P_mech_W
    else:
        loss = abs(P_mech_W) - abs(P_bus_elec_W)

    loss = float(max(loss, 0.0))
    assert_nonneg(loss, name="loss_mg_W")
    return loss


def inverter_losses(*, cfg: SimConfigB, P_mg1_bus_W: float, P_mg2_bus_W: float) -> tuple[float, float]:
    """Compute inverter losses (per MG).

    Frozen behavior:
      - embedded_in_mg_eta: inverter loss is assumed already included in MG efficiency -> returns (0, 0)

    Phase B.0 optional placeholder:
      - separate: use a simple fixed-efficiency placeholder to avoid blocking integration tests.
        (Future Phase B.x will replace this with inverter efficiency maps.)

    Returns:
        (loss_inv1_W, loss_inv2_W) both >= 0
    """
    if cfg.inv_loss_mode == "embedded_in_mg_eta":
        return 0.0, 0.0

    if cfg.inv_loss_mode == "separate":
        # Placeholder constant inverter efficiency (NOT a final model).
        eta_inv = 0.98
        loss1 = abs(float(P_mg1_bus_W)) * (1.0 - eta_inv)
        loss2 = abs(float(P_mg2_bus_W)) * (1.0 - eta_inv)
        loss1 = float(max(loss1, 0.0))
        loss2 = float(max(loss2, 0.0))
        assert_nonneg(loss1, name="loss_inv1_W")
        assert_nonneg(loss2, name="loss_inv2_W")
        return loss1, loss2

    raise ValueError(f"Unknown inv_loss_mode: {cfg.inv_loss_mode!r}")
