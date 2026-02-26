from __future__ import annotations

"""Phase B objective (optional isolation layer).

This module is OPTIONAL per frozen DAG.
It exists to keep weight-tuning out of `step_B.py`.

Frozen philosophy priority order:
1) Wheel demand tracking (hard)
2) Fuel minimization
3) SOC regulation
4) Loss minimization
5) Constraint avoidance + smoothness
"""

from dataclasses import dataclass
from typing import Protocol


class StepBaseLike(Protocol):
    """Minimum base result interface needed for scoring."""

    shortfall_tq_Nm: float
    J_over: float
    soc_next: float
    eng_rpm: float
    mg1_rpm: float


@dataclass(frozen=True)
class ObjectiveWeightsB:
    w_track: float = 1e6
    w_fuel: float = 1.0
    w_soc: float = 2e4
    w_loss: float = 1.0
    w_sat: float = 1e3
    w_smooth: float = 1e-4


def score_candidate_B(*, base: StepBaseLike, P_fuel_W: float, loss_total_W: float, soc_target: float, soc_band: float, weights: ObjectiveWeightsB) -> float:
    """Return scalar objective value for a candidate.

    Args:
        base: Base-like outputs including slack.
        P_fuel_W: Fuel chemical power [W].
        loss_total_W: Total electrical losses [W].
        soc_target: SOC target [0..1].
        soc_band: band half-width.
        weights: weight set.

    Returns:
        Objective scalar (lower is better).

    Raises:
        NotImplementedError: Until Phase B.1.
    """
    raise NotImplementedError("Phase B.1 - objective pending")
