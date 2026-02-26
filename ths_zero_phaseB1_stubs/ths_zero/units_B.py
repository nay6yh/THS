from __future__ import annotations

"""Phase B unit conversions & sign helpers (FROZEN).

This module exists to avoid importing Phase A solver modules (e.g., step_A.py)
just to reuse small helpers. It is a leaf dependency.

Sign conventions (IMMUTABLE):
- Battery power: P_batt_act_W > 0 discharge, < 0 charge
- Electrical power: P_elec_W > 0 consumes electrical, < 0 generates
"""

import math
from typing import TypeVar

T = TypeVar("T", int, float)


def rpm_to_radps(rpm: float) -> float:
    """Convert rpm to rad/s."""
    return float(rpm * 2.0 * math.pi / 60.0)


def radps_to_rpm(radps: float) -> float:
    """Convert rad/s to rpm."""
    return float(radps * 60.0 / (2.0 * math.pi))


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return float(min(max(x, lo), hi))


def sign(x: float, eps: float = 1e-12) -> int:
    """Return -1, 0, or +1 depending on x."""
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def is_nonneg(x: float, eps: float = 0.0) -> bool:
    """True if x >= -eps."""
    return bool(x >= -eps)


def assert_nonneg(x: float, name: str = "value", eps: float = 1e-9) -> None:
    """Raise ValueError if x is negative beyond tolerance."""
    if x < -eps:
        raise ValueError(f"{name} must be >= 0 (tol={eps}), got {x}")
