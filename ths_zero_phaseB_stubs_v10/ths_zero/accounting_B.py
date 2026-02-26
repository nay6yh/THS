from __future__ import annotations

"""Phase B energy accounting utilities (pure functions).

FROZEN CONTRACTS:
- Battery sign: P_batt_term_W < 0 is charging (bus -> battery).
- MG bus sign: P_mg*_bus_W < 0 indicates generating into DC bus.
- split_charging_origin returns two non-negative powers whose sum equals charging magnitude.
"""

from typing import Iterable, Tuple

from .units_B import assert_nonneg


def split_charging_origin(
    *,
    P_batt_term_W: float,
    P_mg1_bus_W: float,
    P_mg2_bus_W: float,
    # Optional context to avoid regen metrics > 1 (Phase B robustness):
    # - Provide either wheel delivered power or a braking flag/mode.
    P_wheel_deliv_W_dbg: float | None = None,
    mode: str | None = None,
    is_braking: bool | None = None,
    eps: float = 1e-9,
) -> Tuple[float, float]:
    """Attribute battery charging to regen vs engine sources (FROZEN).

    Sign conventions (CRITICAL):
      - P_batt_term_W < 0  -> charging (bus -> battery)
      - P_mg1_bus_W < 0    -> MG1 generating (engine -> bus)
      - P_mg2_bus_W < 0    -> MG2 generating (wheel/regen -> bus)

    Returns:
        (P_from_regen_W, P_from_engine_W) both >= 0

    Invariants:
        P_from_regen_W + P_from_engine_W == max(-P_batt_term_W, 0)

    Notes:
      - By default (no braking context provided), this function uses the original
        frozen attribution rule: prioritize MG2 generation as regen first.
      - If braking context is provided (P_wheel_deliv_W_dbg or is_braking or mode),
        then *regen attribution is only allowed during braking*.
        This prevents counting engine-charging/power-circulation as "regen" and
        helps keep regen_utilization in [0,1].
    """
    P_chg = max(-P_batt_term_W, 0.0)
    if P_chg <= eps:
        return 0.0, 0.0

    # Determine braking context if provided
    braking_ctx: bool | None = None

    # 1) explicit flag wins
    if is_braking is not None:
        braking_ctx = bool(is_braking)

    # 2) wheel delivered power (negative => braking) if available and finite
    if braking_ctx is None and P_wheel_deliv_W_dbg is not None:
        try:
            x = float(P_wheel_deliv_W_dbg)
            if x == x:  # not NaN
                braking_ctx = (x < 0.0)
        except Exception:
            braking_ctx = None

    # 3) mode hint
    if braking_ctx is None and mode is not None:
        braking_ctx = (str(mode) in ("Regen", "FrictionBrake"))

    # Generation sources (both >= 0)
    gen_mg1 = max(-P_mg1_bus_W, 0.0)  # engine source
    gen_mg2 = max(-P_mg2_bus_W, 0.0)  # regen candidate (MG2 generating)

    # If we have braking context and it's NOT braking: never count MG2 generation as regen.
    # Treat any battery charging as engine-origin (or "other") to avoid regen metrics > 1.
    if braking_ctx is False:
        P_from_regen = 0.0
        P_from_engine = P_chg
        assert_nonneg(P_from_engine, name="P_from_engine_W")
        return float(P_from_regen), float(P_from_engine)

    # Default / braking case: prioritize regen attribution
    P_from_regen = min(P_chg, gen_mg2)

    # Optional hard cap by wheel negative power magnitude (if provided)
    if P_wheel_deliv_W_dbg is not None:
        try:
            x = float(P_wheel_deliv_W_dbg)
            if x == x and x < 0.0:
                P_from_regen = min(P_from_regen, -x)
        except Exception:
            pass

    P_from_engine = max(P_chg - P_from_regen, 0.0)

    assert_nonneg(P_from_regen, name="P_from_regen_W")
    assert_nonneg(P_from_engine, name="P_from_engine_W")

    if abs((P_from_regen + P_from_engine) - P_chg) > 1e-6:
        raise AssertionError("split_charging_origin invariant violated")

    return float(P_from_regen), float(P_from_engine)
def bus_residual_W(*, P_batt_term_W: float, P_mg1_bus_W: float, P_mg2_bus_W: float, P_aux_W: float) -> float:
    """Compute DC bus residual.

    Frozen bus closure (Phase A contract preserved):
        P_batt_term_W ≈ P_mg1_bus_W + P_mg2_bus_W + P_aux_W

    Returns:
        residual_W = P_batt_term_W - (P_mg1_bus_W + P_mg2_bus_W + P_aux_W)
    """
    return float(P_batt_term_W - (P_mg1_bus_W + P_mg2_bus_W + P_aux_W))


def integrate_trapz_J(*, P_W: Iterable[float], dt_s: Iterable[float]) -> float:
    """Integrate power [W] over time [s] into energy [J] using trapezoidal rule.

    Args:
        P_W: Power samples [W], length N.
        dt_s: Time step samples [s], length N-1 (standard) or N (dt aligned per sample; last ignored).

    Returns:
        Total energy [J] (can be negative if P_W contains negative values).

    Raises:
        ValueError: If input lengths are inconsistent or if any dt is negative.
    """
    P = [float(x) for x in P_W]
    dt = [float(x) for x in dt_s]

    if len(P) < 2:
        return 0.0

    if len(dt) == len(P):
        dt = dt[:-1]
    elif len(dt) == len(P) - 1:
        pass
    else:
        raise ValueError(f"Length mismatch: len(P_W)={len(P)}, len(dt_s)={len(dt)} (expected N or N-1)")

    for i, d in enumerate(dt):
        if d < 0:
            raise ValueError(f"dt_s[{i}] is negative: {d}")

    E = 0.0
    for i, d in enumerate(dt):
        E += 0.5 * (P[i] + P[i + 1]) * d

    return float(E)
