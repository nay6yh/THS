from __future__ import annotations

import numpy as np


def fake_bsfc_g_per_kwh(eng_rpm: float, eng_tq_Nm: float) -> float:
    """
    Placeholder BSFC bowl.
    Replace with real BSFC map interpolation later.
    """
    dr = (eng_rpm - 2400.0) / 1200.0
    dt = (eng_tq_Nm - 95.0) / 60.0
    return float(220.0 + 60.0 * (dr * dr + dt * dt))


def pick_engine_point_min_bsfc(
    P_eng_cmd_W: float,
    ring_spd_rpm: float,
    alpha: float,
    beta: float,
    eng_rpm_min: float,
    eng_rpm_max: float,
    eng_tq_max_Nm: float,
    mg1_rpm_max: float,
) -> tuple[float, float]:
    """
    Choose (eng_rpm, eng_tq) to match engine power command, and keep MG1 speed feasible:
      eng_spd = alpha*mg1_spd + beta*ring_spd  -> mg1_spd = (eng_spd - beta*ring)/alpha
    """
    if P_eng_cmd_W <= 0.0:
        return 0.0, 0.0

    cand_rpm = np.arange(eng_rpm_min, eng_rpm_max + 1e-9, 100.0)
    best = None

    for eng_rpm in cand_rpm:
        omega = eng_rpm * 2 * np.pi / 60.0
        tq = P_eng_cmd_W / max(omega, 1e-9)
        if tq <= 0 or tq > eng_tq_max_Nm:
            continue

        omega_ring = ring_spd_rpm * 2 * np.pi / 60.0
        omega_mg1 = (omega - beta * omega_ring) / max(alpha, 1e-9)
        mg1_rpm = omega_mg1 * 60.0 / (2 * np.pi)

        if abs(mg1_rpm) > mg1_rpm_max:
            continue

        bsfc = fake_bsfc_g_per_kwh(eng_rpm, tq)
        if best is None or bsfc < best[0]:
            best = (bsfc, eng_rpm, tq)

    if best is None:
        # fallback
        eng_rpm = eng_rpm_min
        omega = eng_rpm * 2 * np.pi / 60.0
        tq = min(P_eng_cmd_W / max(omega, 1e-9), eng_tq_max_Nm)
        return float(eng_rpm), float(tq)

    return float(best[1]), float(best[2])