from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


def fake_bsfc_g_per_kwh(eng_rpm: float, eng_tq_Nm: float) -> float:
    """
    Placeholder BSFC bowl.
    Replace with real BSFC map interpolation later.
    """
    dr = (eng_rpm - 2400.0) / 1200.0
    dt = (eng_tq_Nm - 95.0) / 60.0
    return float(220.0 + 60.0 * (dr * dr + dt * dt))

def load_bsfc_map_grid_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a grid-style BSFC map CSV.

    Expected format:
      eng_rpm, 0,10,20,...,120
      1000, 999, 320, 280, ...

    Returns:
      rpm_grid: (Nr,)
      tq_grid:  (Nt,)
      bsfc:     (Nr, Nt)  in g/kWh
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"BSFC map csv is empty or too short: {csv_path}")

    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"BSFC map csv header must have eng_rpm and torque columns: {csv_path}")

    # torque axis from header[1:]
    try:
        tq_grid = np.asarray([float(x) for x in header[1:]], dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse torque axis from header: {header}") from e

    rpm_list: list[float] = []
    bsfc_list: list[list[float]] = []
    for r in rows[1:]:
        if not r or all((str(x).strip() == "" for x in r)):
            continue
        if len(r) < 2:
            continue
        rpm_list.append(float(r[0]))
        bsfc_list.append([float(x) for x in r[1:]])

    rpm_grid = np.asarray(rpm_list, dtype=float)
    bsfc = np.asarray(bsfc_list, dtype=float)

    if bsfc.shape != (rpm_grid.size, tq_grid.size):
        raise ValueError(
            f"BSFC grid shape mismatch: bsfc={bsfc.shape}, rpm={rpm_grid.size}, tq={tq_grid.size}"
        )

    # sort by rpm ascending just in case
    order = np.argsort(rpm_grid)
    rpm_grid = rpm_grid[order]
    bsfc = bsfc[order, :]

    return rpm_grid, tq_grid, bsfc


def _ffill_bfill_1d(x: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaNs in 1D."""
    y = x.copy()
    # forward fill
    last = np.nan
    for i in range(y.size):
        if np.isfinite(y[i]):
            last = y[i]
        elif np.isfinite(last):
            y[i] = last
    # backward fill
    last = np.nan
    for i in range(y.size - 1, -1, -1):
        if np.isfinite(y[i]):
            last = y[i]
        elif np.isfinite(last):
            y[i] = last
    return y


def _fill_nans_2d(z: np.ndarray, fill_value: float) -> np.ndarray:
    """Fill NaNs in 2D by row/col ffillbfill, then fallback to fill_value."""
    a = z.copy()
    # row-wise
    for i in range(a.shape[0]):
        a[i, :] = _ffill_bfill_1d(a[i, :])
    # col-wise
    for j in range(a.shape[1]):
        a[:, j] = _ffill_bfill_1d(a[:, j])
    # final fallback
    a[~np.isfinite(a)] = float(fill_value)
    return a


@dataclass(frozen=True)
class BsfcMap2D:
    """
    2D BSFC map with bilinear interpolation and clamp OOB policy.

    Callable signature matches fuel_B / sim_grid_B expectation:
      bsfc_map(eng_rpm, eng_tq_Nm) -> g/kWh
    """
    rpm_grid: np.ndarray
    tq_grid: np.ndarray
    bsfc_g_per_kWh: np.ndarray
    min_bsfc: float = 120.0
    max_bsfc: float = 700.0

    def __call__(self, eng_rpm: float, eng_tq_Nm: float) -> float:
        r = float(eng_rpm)
        t = float(eng_tq_Nm)

        rpm = self.rpm_grid
        tq = self.tq_grid
        z = self.bsfc_g_per_kWh

        # clamp OOB
        r = float(np.clip(r, rpm[0], rpm[-1]))
        t = float(np.clip(t, tq[0], tq[-1]))

        # cell indices
        i = int(np.searchsorted(rpm, r) - 1)
        j = int(np.searchsorted(tq, t) - 1)
        i = int(np.clip(i, 0, rpm.size - 2))
        j = int(np.clip(j, 0, tq.size - 2))

        r0, r1 = float(rpm[i]), float(rpm[i + 1])
        t0, t1 = float(tq[j]), float(tq[j + 1])
        fr = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
        ft = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)

        z00 = float(z[i, j])
        z01 = float(z[i, j + 1])
        z10 = float(z[i + 1, j])
        z11 = float(z[i + 1, j + 1])

        # bilinear interpolation
        z0 = (1.0 - ft) * z00 + ft * z01
        z1 = (1.0 - ft) * z10 + ft * z11
        val = (1.0 - fr) * z0 + fr * z1

        # clamp to sane bounds (still strictly >0)
        val = float(np.clip(val, self.min_bsfc, self.max_bsfc))
        return val


def make_bsfc_map_from_grid_csv(
    csv_path: str,
    *,
    invalid_threshold: float = 900.0,
    fill_value: float = 350.0,
    min_bsfc: float = 120.0,
    max_bsfc: float = 700.0,
) -> BsfcMap2D:
    """
    Build a callable BSFC map from a grid CSV.

    - Any cell >= invalid_threshold is treated as invalid and filled.
    - NaNs are filled by ffill/bfill in row/col directions.
    - OOB policy is clamp in BsfcMap2D.__call__.
    """
    rpm, tq, bsfc = load_bsfc_map_grid_csv(csv_path)
    bsfc = bsfc.astype(float)

    # mark invalid
    bsfc = np.where(bsfc >= float(invalid_threshold), np.nan, bsfc)
    bsfc = _fill_nans_2d(bsfc, float(fill_value))

    return BsfcMap2D(
        rpm_grid=rpm,
        tq_grid=tq,
        bsfc_g_per_kWh=bsfc,
        min_bsfc=float(min_bsfc),
        max_bsfc=float(max_bsfc),
    )


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