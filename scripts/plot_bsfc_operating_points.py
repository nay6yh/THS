from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_bsfc_grid(bsfc_csv: Path, invalid_threshold: float = 900.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid CSV format:
      eng_rpm,0,10,20,...,120
      1000,999,320,280,...
    Returns:
      rpm_grid (Nr,), tq_grid (Nt,), bsfc (Nr,Nt) with invalid cells as NaN.
    """
    df = pd.read_csv(bsfc_csv)
    if "eng_rpm" not in df.columns:
        raise ValueError(f"BSFC CSV must have 'eng_rpm' column, got: {df.columns.tolist()}")

    rpm = df["eng_rpm"].to_numpy(dtype=float)
    tq_cols = [c for c in df.columns if c != "eng_rpm"]
    tq = np.asarray([float(c) for c in tq_cols], dtype=float)

    z = df[tq_cols].to_numpy(dtype=float)
    z = np.where(z >= invalid_threshold, np.nan, z)

    # sort by rpm in case
    order = np.argsort(rpm)
    rpm = rpm[order]
    z = z[order, :]
    return rpm, tq, z


def load_operating_points(ts_csv: Path) -> pd.DataFrame:
    """
    Needs at least:
      eng_rpm, eng_tq_Nm
    Optional (for filtering):
      mdot_fuel_gps, P_fuel_W, P_eng_mech_W
    """
    ts = pd.read_csv(ts_csv)

    for c in ("eng_rpm", "eng_tq_Nm"):
        if c not in ts.columns:
            raise ValueError(f"timeseries missing required column '{c}'")

    # Fuel-on mask (best effort)
    if all(c in ts.columns for c in ("mdot_fuel_gps", "P_fuel_W", "P_eng_mech_W")):
        fuel_on = (ts["mdot_fuel_gps"].to_numpy(float) > 0.0) & (ts["P_fuel_W"].to_numpy(float) > 1.0) & (
            ts["P_eng_mech_W"].to_numpy(float) > 1.0
        )
    elif "mdot_fuel_gps" in ts.columns:
        fuel_on = ts["mdot_fuel_gps"].to_numpy(float) > 0.0
    else:
        fuel_on = np.ones(len(ts), dtype=bool)

    ts = ts.copy()
    ts["fuel_on"] = fuel_on

    # Implied BSFC for coloring (optional)
    if all(c in ts.columns for c in ("mdot_fuel_gps", "P_eng_mech_W")):
        mdot = ts["mdot_fuel_gps"].to_numpy(float)
        PkW = np.maximum(ts["P_eng_mech_W"].to_numpy(float) / 1000.0, 1e-12)
        bsfc_impl = (mdot * 3600.0) / PkW
        bsfc_impl[~np.isfinite(bsfc_impl)] = np.nan
        ts["bsfc_implied_g_per_kWh"] = bsfc_impl

    # Engine power for coloring (optional)
    if "P_eng_mech_W" in ts.columns:
        ts["P_eng_kW"] = ts["P_eng_mech_W"].to_numpy(float) / 1000.0

    return ts


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay engine operating points on BSFC map (rpm×tq).")
    ap.add_argument("--bsfc_csv", required=True, help="BSFC grid CSV (eng_rpm × tq header)")
    ap.add_argument("--timeseries", required=True, help="Phase B timeseries CSV (canonical or derived)")
    ap.add_argument("--out", default="bsfc_operating_points.png", help="Output PNG path")
    ap.add_argument(
        "--color_points_by",
        default="none",
        choices=["none", "power_kW", "bsfc_implied"],
        help="Color scatter points by this metric (optional).",
    )
    ap.add_argument("--only_fuel_on", action="store_true", help="Plot only fuel-on points (recommended).")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    bsfc_csv = Path(args.bsfc_csv)
    ts_csv = Path(args.timeseries)
    out = Path(args.out)

    rpm_grid, tq_grid, z = load_bsfc_grid(bsfc_csv)
    ts = load_operating_points(ts_csv)

    if args.only_fuel_on:
        ts = ts.loc[ts["fuel_on"].to_numpy(bool)]

    rpm_pts = ts["eng_rpm"].to_numpy(dtype=float)
    tq_pts = ts["eng_tq_Nm"].to_numpy(dtype=float)

    # OOB mask (relative to map axes)
    oob = (rpm_pts < rpm_grid.min()) | (rpm_pts > rpm_grid.max()) | (tq_pts < tq_grid.min()) | (tq_pts > tq_grid.max())

    # Build pcolormesh edges
    def edges(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        mid = 0.5 * (x[1:] + x[:-1])
        e0 = x[0] - (mid[0] - x[0])
        eN = x[-1] + (x[-1] - mid[-1])
        return np.r_[e0, mid, eN]

    rpm_e = edges(rpm_grid)
    tq_e = edges(tq_grid)

    fig, ax = plt.subplots(figsize=(10.5, 6.3), dpi=args.dpi)

    # Heatmap (BSFC)
    pcm = ax.pcolormesh(rpm_e, tq_e, z.T, shading="auto")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("BSFC [g/kWh]")

    # Scatter points
    if args.color_points_by == "none":
        ax.scatter(rpm_pts[~oob], tq_pts[~oob], s=10, marker="o", linewidths=0.2, edgecolors="k", alpha=0.8)
    elif args.color_points_by == "power_kW":
        if "P_eng_kW" not in ts.columns:
            raise ValueError("timeseries missing P_eng_mech_W (needed for power_kW coloring)")
        c = ts["P_eng_kW"].to_numpy(float)
        sc = ax.scatter(rpm_pts[~oob], tq_pts[~oob], s=14, c=c[~oob], marker="o", alpha=0.85)
        cb2 = fig.colorbar(sc, ax=ax, pad=0.02)
        cb2.set_label("Engine mech power [kW]")
    else:  # bsfc_implied
        if "bsfc_implied_g_per_kWh" not in ts.columns:
            raise ValueError("timeseries missing mdot_fuel_gps and/or P_eng_mech_W (needed for bsfc_implied coloring)")
        c = ts["bsfc_implied_g_per_kWh"].to_numpy(float)
        sc = ax.scatter(rpm_pts[~oob], tq_pts[~oob], s=14, c=c[~oob], marker="o", alpha=0.85)
        cb2 = fig.colorbar(sc, ax=ax, pad=0.02)
        cb2.set_label("Implied BSFC [g/kWh]")

    # Mark OOB points (if any)
    if np.any(oob):
        ax.scatter(rpm_pts[oob], tq_pts[oob], s=20, marker="x", linewidths=0.8, color="red", alpha=0.9, label="OOB")

    ax.set_title("BSFC map (rpm×tq) with operating points overlay")
    ax.set_xlabel("Engine speed [rpm]")
    ax.set_ylabel("Engine torque [Nm]")
    ax.set_xlim(rpm_grid.min(), rpm_grid.max())
    ax.set_ylim(tq_grid.min(), tq_grid.max())
    ax.grid(True, alpha=0.25)

    if np.any(oob):
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    print(f"[OK] saved: {out.resolve()}")


if __name__ == "__main__":
    main()