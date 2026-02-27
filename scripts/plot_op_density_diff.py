from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pts(csv_path: Path) -> pd.DataFrame:
    ts = pd.read_csv(csv_path)

    need = ["eng_rpm", "eng_tq_Nm", "dt_s"]
    for c in need:
        if c not in ts.columns:
            raise ValueError(f"{csv_path} missing required column: {c}")

    # fuel-on mask (best effort, consistent with your audit_B logic)
    if all(c in ts.columns for c in ["mdot_fuel_gps", "P_fuel_W", "P_eng_mech_W"]):
        fuel_on = (
            (ts["mdot_fuel_gps"].to_numpy(float) > 0.0)
            & (ts["P_fuel_W"].to_numpy(float) > 1.0)
            & (ts["P_eng_mech_W"].to_numpy(float) > 1.0)
        )
    elif "mdot_fuel_gps" in ts.columns:
        fuel_on = ts["mdot_fuel_gps"].to_numpy(float) > 0.0
    else:
        fuel_on = np.ones(len(ts), dtype=bool)

    ts = ts.copy()
    ts["fuel_on"] = fuel_on

    # fuel per step [g]
    if "mdot_fuel_gps" in ts.columns:
        ts["fuel_step_g"] = ts["mdot_fuel_gps"].to_numpy(float) * ts["dt_s"].to_numpy(float)
    else:
        ts["fuel_step_g"] = 0.0

    return ts


def hist2d(pts: pd.DataFrame, rpm_bins: np.ndarray, tq_bins: np.ndarray, weight: np.ndarray) -> np.ndarray:
    H, _, _ = np.histogram2d(
        pts["eng_rpm"].to_numpy(float),
        pts["eng_tq_Nm"].to_numpy(float),
        bins=[rpm_bins, tq_bins],
        weights=weight,
    )
    return H.T  # (tq, rpm) for imshow


def main() -> None:
    ap = argparse.ArgumentParser(description="rpm×tq density diff (C - B) for engine operating points")
    ap.add_argument("--B", required=True, help="B timeseries CSV (e.g., 202200)")
    ap.add_argument("--C", required=True, help="C timeseries CSV (e.g., 201816)")
    ap.add_argument("--out", default="op_density_diff_C_minus_B.png", help="Output PNG path")
    ap.add_argument("--only_fuel_on", action="store_true", help="Use only fuel-on points")
    ap.add_argument("--rpm_min", type=float, default=1000.0)
    ap.add_argument("--rpm_max", type=float, default=5600.0)
    ap.add_argument("--rpm_step", type=float, default=100.0)
    ap.add_argument("--tq_min", type=float, default=0.0)
    ap.add_argument("--tq_max", type=float, default=125.0)
    ap.add_argument("--tq_step", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    pB = Path(args.B)
    pC = Path(args.C)
    out = Path(args.out)

    B = load_pts(pB)
    C = load_pts(pC)

    if args.only_fuel_on:
        B = B.loc[B["fuel_on"].to_numpy(bool)]
        C = C.loc[C["fuel_on"].to_numpy(bool)]

    rpm_bins = np.arange(args.rpm_min, args.rpm_max + args.rpm_step, args.rpm_step)
    tq_bins = np.arange(args.tq_min, args.tq_max + args.tq_step, args.tq_step)

    # Δ dwell time [s] = C - B
    Hb_t = hist2d(B, rpm_bins, tq_bins, weight=B["dt_s"].to_numpy(float))
    Hc_t = hist2d(C, rpm_bins, tq_bins, weight=C["dt_s"].to_numpy(float))
    D_time = Hc_t - Hb_t

    # Δ fuel [g] = C - B
    Hb_f = hist2d(B, rpm_bins, tq_bins, weight=B["fuel_step_g"].to_numpy(float))
    Hc_f = hist2d(C, rpm_bins, tq_bins, weight=C["fuel_step_g"].to_numpy(float))
    D_fuel = Hc_f - Hb_f

    # extent for axes in real units
    extent = [rpm_bins[0], rpm_bins[-1], tq_bins[0], tq_bins[-1]]

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.4), dpi=args.dpi)

    im0 = ax[0].imshow(D_time, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r")
    ax[0].set_title("Δ dwell time [s]  (C − B)")
    ax[0].set_xlabel("Engine speed [rpm]")
    ax[0].set_ylabel("Engine torque [Nm]")
    cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)
    cb0.set_label("seconds")

    im1 = ax[1].imshow(D_fuel, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r")
    ax[1].set_title("Δ fuel [g]  (C − B)")
    ax[1].set_xlabel("Engine speed [rpm]")
    ax[1].set_ylabel("Engine torque [Nm]")
    cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.03)
    cb1.set_label("grams")

    for a in ax:
        a.grid(True, alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

    # print compact summary
    fuel_B = float(B["fuel_step_g"].sum())
    fuel_C = float(C["fuel_step_g"].sum())
    dt_B = float(B["dt_s"].sum())
    dt_C = float(C["dt_s"].sum())
    print("[OK] saved:", out.resolve())
    print(f"Fuel(B)={fuel_B:.3f} g, Fuel(C)={fuel_C:.3f} g, Δ(C-B)={fuel_C-fuel_B:.3f} g")
    print(f"Dwell(B)={dt_B:.3f} s, Dwell(C)={dt_C:.3f} s, Δ(C-B)={dt_C-dt_B:.3f} s")


if __name__ == "__main__":
    main()