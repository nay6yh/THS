# viz_ths_dashboards.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath


# ===== Frozen visual spec =====
MODE_COLORS = {
    "EV": "#1f77b4",
    "HybridDrive": "#2ca02c",
    "Charge": "#ff7f0e",
    "Regen": "#9467bd",
    "FrictionBrake": "#7f7f7f",
}

PHASE_COLORS = {
    "Low": "#a6cee3",
    "Mid": "#b2df8a",
    "High": "#fdbf6f",
    "Extra-high": "#fb9a99",
}

ENERGY_COLORS = {
    "Engine_mech": "#ff7f0e",
    "Battery_discharge": "#1f77b4",
    "Wheel_regen_available": "#a6cee3",
    "Wheel_traction": "#2ca02c",
    "Battery_charge": "#9467bd",
    "Aux_load": "#d62728",
    "Friction_brake_heat": "#7f7f7f",
    "Other": "#c7c7c7",
}

NOTE_SANK = "This is an explanatory flow diagram; not a strict energy balance."

# ===== Signal colors (fix "same color" confusion) =====
SIG_COL = {
    "speed": "#000000",        # black
    "accel": "#7f7f7f",        # gray

    "wheel_req": "#000000",    # black dashed
    "wheel_deliv": "#1f77b4",  # blue
    "brake_fric": "#7f7f7f",   # gray (fill)

    "soc": "#2ca02c",          # green
    "batt_act": "#1f77b4",     # blue
    "lim_dis": "#d62728",      # red
    "lim_chg": "#9467bd",      # purple

    "eng_rpm": "#1f77b4",      # blue
    "eng_tq": "#ff7f0e",       # orange
}

def _choose_temp_x(ts: pd.DataFrame) -> Optional[str]:
    """Pick a temperature axis that actually varies. If none varies, return None."""
    if "Tbatt_C" in ts.columns:
        if float(np.nanstd(ts["Tbatt_C"].to_numpy(float))) >= 0.5:
            return "Tbatt_C"
    if "Tamb_C" in ts.columns:
        if float(np.nanstd(ts["Tamb_C"].to_numpy(float))) >= 0.5:
            return "Tamb_C"
    return None

# ===== Helpers =====
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path, dpi: int):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _as_kmh(x_mps: np.ndarray) -> np.ndarray:
    return np.asarray(x_mps, dtype=float) * 3.6


def _robust_sym_ylim(y: np.ndarray, q: float = 0.99, pad: float = 1.10, min_span: float = 1e-6) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    a = np.quantile(np.abs(y), q)
    a = max(float(a) * pad, min_span)
    return (-a, a)


def _robust_ylim(y: np.ndarray, q: float = 0.99, pad: float = 1.10) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.quantile(y, [1 - q, q])
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    mid = (hi + lo) / 2.0
    lo2 = mid - span * pad / 2.0
    hi2 = mid + span * pad / 2.0
    return (float(lo2), float(hi2))


def _phase_color(phase: str) -> str:
    if phase in PHASE_COLORS:
        return PHASE_COLORS[phase]
    for k, v in PHASE_COLORS.items():
        if str(phase).startswith(k):
            return v
    return "#dddddd"


def _shade_by_segments(ax, t: np.ndarray, labels: np.ndarray, color_fn, alpha: float = 0.10) -> None:
    if labels is None or len(labels) != len(t):
        return
    lab = labels.astype(str)
    change = np.where(lab[1:] != lab[:-1])[0] + 1
    bounds = np.r_[0, change, len(lab)]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        ax.axvspan(t[s], t[e - 1], color=color_fn(lab[s]), alpha=alpha, linewidth=0)


def _draw_mode_band(ax, t: np.ndarray, mode: np.ndarray, y0: float = 0.83, y1: float = 1.00, alpha: float = 0.25) -> None:
    if mode is None or len(mode) != len(t):
        return
    mode = mode.astype(str)
    change = np.where(mode[1:] != mode[:-1])[0] + 1
    bounds = np.r_[0, change, len(mode)]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        c = MODE_COLORS.get(mode[s], "#cccccc")
        ax.axvspan(t[s], t[e - 1], ymin=y0, ymax=y1, color=c, alpha=alpha, linewidth=0)


def _energy_MJ(P_W: np.ndarray, dt_s: np.ndarray) -> float:
    return float(np.nansum(np.asarray(P_W, float) * np.asarray(dt_s, float)) / 1e6)


def _split_pos_neg(P_W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P_W, dtype=float)
    return np.maximum(P, 0.0), np.maximum(-P, 0.0)


def _export_table_png(df: pd.DataFrame, outpath: Path, title: str, dpi: int = 160, max_rows: int = 20) -> None:
    df2 = df.head(max_rows).copy()
    fig, ax = plt.subplots(figsize=(12.5, 0.55 * (len(df2) + 2)))
    ax.axis("off")
    table = ax.table(cellText=df2.astype(str).values.tolist(),
                     colLabels=list(df2.columns),
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    ax.set_title(title)
    _savefig(outpath, dpi=dpi)


# ===== ② Energy summary =====
@dataclass
class EnergySummary:
    E_wheel_pos_MJ: float
    E_wheel_neg_MJ: float
    E_fric_MJ: float
    E_aux_MJ: float
    E_hvac_MJ: Optional[float]

    E_batt_dis_MJ: float
    E_batt_chg_MJ: float
    E_batt_net_MJ: float

    E_eng_mech_MJ: float

    # ★追加：回生由来の充電（上限=min）とエンジン充電proxy
    E_regen_to_batt_MJ: float
    E_engine_to_batt_proxy_MJ: float

    # ★修正：必ず0〜1
    regen_utilization: float
    friction_share: float

    count_flag_batt_sat: int
    count_flag_eng_sat: int

    def to_dict(self) -> Dict[str, float]:
        d = {
            "E_wheel_pos_MJ": self.E_wheel_pos_MJ,
            "E_wheel_neg_MJ": self.E_wheel_neg_MJ,
            "E_fric_MJ": self.E_fric_MJ,
            "E_aux_MJ": self.E_aux_MJ,

            "E_batt_dis_MJ": self.E_batt_dis_MJ,
            "E_batt_chg_MJ": self.E_batt_chg_MJ,
            "E_batt_net_MJ": self.E_batt_net_MJ,

            "E_eng_mech_MJ": self.E_eng_mech_MJ,

            "E_regen_to_batt_MJ": self.E_regen_to_batt_MJ,
            "E_engine_to_batt_proxy_MJ": self.E_engine_to_batt_proxy_MJ,

            "regen_utilization": self.regen_utilization,
            "friction_share": self.friction_share,

            "count_flag_batt_sat": float(self.count_flag_batt_sat),
            "count_flag_eng_sat": float(self.count_flag_eng_sat),
        }
        if self.E_hvac_MJ is not None:
            d["E_hvac_MJ"] = self.E_hvac_MJ
            d["E_aux_other_MJ"] = self.E_aux_MJ - self.E_hvac_MJ
        return d


def compute_energy_summary(ts: pd.DataFrame) -> EnergySummary:
    dt = ts["dt_s"].to_numpy(float)

    Pw_del = ts["P_wheel_deliv_W_dbg"].to_numpy(float)
    Pb = ts["P_batt_act_W"].to_numpy(float)
    Paux = ts["P_aux_W"].to_numpy(float)
    Peng = ts["P_eng_mech_W"].to_numpy(float)
    Pfric = ts["P_brake_fric_W"].to_numpy(float)

    Pw_pos = np.maximum(Pw_del, 0.0)
    Pw_neg = np.maximum(-Pw_del, 0.0)

    Pb_dis = np.maximum(Pb, 0.0)   # discharge
    Pb_chg = np.maximum(-Pb, 0.0)  # charge magnitude

    E_wheel_pos = _energy_MJ(Pw_pos, dt)
    E_wheel_neg = _energy_MJ(Pw_neg, dt)
    E_fric = _energy_MJ(Pfric, dt)
    E_aux = _energy_MJ(Paux, dt)
    E_batt_dis = _energy_MJ(Pb_dis, dt)
    E_batt_chg = _energy_MJ(Pb_chg, dt)
    E_batt_net = E_batt_dis - E_batt_chg
    E_eng_mech = _energy_MJ(np.maximum(Peng, 0.0), dt)

    E_hvac = None
    if "P_hvac_W" in ts.columns:
        E_hvac = _energy_MJ(ts["P_hvac_W"].to_numpy(float), dt)

    # ★ここが重要：回生由来の充電は wheel_neg を超えない
    E_regen_to_batt = min(E_batt_chg, E_wheel_neg)
    E_engine_to_batt_proxy = max(E_batt_chg - E_regen_to_batt, 0.0)

    eps = 1e-9
    regen_util = float(E_regen_to_batt / max(E_wheel_neg, eps))   # 0..1
    fric_share = float(E_fric / max(E_wheel_neg, eps))

    count_batt_sat = int(ts["flag_batt_sat"].sum()) if "flag_batt_sat" in ts.columns else 0
    count_eng_sat = int(ts["flag_eng_sat"].sum()) if "flag_eng_sat" in ts.columns else 0

    return EnergySummary(
        E_wheel_pos_MJ=E_wheel_pos,
        E_wheel_neg_MJ=E_wheel_neg,
        E_fric_MJ=E_fric,
        E_aux_MJ=E_aux,
        E_hvac_MJ=E_hvac,
        E_batt_dis_MJ=E_batt_dis,
        E_batt_chg_MJ=E_batt_chg,
        E_batt_net_MJ=E_batt_net,
        E_eng_mech_MJ=E_eng_mech,
        E_regen_to_batt_MJ=E_regen_to_batt,
        E_engine_to_batt_proxy_MJ=E_engine_to_batt_proxy,
        regen_utilization=regen_util,
        friction_share=fric_share,
        count_flag_batt_sat=count_batt_sat,
        count_flag_eng_sat=count_eng_sat,
    )


# ===== Custom sankey-like ribbons (explanatory) =====
def _ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.8):
    cx0 = x0 + (x1 - x0) * 0.35
    cx1 = x0 + (x1 - x0) * 0.65

    verts = [
        (x0, y0),                         # bottom left
        (cx0, y0), (cx1, y1), (x1, y1),   # bottom curve to bottom right
        (x1, y1 + h1),                    # up to top right
        (cx1, y1 + h1), (cx0, y0 + h0), (x0, y0 + h0),  # top curve back
        (x0, y0),                         # close
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]

    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)

def _fmt_mj(v: float) -> str:
    return f"{v:.2f}" if abs(v) < 0.1 else f"{v:.1f}"

def plot_energy_sankey(ax, es: EnergySummary) -> None:
    ax.set_axis_off()

    Ew_pos, Ew_neg = es.E_wheel_pos_MJ, es.E_wheel_neg_MJ
    Efric, Eaux = es.E_fric_MJ, es.E_aux_MJ
    Eb_dis, Eb_chg = es.E_batt_dis_MJ, es.E_batt_chg_MJ
    Eeng = es.E_eng_mech_MJ

    # Split (heuristic but consistent + explanatory)
    regen_to_batt = min(Eb_chg, Ew_neg)
    engine_to_batt = max(Eb_chg - regen_to_batt, 0.0)
    regen_not_stored = max(Ew_neg - regen_to_batt, 0.0)

    aux_from_batt = min(Eb_dis, Eaux)
    batt_to_wheel = max(Eb_dis - aux_from_batt, 0.0)

    wheel_need = max(Ew_pos - batt_to_wheel, 0.0)
    eng_to_wheel = min(Eeng, wheel_need)
    eng_unused = max(Eeng - eng_to_wheel - engine_to_batt, 0.0)

    xL, xR = 0.05, 0.75
    boxW, gap = 0.18, 0.04

    sources = [
        ("Engine_mech", Eeng, ENERGY_COLORS["Engine_mech"]),
        ("Battery_discharge", Eb_dis, ENERGY_COLORS["Battery_discharge"]),
        ("Wheel_regen_available", Ew_neg, ENERGY_COLORS["Wheel_regen_available"]),
    ]
    sinks = [
        ("Wheel_traction", Ew_pos, ENERGY_COLORS["Wheel_traction"]),
        ("Battery_charge", Eb_chg, ENERGY_COLORS["Battery_charge"]),
        ("Aux_load", Eaux, ENERGY_COLORS["Aux_load"]),
        ("Friction_brake_heat", Efric, ENERGY_COLORS["Friction_brake_heat"]),
    ]
    other = eng_unused + regen_not_stored
    if other > 1e-6:
        sinks.append(("Other", other, ENERGY_COLORS["Other"]))

    totalL = sum(v for _, v, _ in sources)
    totalR = sum(v for _, v, _ in sinks)
    scale = 1.0 / max(totalL, totalR, 1e-9)

    y_top = 0.92
    src_boxes, sink_boxes = {}, {}

    y = y_top
    for name, v, c in sources:
        h = v * scale * 0.8
        y -= h
        src_boxes[name] = (xL, y, boxW, h, c, v)
        y -= gap

    y = y_top
    for name, v, c in sinks:
        h = v * scale * 0.8
        y -= h
        sink_boxes[name] = (xR, y, boxW, h, c, v)
        y -= gap

    for name, (x, y, w, h, c, v) in src_boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="none", alpha=0.85))
        ax.text(x + w + 0.01, y + h / 2, f"{name}\n{_fmt_mj(v)} MJ", va="center", ha="left", fontsize=9)

    for name, (x, y, w, h, c, v) in sink_boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="none", alpha=0.85))
        ax.text(x - 0.01, y + h / 2, f"{name}\n{_fmt_mj(v)} MJ", va="center", ha="right", fontsize=9)

    src_off = {k: 0.0 for k in src_boxes}
    sink_off = {k: 0.0 for k in sink_boxes}

    def connect(src: str, dst: str, v: float, color: str) -> None:
        if v <= 0:
            return
        x0, y0, w0, _, _, _ = src_boxes[src]
        x1, y1, _, _, _, _ = sink_boxes[dst]
        hh = v * scale * 0.8
        y0b = y0 + src_off[src]
        y1b = y1 + sink_off[dst]
        _ribbon(ax, x0 + w0, y0b, hh, x1, y1b, hh, color=color, alpha=0.65)
        src_off[src] += hh
        sink_off[dst] += hh

    connect("Engine_mech", "Wheel_traction", eng_to_wheel, ENERGY_COLORS["Engine_mech"])
    connect("Engine_mech", "Battery_charge", engine_to_batt, ENERGY_COLORS["Engine_mech"])
    if "Other" in sink_boxes:
        connect("Engine_mech", "Other", eng_unused, ENERGY_COLORS["Engine_mech"])

    connect("Battery_discharge", "Aux_load", aux_from_batt, ENERGY_COLORS["Battery_discharge"])
    connect("Battery_discharge", "Wheel_traction", batt_to_wheel, ENERGY_COLORS["Battery_discharge"])

    connect("Wheel_regen_available", "Battery_charge", regen_to_batt, ENERGY_COLORS["Wheel_regen_available"])
    if "Other" in sink_boxes:
        connect("Wheel_regen_available", "Other", regen_not_stored, ENERGY_COLORS["Wheel_regen_available"])

    # friction is separate (not strictly a split of wheel_neg), shown as sink; diagram is explanatory
    ax.text(0.5, 0.02, NOTE_SANK, ha="center", va="bottom", fontsize=8)


# ===== ① Overview =====
def plot_overview(ts: pd.DataFrame, outdir: Path, dpi: int = 160) -> None:
    _ensure_dir(outdir)
    t = ts["t_s"].to_numpy(float)
    phase = ts["phase"].astype(str).to_numpy()
    mode = ts["mode"].astype(str).to_numpy()

    fig = plt.figure(figsize=(13.5, 10.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.25, 1.25, 1.25], hspace=0.12)

    # ---- A: speed/accel (fix same color)
    ax1 = fig.add_subplot(gs[0, 0])
    _shade_by_segments(ax1, t, phase, _phase_color, alpha=0.10)
    spd_kmh = _as_kmh(ts["veh_spd_mps"].to_numpy(float))
    acc = ts["veh_acc_mps2"].to_numpy(float)
    ax1.plot(t, spd_kmh, color=SIG_COL["speed"], linewidth=1.35, label="VehSpd")
    ax1.set_ylabel("Speed [km/h]")
    ax1b = ax1.twinx()
    ax1b.plot(t, acc, color=SIG_COL["accel"], linewidth=1.0, alpha=0.85, label="VehAcc")
    ax1b.set_ylabel("Accel [m/s²]")
    ax1.set_title("Overview①: WLTC / Wheel / Battery / Mode")
    h1, l1 = ax1.get_legend_handles_labels()
    h1b, l1b = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h1b, l1 + l1b, loc="upper left", fontsize=8, framealpha=0.9)

    # ---- B: wheel power (req vs deliv vs fric) clearly separated
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    _shade_by_segments(ax2, t, phase, _phase_color, alpha=0.10)
    P_req = ts["P_wheel_req_W"].to_numpy(float) / 1000.0
    P_del = ts["P_wheel_deliv_W_dbg"].to_numpy(float) / 1000.0
    P_fric = ts["P_brake_fric_W"].to_numpy(float) / 1000.0

    ax2.plot(t, P_req, color=SIG_COL["wheel_req"], linestyle="--", linewidth=1.15, label="P_wheel_req")
    ax2.plot(t, P_del, color=SIG_COL["wheel_deliv"], linewidth=1.25, label="P_wheel_deliv")
    ax2.fill_between(t, 0.0, P_fric, color=SIG_COL["brake_fric"], alpha=0.25, label="P_brake_fric")
    ax2.axhline(0, linewidth=0.8, color="#000000", alpha=0.6)
    ax2.set_ylabel("Wheel power [kW]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(_robust_sym_ylim(np.r_[P_req, P_del, P_fric]))
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- C: SOC + batt power + limits (fix same color)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    _shade_by_segments(ax3, t, phase, _phase_color, alpha=0.10)
    soc = ts["soc_pct"].to_numpy(float)
    ax3.plot(t, soc, color=SIG_COL["soc"], linewidth=1.35, label="SOC")
    ax3.set_ylabel("SOC [%]")

    ax3b = ax3.twinx()
    P_batt = ts["P_batt_act_W"].to_numpy(float) / 1000.0
    lim_dis = ts["lim_batt_discharge_W"].to_numpy(float) / 1000.0
    lim_chg = ts["lim_batt_charge_W"].to_numpy(float) / 1000.0

    ax3b.plot(t, P_batt, color=SIG_COL["batt_act"], linewidth=1.05, alpha=0.92, label="P_batt_act")
    ax3b.axhline(0, linewidth=0.8, color="#000000", alpha=0.6)
    ax3b.plot(t, +lim_dis, color=SIG_COL["lim_dis"], linewidth=0.9, linestyle="--", alpha=0.9, label="lim_dis")
    ax3b.plot(t, -lim_chg, color=SIG_COL["lim_chg"], linewidth=0.9, linestyle="--", alpha=0.9, label="lim_chg")
    ax3b.set_ylabel("P_batt_act [kW] (+dis / -chg)")

    sat = ts["flag_batt_sat"].to_numpy(int)
    idx = np.where(sat > 0)[0]
    if idx.size > 0:
        # mark saturation points clearly
        ax3b.scatter(t[idx], P_batt[idx], marker="^", s=22, c="#d62728", alpha=0.9, linewidths=0.0, label="batt_sat")

    ax3b.set_ylim(_robust_sym_ylim(np.r_[P_batt, +lim_dis, -lim_chg]))
    h3, l3 = ax3.get_legend_handles_labels()
    h3b, l3b = ax3b.get_legend_handles_labels()
    ax3.legend(h3 + h3b, l3 + l3b, loc="upper left", fontsize=8, framealpha=0.9)


    # ---- D: engine rpm/tq with mode band (fix same color)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    _shade_by_segments(ax4, t, phase, _phase_color, alpha=0.10)
    _draw_mode_band(ax4, t, mode, y0=0.83, y1=1.00, alpha=0.25)

    eng_rpm = ts["eng_rpm"].to_numpy(float)
    eng_tq = ts["eng_tq_Nm"].to_numpy(float)

    ax4.plot(t, eng_rpm, color=SIG_COL["eng_rpm"], linewidth=1.05, label="eng_rpm")
    ax4.set_ylabel("Engine speed [rpm]")

    ax4b = ax4.twinx()
    ax4b.plot(t, eng_tq, color=SIG_COL["eng_tq"], linewidth=1.0, alpha=0.9, label="eng_tq")
    ax4b.set_ylabel("Engine torque [Nm]")

    idx_sat = np.where(ts["flag_eng_sat"].to_numpy(int) > 0)[0]
    if idx_sat.size > 0:
        ax4b.scatter(t[idx_sat], eng_tq[idx_sat], marker="o", s=14, c="#d62728", alpha=0.9, linewidths=0.0, label="eng_sat")

    ax4.set_xlabel("Time [s]")
    ax4.set_ylim(_robust_ylim(eng_rpm, q=0.995, pad=1.1))
    h4, l4 = ax4.get_legend_handles_labels()
    h4b, l4b = ax4b.get_legend_handles_labels()
    ax4.legend(h4 + h4b, l4 + l4b, loc="upper left", fontsize=8, framealpha=0.9)

    _savefig(outdir / "01_overview.png", dpi=dpi)

# ===== ② Energy Flow =====
def plot_energy_flow(ts: pd.DataFrame, outdir: Path, dpi: int = 160) -> None:
    _ensure_dir(outdir)
    es = compute_energy_summary(ts)
    with open(outdir / "energy_summary.json", "w", encoding="utf-8") as f:
        json.dump(es.to_dict(), f, indent=2)

    fig = plt.figure(figsize=(14.0, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.1, 0.9], hspace=0.18, wspace=0.12)

    axS = fig.add_subplot(gs[:, 0])
    plot_energy_sankey(axS, es)
    axS.set_title("Energy Flow②: Explanatory Sankey (MJ)")

    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 1])

    dt = ts["dt_s"].to_numpy(float)
    Pw_del = ts["P_wheel_deliv_W_dbg"].to_numpy(float)
    Pw_pos = np.maximum(Pw_del, 0.0)
    Pw_neg = np.maximum(-Pw_del, 0.0)
    Pfric = ts["P_brake_fric_W"].to_numpy(float)
    Paux = ts["P_aux_W"].to_numpy(float)
    Pb = ts["P_batt_act_W"].to_numpy(float)
    Pb_dis = np.maximum(Pb, 0.0)
    Pb_chg = np.maximum(-Pb, 0.0)
    Peng = np.maximum(ts["P_eng_mech_W"].to_numpy(float), 0.0)

    df = pd.DataFrame({
        "phase": ts["phase"].astype(str),
        "E_wheel_pos_MJ": Pw_pos * dt / 1e6,
        "E_wheel_neg_MJ": Pw_neg * dt / 1e6,
        "E_fric_MJ": Pfric * dt / 1e6,
        "E_aux_MJ": Paux * dt / 1e6,
        "E_batt_dis_MJ": Pb_dis * dt / 1e6,
        "E_batt_chg_MJ": Pb_chg * dt / 1e6,
        "E_eng_mech_MJ": Peng * dt / 1e6,
    })
    grp = df.groupby("phase").sum(numeric_only=True)
    phases = grp.index.tolist()
    x = np.arange(len(phases))

    bottom = np.zeros(len(phases))
    for key, label, color in [
        ("E_wheel_pos_MJ", "Wheel traction", ENERGY_COLORS["Wheel_traction"]),
        ("E_aux_MJ", "Aux", ENERGY_COLORS["Aux_load"]),
        ("E_fric_MJ", "Friction heat", ENERGY_COLORS["Friction_brake_heat"]),
    ]:
        vals = grp[key].to_numpy(float)
        axC.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.9)
        bottom += vals
    axC.set_xticks(x)
    axC.set_xticklabels(phases, rotation=20)
    axC.set_ylabel("Energy [MJ]")
    axC.set_title("Phase breakdown: Consumption/Loss (MJ)")
    axC.legend(fontsize=8)

    bottom = np.zeros(len(phases))
    for key, label, color in [
        ("E_eng_mech_MJ", "Engine mech", ENERGY_COLORS["Engine_mech"]),
        ("E_batt_dis_MJ", "Batt discharge", ENERGY_COLORS["Battery_discharge"]),
        ("E_batt_chg_MJ", "Batt charge", ENERGY_COLORS["Battery_charge"]),
    ]:
        vals = grp[key].to_numpy(float)
        axD.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.9)
        bottom += vals
    axD.set_xticks(x)
    axD.set_xticklabels(phases, rotation=20)
    axD.set_ylabel("Energy [MJ]")
    axD.set_title("Phase breakdown: Supply (MJ)")
    axD.legend(fontsize=8)

    _savefig(outdir / "02_energy_flow.png", dpi=dpi)

    # KPI table
    rows = []
    d = es.to_dict()
    for k in [
        "E_wheel_pos_MJ","E_wheel_neg_MJ","E_fric_MJ","E_aux_MJ",
        "E_batt_dis_MJ","E_batt_chg_MJ","E_batt_net_MJ","E_eng_mech_MJ",
        "E_regen_to_batt_MJ","E_engine_to_batt_proxy_MJ",   # ★追加
        "regen_utilization","friction_share",
        "count_flag_batt_sat","count_flag_eng_sat",
        "E_hvac_MJ","E_aux_other_MJ"
    ]:
        if k not in d:
            continue
        v = d[k]
        if "utilization" in k or "share" in k:
            rows.append([k, f"{float(v):.3f}"])
        elif "count" in k:
            rows.append([k, f"{int(v):d}"])
        else:
            fv = float(v)
            rows.append([k, f"{fv:.2f}" if abs(fv) < 0.1 else f"{fv:.1f}"])

    fig2, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["KPI", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    ax.set_title("Energy Flow KPIs (MJ / ratios / counts)")
    _savefig(outdir / "02_energy_kpis.png", dpi=dpi)


# ===== ③ Operating Points =====
def plot_operating_points(ts: pd.DataFrame, outdir: Path, dpi: int = 160) -> None:
    _ensure_dir(outdir)
    mode = ts["mode"].astype(str).to_numpy()

    eng_rpm = ts["eng_rpm"].to_numpy(float)
    eng_tq = ts["eng_tq_Nm"].to_numpy(float)
    P_eng_kw = ts["P_eng_mech_W"].to_numpy(float) / 1000.0

    mg2_rpm = ts["mg2_rpm"].to_numpy(float)
    mg2_tq = ts["mg2_tq_Nm"].to_numpy(float)
    P_mg2_kw = ts["P_mg2_elec_W"].to_numpy(float) / 1000.0

    mg1_rpm = ts["mg1_rpm"].to_numpy(float)
    mg1_tq = ts["mg1_tq_Nm"].to_numpy(float)
    P_mg1_kw = ts["P_mg1_elec_W"].to_numpy(float) / 1000.0

    clim_eng = _robust_ylim(P_eng_kw, q=0.99, pad=1.1)
    mmax = np.quantile(np.abs(np.r_[P_mg1_kw, P_mg2_kw]), 0.99) * 1.1
    mmax = max(float(mmax), 1e-6)
    clim_mg = (-mmax, mmax)

    fig = plt.figure(figsize=(14.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.16)

    axE = fig.add_subplot(gs[0, 0])
    sc = axE.scatter(eng_rpm, eng_tq, c=P_eng_kw, s=10, alpha=0.35, vmin=clim_eng[0], vmax=clim_eng[1])
    axE.set_xlabel("eng_rpm [rpm]")
    axE.set_ylabel("eng_tq [Nm]")
    axE.set_title("③-A Engine Operating Points (color=P_eng_mech [kW])")
    cb = plt.colorbar(sc, ax=axE)
    cb.set_label("P_eng_mech [kW]")

    idx = np.where(ts["flag_eng_sat"].to_numpy(int) > 0)[0]
    if idx.size > 0:
        axE.scatter(eng_rpm[idx], eng_tq[idx], s=18, facecolors="none", edgecolors="red", linewidths=0.9, alpha=0.9)

    idxc = np.where(mode == "Charge")[0]
    if idxc.size > 0:
        axE.scatter(eng_rpm[idxc], eng_tq[idxc], s=16, facecolors="none",
                    edgecolors=MODE_COLORS["Charge"], linewidths=0.8, alpha=0.7)

    axM2 = fig.add_subplot(gs[0, 1])
    sc2 = axM2.scatter(mg2_rpm, mg2_tq, c=P_mg2_kw, s=10, alpha=0.35, vmin=clim_mg[0], vmax=clim_mg[1])
    axM2.axhline(0, linewidth=0.8)
    axM2.set_xlabel("mg2_rpm [rpm]")
    axM2.set_ylabel("mg2_tq [Nm]")
    axM2.set_title("③-B MG2 Operating Points (color=P_mg2_elec [kW])")
    cb2 = plt.colorbar(sc2, ax=axM2)
    cb2.set_label("P_mg2_elec [kW] (+consume / -generate)")

    idx = np.where(ts["flag_batt_sat"].to_numpy(int) > 0)[0]
    if idx.size > 0:
        axM2.scatter(mg2_rpm[idx], mg2_tq[idx], s=24, marker="^", c="red", alpha=0.85, linewidths=0.0)

    idxf = np.where(ts["P_brake_fric_W"].to_numpy(float) > 0.0)[0]
    if idxf.size > 0:
        axM2.scatter(mg2_rpm[idxf], mg2_tq[idxf], s=12, c="#7f7f7f", alpha=0.25, linewidths=0.0)

    axM1 = fig.add_subplot(gs[1, 0])
    sc3 = axM1.scatter(mg1_rpm, mg1_tq, c=P_mg1_kw, s=10, alpha=0.35, vmin=clim_mg[0], vmax=clim_mg[1])
    axM1.axhline(0, linewidth=0.8)
    axM1.set_xlabel("mg1_rpm [rpm]")
    axM1.set_ylabel("mg1_tq [Nm]")
    axM1.set_title("③-C MG1 Operating Points (color=P_mg1_elec [kW])")
    cb3 = plt.colorbar(sc3, ax=axM1)
    cb3.set_label("P_mg1_elec [kW] (+consume / -generate)")

    axT = fig.add_subplot(gs[1, 1])

    xcol = _choose_temp_x(ts)
    if xcol is None:
        axT.axis("off")
        axT.text(0.5, 0.5,
                 "③-D disabled: Tbatt/Tamb is (almost) constant in this run",
                 ha="center", va="center", fontsize=10)
    else:
        xT = ts[xcol].to_numpy(float)

        # color by mode to avoid "single color" confusion
        colors = [MODE_COLORS.get(m, "#cccccc") for m in mode]
        axT.scatter(xT, P_mg2_kw, s=10, alpha=0.30, c=colors, linewidths=0.0)

        axT.set_xlabel(f"{xcol.replace('_C','')} [°C]")
        axT.set_ylabel("P_mg2_elec [kW] (negative=regen)")
        axT.set_title("③-D Temp vs MG2 electrical power (color=mode)")

        idx_sat = np.where(ts["flag_batt_sat"].to_numpy(int) > 0)[0]
        if idx_sat.size > 0:
            axT.scatter(xT[idx_sat], P_mg2_kw[idx_sat], s=24, marker="^", c="#d62728",
                        alpha=0.9, linewidths=0.0)

    mode_handles = [
        Line2D([0],[0], marker='o', color='w', label=m, markerfacecolor=c, markersize=7)
        for m, c in MODE_COLORS.items()
    ]
    fig.legend(handles=mode_handles, loc="upper right", fontsize=8, framealpha=0.9)

    _savefig(outdir / "03_operating_points.png", dpi=dpi)

    # event tables
    if "flag_batt_sat" in ts.columns:
        df = ts.loc[ts["flag_batt_sat"] == 1, [
            "t_s","phase","mode","Tbatt_C","veh_spd_mps",
            "P_batt_req_W","P_batt_act_W","lim_batt_charge_W","lim_batt_discharge_W",
            "mg2_tq_Nm","P_brake_fric_W"
        ]].copy()
        if len(df) > 0:
            df["dP_batt_kW"] = (df["P_batt_req_W"] - df["P_batt_act_W"]) / 1000.0
            df = df.sort_values("dP_batt_kW", ascending=False)
            df.to_csv(outdir / "03_events_batt_sat_top20.csv", index=False)
            _export_table_png(df.head(20), outdir / "03_events_batt_sat_top20.png",
                              "Top batt_sat events (sorted by ΔP_batt_kW)", dpi=dpi, max_rows=20)

    if "flag_eng_sat" in ts.columns:
        df = ts.loc[ts["flag_eng_sat"] == 1, [
            "t_s","phase","mode","veh_spd_mps","eng_rpm","eng_tq_Nm",
            "P_eng_mech_W","P_batt_act_W","soc_pct"
        ]].copy()
        if len(df) > 0:
            df = df.sort_values("eng_tq_Nm", ascending=False)
            df.to_csv(outdir / "03_events_eng_sat_top20.csv", index=False)
            _export_table_png(df.head(20), outdir / "03_events_eng_sat_top20.png",
                              "Top eng_sat events (sorted by eng_tq)", dpi=dpi, max_rows=20)


# ===== ④ Constraint Atlas =====
def plot_constraint_atlas(ts: pd.DataFrame, outdir: Path, dpi: int = 160) -> None:
    _ensure_dir(outdir)
    ts = ts.copy()

    # signed & magnitude
    ts["dP_batt_kW_signed"] = (ts["P_batt_req_W"] - ts["P_batt_act_W"]) / 1000.0
    ts["cut_batt_kW"] = np.abs(ts["dP_batt_kW_signed"])

    es = compute_energy_summary(ts)

    # KPI: use batt_sat-only distribution (avoid p95=0)
    cut_sat = ts.loc[ts["flag_batt_sat"] == 1, "cut_batt_kW"].to_numpy(float)
    kpi = {
        "count_flag_batt_sat": int(ts["flag_batt_sat"].sum()),
        "count_flag_eng_sat": int(ts["flag_eng_sat"].sum()),
        "E_fric_MJ": es.E_fric_MJ,
        "E_wheel_neg_MJ": es.E_wheel_neg_MJ,
        "friction_share": es.friction_share,
        "max_cut_batt_kW_sat": float(np.nanmax(cut_sat)) if cut_sat.size > 0 else 0.0,
        "p95_cut_batt_kW_sat": float(np.nanquantile(cut_sat, 0.95)) if cut_sat.size > 0 else 0.0,
    }
    with open(outdir / "04_constraint_kpis.json", "w", encoding="utf-8") as f:
        json.dump(kpi, f, indent=2)

    # ---- A: batt_sat events (CSV full, PNG compact)
    dfA_full = ts.loc[ts["flag_batt_sat"] == 1, [
        "t_s","phase","mode","veh_spd_mps","Tbatt_C","Tamb_C","soc_pct",
        "P_batt_req_W","P_batt_act_W","lim_batt_charge_W","lim_batt_discharge_W","dP_batt_kW_signed","cut_batt_kW",
        "P_wheel_req_W","P_wheel_deliv_W_dbg","P_brake_fric_W",
        "eng_rpm","eng_tq_Nm","mg2_tq_Nm"
    ]].copy()
    dfA_full = dfA_full[[c for c in dfA_full.columns if c in ts.columns or c in ["dP_batt_kW_signed","cut_batt_kW"]]]

    if len(dfA_full) > 0:
        dfA_full = dfA_full.sort_values("cut_batt_kW", ascending=False)
        dfA_full.to_csv(outdir / "04_batt_sat_events_top20.csv", index=False)

        # PNG: compact + rounded + kW
        dfA = dfA_full.head(20).copy()
        dfA["veh_spd_kmh"] = dfA["veh_spd_mps"] * 3.6
        dfA["P_batt_req_kW"] = dfA["P_batt_req_W"] / 1000.0
        dfA["P_batt_act_kW"] = dfA["P_batt_act_W"] / 1000.0
        dfA["lim_chg_kW"] = dfA["lim_batt_charge_W"] / 1000.0
        dfA["lim_dis_kW"] = dfA["lim_batt_discharge_W"] / 1000.0
        dfA["P_fric_kW"] = dfA["P_brake_fric_W"] / 1000.0

        keep = ["t_s","phase","mode","Tbatt_C","veh_spd_kmh",
                "P_batt_req_kW","P_batt_act_kW","lim_chg_kW","lim_dis_kW",
                "cut_batt_kW","P_fric_kW","mg2_tq_Nm"]
        keep = [c for c in keep if c in dfA.columns]
        dfA = dfA[keep].round({
            "t_s":0, "Tbatt_C":1, "veh_spd_kmh":1,
            "P_batt_req_kW":1,"P_batt_act_kW":1,"lim_chg_kW":1,"lim_dis_kW":1,
            "cut_batt_kW":2,"P_fric_kW":2,"mg2_tq_Nm":1
        })
        _export_table_png(dfA, outdir / "04_batt_sat_events_top20.png",
                          "④-A batt_sat events (Top 20 by cut |P_req-P_act| [kW])", dpi=dpi, max_rows=20)

    # ---- B: eng_sat events (CSV full, PNG compact)
    dfB_full = ts.loc[ts["flag_eng_sat"] == 1, [
        "t_s","phase","mode","veh_spd_mps","soc_pct",
        "eng_rpm","eng_tq_Nm","P_eng_mech_W",
        "P_batt_act_W","P_wheel_req_W","P_wheel_deliv_W_dbg"
    ]].copy()
    if len(dfB_full) > 0:
        dfB_full = dfB_full.sort_values("eng_tq_Nm", ascending=False)
        dfB_full.to_csv(outdir / "04_eng_sat_events_top20.csv", index=False)

        dfB = dfB_full.head(20).copy()
        dfB["veh_spd_kmh"] = dfB["veh_spd_mps"] * 3.6
        dfB["P_eng_kW"] = dfB["P_eng_mech_W"] / 1000.0
        dfB["P_batt_kW"] = dfB["P_batt_act_W"] / 1000.0
        dfB["P_wheel_req_kW"] = dfB["P_wheel_req_W"] / 1000.0
        dfB["P_wheel_del_kW"] = dfB["P_wheel_deliv_W_dbg"] / 1000.0

        keep = ["t_s","phase","mode","veh_spd_kmh","eng_rpm","eng_tq_Nm","P_eng_kW","P_batt_kW","soc_pct"]
        keep = [c for c in keep if c in dfB.columns]
        dfB = dfB[keep].round({"t_s":0,"veh_spd_kmh":1,"eng_rpm":0,"eng_tq_Nm":1,"P_eng_kW":1,"P_batt_kW":1,"soc_pct":1})
        _export_table_png(dfB, outdir / "04_eng_sat_events_top20.png",
                          "④-B eng_sat events (Top 20 by eng_tq_Nm)", dpi=dpi, max_rows=20)

    # ---- C/D: scatter plots
    fig = plt.figure(figsize=(14.0, 7.6))
    gs = fig.add_gridspec(1, 2, wspace=0.18)

    # C: batt sat map
    axC = fig.add_subplot(gs[0, 0])
    xcol = _choose_temp_x(ts)
    if xcol is None:
        axC.axis("off")
        axC.text(0.5, 0.5, "④-C disabled: Tbatt/Tamb is (almost) constant", ha="center", va="center", fontsize=10)
    else:
        xT = ts[xcol].to_numpy(float)
        y = ts["cut_batt_kW"].to_numpy(float)

        size = np.clip(ts["P_brake_fric_W"].to_numpy(float) / 1000.0, 0, None) * 10.0 + 10.0
        m = ts["mode"].astype(str).to_numpy()
        colors = [MODE_COLORS.get(mm, "#cccccc") for mm in m]

        axC.scatter(xT, y, s=size, c=colors, alpha=0.40, linewidths=0.0)
        axC.axhline(0, linewidth=0.8, color="#000000", alpha=0.6)
        axC.set_xlabel(f"{xcol.replace('_C','')} [°C]")
        axC.set_ylabel("cut |P_req - P_act| [kW]")
        axC.set_title("④-C batt saturation strength (color=mode, size=friction kW)")

        idx_reg = np.where(m == "Regen")[0]
        if idx_reg.size > 0:
            axC.scatter(xT[idx_reg], y[idx_reg], s=size[idx_reg], facecolors="none",
                        edgecolors="#000000", marker="^", alpha=0.25, linewidths=0.6)

    # D: friction map
    axD = fig.add_subplot(gs[0, 1])
    idx = np.where(ts["P_brake_fric_W"].to_numpy(float) > 0.0)[0]
    spd = ts["veh_spd_mps"].to_numpy(float)[idx] * 3.6
    fr = ts["P_brake_fric_W"].to_numpy(float)[idx] / 1000.0
    sat = ts["flag_batt_sat"].to_numpy(int)[idx]
    c2 = np.where(sat > 0, "#d62728", "#1f77b4")  # red vs blue
    axD.scatter(spd, fr, s=14, c=c2, alpha=0.45, linewidths=0.0)
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='batt_sat=1',
            markerfacecolor="#d62728", markersize=7),
        Line2D([0], [0], marker='o', color='w', label='batt_sat=0',
            markerfacecolor="#1f77b4", markersize=7),
    ]
    axD.legend(handles=legend_elems, loc="upper left", fontsize=8, framealpha=0.9)
    axD.set_xlabel("Speed [km/h]")
    axD.set_ylabel("P_brake_fric [kW]")
    axD.set_title(f"④-D friction map (n={len(idx)}; color=batt_sat 0/1)")
    if len(idx) > 0:
        axD.set_ylim(_robust_ylim(fr, q=0.99, pad=1.2))

    _savefig(outdir / "04_constraint_atlas.png", dpi=dpi)


# ===== ⑤ Comparison (B-A) =====
def plot_comparison(tsA: pd.DataFrame, tsB: pd.DataFrame, outdir: Path, labelA: str, labelB: str, dpi: int = 160) -> None:
    _ensure_dir(outdir)

    dfA = tsA.copy()
    dfB = tsB.copy()
    dfA["t_key"] = dfA["t_s"].round(3)
    dfB["t_key"] = dfB["t_s"].round(3)
    df = dfA.merge(dfB, on="t_key", suffixes=("_A", "_B"), how="inner")
    t = df["t_s_A"].to_numpy(float)

    reqA = df["P_wheel_req_W_A"].to_numpy(float)
    reqB = df["P_wheel_req_W_B"].to_numpy(float)
    req_err = float(np.nanmax(np.abs(reqA - reqB))) if np.isfinite(reqA).any() else 0.0
    with open(outdir / "05_compare_checks.json", "w", encoding="utf-8") as f:
        json.dump({"max_abs_P_wheel_req_diff_W": req_err}, f, indent=2)

    def delta(col: str) -> np.ndarray:
        return (df[f"{col}_B"].to_numpy(float) - df[f"{col}_A"].to_numpy(float)) / 1000.0

    dP_hvac = delta("P_hvac_W") if "P_hvac_W_A" in df.columns else delta("P_aux_W")
    dP_fric = delta("P_brake_fric_W")
    dP_batt = delta("P_batt_act_W")
    dP_eng = delta("P_eng_mech_W")

    phase = df["phase_A"].astype(str).to_numpy() if "phase_A" in df.columns else None

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 1, hspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    if phase is not None:
        _shade_by_segments(ax1, t, phase, _phase_color, alpha=0.10)
    ax1.plot(t, dP_hvac, linewidth=1.0, label="ΔP_hvac" if "P_hvac_W_A" in df.columns else "ΔP_aux")
    ax1.plot(t, dP_fric, linewidth=1.0, label="ΔP_brake_fric")
    ax1.plot(t, dP_batt, linewidth=1.0, label="ΔP_batt_act")
    ax1.plot(t, dP_eng, linewidth=1.0, label="ΔP_eng_mech")
    ax1.axhline(0, linewidth=0.8)
    ax1.set_ylabel("ΔPower (B-A) [kW]")
    ax1.set_title(f"⑤ Comparison (B−A): {labelB} − {labelA}")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_ylim(_robust_sym_ylim(np.r_[dP_hvac, dP_fric, dP_batt, dP_eng]))

    if "flag_batt_sat_A" in df.columns and "flag_batt_sat_B" in df.columns:
        idxB = np.where(df["flag_batt_sat_B"].to_numpy(int) > 0)[0]
        idxA = np.where(df["flag_batt_sat_A"].to_numpy(int) > 0)[0]
        if idxB.size > 0:
            ax1.scatter(t[idxB], np.zeros_like(idxB), marker="^", s=20, c="red", alpha=0.8)
        if idxA.size > 0:
            ax1.scatter(t[idxA], np.zeros_like(idxA), marker="^", s=20, c="blue", alpha=0.8)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if phase is not None:
        _shade_by_segments(ax2, t, phase, _phase_color, alpha=0.10)
    ax2.plot(t, reqA / 1000.0, linewidth=0.8, alpha=0.6, label="P_wheel_req_A")
    ax2.plot(t, reqB / 1000.0, linewidth=0.8, alpha=0.6, label="P_wheel_req_B")
    dP_wheel_del = (df["P_wheel_deliv_W_dbg_B"].to_numpy(float) - df["P_wheel_deliv_W_dbg_A"].to_numpy(float)) / 1000.0
    ax2.plot(t, dP_wheel_del, linewidth=1.1, label="ΔP_wheel_deliv")
    ax2.fill_between(t, 0.0, dP_fric, alpha=0.20, label="ΔP_brake_fric")
    ax2.axhline(0, linewidth=0.8)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("[kW]")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_ylim(_robust_sym_ylim(np.r_[reqA / 1000.0, reqB / 1000.0, dP_wheel_del, dP_fric]))

    _savefig(outdir / "05_compare_timeseries.png", dpi=dpi)

    esA = compute_energy_summary(tsA)
    esB = compute_energy_summary(tsB)
    dA, dB = esA.to_dict(), esB.to_dict()
    keys = sorted(set(dA.keys()) | set(dB.keys()))
    deltaE = {k: float(dB.get(k, 0.0) - dA.get(k, 0.0)) for k in keys}
    with open(outdir / "05_compare_energy_delta.json", "w", encoding="utf-8") as f:
        json.dump(deltaE, f, indent=2)

    rows = []
    for k in [
        "E_wheel_pos_MJ","E_wheel_neg_MJ","E_fric_MJ","E_aux_MJ",
        "E_batt_dis_MJ","E_batt_chg_MJ","E_batt_net_MJ","E_eng_mech_MJ",
        "regen_utilization","friction_share","count_flag_batt_sat","count_flag_eng_sat"
    ]:
        if k not in deltaE:
            continue
        v = deltaE[k]
        if "utilization" in k or "share" in k:
            rows.append([f"Δ{k}", f"{v:.3f}"])
        elif "count" in k:
            rows.append([f"Δ{k}", f"{int(v):d}"])
        else:
            rows.append([f"Δ{k}", f"{v:.1f}"])

    figT, axT = plt.subplots(figsize=(8.2, 5.3))
    axT.axis("off")
    table = axT.table(cellText=rows, colLabels=["KPI", "B-A"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    axT.set_title("⑤ ΔKPI table (MJ / ratios / counts)")
    _savefig(outdir / "05_compare_kpis.png", dpi=dpi)

    # Event diffs
    if "flag_batt_sat_A" in df.columns and "flag_batt_sat_B" in df.columns:
        sel = df[(df["flag_batt_sat_B"] == 1) & (df["flag_batt_sat_A"] == 0)].copy()
        if len(sel) > 0:
            sel["dP_batt_kW"] = (sel["P_batt_req_W_B"] - sel["P_batt_act_W_B"]) / 1000.0
            cols = ["t_s_B","phase_B","mode_B","veh_spd_mps_B","Tbatt_C_B",
                    "P_batt_req_W_B","P_batt_act_W_B","lim_batt_charge_W_B","lim_batt_discharge_W_B",
                    "dP_batt_kW","P_brake_fric_W_B","mg2_tq_Nm_B"]
            cols = [c for c in cols if c in sel.columns]
            sel = sel.sort_values("dP_batt_kW", ascending=False)[cols]
            sel.to_csv(outdir / "05_events_batt_sat_Bonly_top20.csv", index=False)
            df_png = df.head(20).copy()
            df_png["veh_spd_kmh"] = df_png["veh_spd_mps"] * 3.6
            df_png["P_req_kW"] = df_png["P_batt_req_W"] / 1000.0
            df_png["P_act_kW"] = df_png["P_batt_act_W"] / 1000.0
            df_png["lim_chg_kW"] = df_png["lim_batt_charge_W"] / 1000.0
            df_png["lim_dis_kW"] = df_png["lim_batt_discharge_W"] / 1000.0
            df_png["P_fric_kW"] = df_png["P_brake_fric_W"] / 1000.0
            df_png["cut_kW"] = np.abs((df_png["P_batt_req_W"] - df_png["P_batt_act_W"]) / 1000.0)

            keep = ["t_s","phase","mode","Tbatt_C","veh_spd_kmh","P_req_kW","P_act_kW","lim_chg_kW","lim_dis_kW","cut_kW","P_fric_kW","mg2_tq_Nm"]
            df_png = df_png[keep].round({
                "t_s":0,"Tbatt_C":1,"veh_spd_kmh":1,
                "P_req_kW":1,"P_act_kW":1,"lim_chg_kW":1,"lim_dis_kW":1,
                "cut_kW":2,"P_fric_kW":2,"mg2_tq_Nm":1
            })

            _export_table_png(df_png, outdir / "03_events_batt_sat_top20.png",
                            "Top batt_sat events (sorted by cut |P_req-P_act| [kW])",
                            dpi=dpi, max_rows=20)

    dfr = (df["P_brake_fric_W_B"].to_numpy(float) - df["P_brake_fric_W_A"].to_numpy(float)) / 1000.0
    df2 = df.copy()
    df2["dP_brake_fric_kW"] = dfr
    sel = df2[df2["dP_brake_fric_kW"] > 1.0].copy()
    if len(sel) > 0:
        cols = ["t_s_B","phase_B","mode_A","mode_B","veh_spd_mps_B",
                "P_brake_fric_W_A","P_brake_fric_W_B","dP_brake_fric_kW"]
        cols = [c for c in cols if c in sel.columns]
        sel = sel.sort_values("dP_brake_fric_kW", ascending=False)[cols]
        sel.to_csv(outdir / "05_events_friction_delta_top20.csv", index=False)
        _export_table_png(sel.head(20), outdir / "05_events_friction_delta_top20.png",
                          "⑤-E2 friction delta events (ΔP_brake_fric_kW > 1kW)", dpi=dpi, max_rows=20)


# ===== Orchestrator =====
def run_all(timeseries: str, outdir: str, dpi: int = 160,
            timeseries_B: Optional[str] = None,
            labelA: str = "RunA", labelB: str = "RunB") -> None:
    out = Path(outdir)
    _ensure_dir(out)

    ts = pd.read_csv(timeseries)
    plot_overview(ts, out / "01_overview", dpi=dpi)
    plot_energy_flow(ts, out / "02_energy_flow", dpi=dpi)
    plot_operating_points(ts, out / "03_operating_points", dpi=dpi)
    plot_constraint_atlas(ts, out / "04_constraint_atlas", dpi=dpi)

    if timeseries_B is not None:
        tsB = pd.read_csv(timeseries_B)
        plot_comparison(ts, tsB, out / "05_comparison", labelA=labelA, labelB=labelB, dpi=dpi)

    print(f"[OK] Saved dashboards to: {out.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="THS dashboards (frozen 5-panel spec)")
    ap.add_argument("--timeseries", required=True, help="Path to timeseries.csv")
    ap.add_argument("--outdir", default="plots_dashboards", help="Output directory")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--timeseries_B", default=None, help="Optional timeseries.csv for RunB (enables comparison view)")
    ap.add_argument("--labelA", default="RunA")
    ap.add_argument("--labelB", default="RunB")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run_all(args.timeseries, args.outdir, dpi=args.dpi,
            timeseries_B=args.timeseries_B,
            labelA=args.labelA, labelB=args.labelB)


if __name__ == "__main__":
    main()