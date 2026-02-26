# viz_ths_dashboards_phaseB.py
from __future__ import annotations

"""
Phase-B aware dashboard generator for THS Zero Simulator.

What this fixes vs the Phase-A-oriented visualizer:
- Regen utilization and regen-to-batt energy use *explicit Phase B split columns*:
    P_batt_chg_from_regen_W, P_batt_chg_from_engine_W
  instead of the Phase A heuristic min(E_batt_chg, E_wheel_neg).

- Sankey uses the same split numbers as the KPI table (no re-derivation).
- Keeps Phase-A fallback if Phase-B columns are missing (useful for A/B compare).

Usage:
  python viz_ths_dashboards_phaseB.py --timeseries out_phaseB/timeseries_phaseB_*.csv --outdir plots_dashboards_B
  (optional) --timeseries_B for comparison plot (B-A)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
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
    # ---- Phase B loss-budget (②-B) ----
    "Fuel_energy": "#8c564b",        # brown (source)
    "Engine_loss": "#f2b6a0",        # peach (heat)
    "MG_loss": "#17becf",            # cyan
    "INV_loss": "#9edae5",           # light cyan
    "Battery_loss": "#bcbd22",       # yellow-green
    "Delta_batt_storage": "#9467bd", # purple (ΔSOC)
    "Residual": "#bdbdbd",           # light gray
}

NOTE_SANK = "This is an explanatory flow diagram; not a strict energy balance."
# ---- Phase B loss budget requires these columns ----
REQUIRED_BUDGET_COLS = {
    "dt_s",
    "P_fuel_W",
    "P_wheel_deliv_W_dbg",
    "P_aux_W",
    "loss_engine_W",
    "loss_mg1_W",
    "loss_mg2_W",
    "loss_inv_W",
    "loss_batt_W",
    "E_batt_Wh",
}
# ===== Signal colors =====
SIG_COL = {
    "speed": "#000000",
    "accel": "#7f7f7f",
    "wheel_req": "#000000",
    "wheel_deliv": "#1f77b4",
    "brake_fric": "#7f7f7f",
    "soc": "#2ca02c",
    "batt_act": "#1f77b4",
    "lim_dis": "#d62728",
    "lim_chg": "#9467bd",
    "eng_rpm": "#1f77b4",
    "eng_tq": "#ff7f0e",
}


def _choose_temp_x(ts: pd.DataFrame) -> Optional[str]:
    """Pick a temperature axis that actually varies. If none varies, return None."""
    if "Tbatt_C" in ts.columns and float(np.nanstd(ts["Tbatt_C"].to_numpy(float))) >= 0.5:
        return "Tbatt_C"
    if "Tamb_C" in ts.columns and float(np.nanstd(ts["Tamb_C"].to_numpy(float))) >= 0.5:
        return "Tamb_C"
    return None


# ===== Helpers =====
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path, dpi: int, pad_inches: float = 0.15):
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
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

def compute_loss_budget(ts: pd.DataFrame) -> dict:
    dt = ts["dt_s"].to_numpy(float)

    def E_from_P(col: str) -> float:
        return float((ts[col].to_numpy(float) * dt).sum() / 1e6)

    # --- sources ---
    E_fuel_MJ = E_from_P("P_fuel_W")

    Pw = ts["P_wheel_deliv_W_dbg"].to_numpy(float)
    E_wheel_pos_MJ = float((np.clip(Pw, 0.0, None) * dt).sum() / 1e6)
    E_wheel_neg_MJ = float((np.clip(-Pw, 0.0, None) * dt).sum() / 1e6)  # ★ source #2

    # --- sinks / terms ---
    E_aux_MJ = E_from_P("P_aux_W")

    if "P_brake_fric_W" in ts.columns:
        E_fric_MJ = float((np.clip(ts["P_brake_fric_W"].to_numpy(float), 0.0, None) * dt).sum() / 1e6)
    else:
        E_fric_MJ = 0.0

    E_loss_engine_MJ = E_from_P("loss_engine_W")
    E_loss_mg_MJ = float(((ts["loss_mg1_W"] + ts["loss_mg2_W"]).to_numpy(float) * dt).sum() / 1e6)
    E_loss_inv_MJ = E_from_P("loss_inv_W")
    E_loss_batt_MJ = E_from_P("loss_batt_W")

    # ΔBatt storage from E_batt_Wh (end - start)
    E0_Wh = float(ts["E_batt_Wh"].iloc[0])
    E1_Wh = float(ts["E_batt_Wh"].iloc[-1])
    dE_batt_storage_MJ = (E1_Wh - E0_Wh) * 3600.0 / 1e6  # Wh -> J -> MJ

    # --- residual (should approach ~0) ---
    # Balance: Fuel + Wheel_neg_source ≈ Wheel_pos + Aux + Fric + Losses + Δstorage + Residual
    sinks_sum = (
        E_wheel_pos_MJ
        + E_aux_MJ
        + E_fric_MJ
        + (E_loss_engine_MJ + E_loss_mg_MJ + E_loss_inv_MJ + E_loss_batt_MJ)
        + dE_batt_storage_MJ
    )
    sources_sum = E_fuel_MJ + E_wheel_neg_MJ
    E_residual_MJ = sources_sum - sinks_sum

    return dict(
        # sources
        E_fuel_MJ=E_fuel_MJ,
        E_wheel_neg_source_MJ=E_wheel_neg_MJ,

        # sinks
        E_wheel_pos_MJ=E_wheel_pos_MJ,
        E_aux_MJ=E_aux_MJ,
        E_fric_MJ=E_fric_MJ,
        E_loss_engine_MJ=E_loss_engine_MJ,
        E_loss_mg_MJ=E_loss_mg_MJ,
        E_loss_inv_MJ=E_loss_inv_MJ,
        E_loss_batt_MJ=E_loss_batt_MJ,
        dE_batt_storage_MJ=dE_batt_storage_MJ,

        # residual
        E_residual_MJ=E_residual_MJ,
    )

# ===== ② Energy summary (Phase B aware) =====
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

    # Phase B: explicit split
    E_regen_to_batt_MJ: float
    E_engine_to_batt_MJ: float

    regen_utilization: float
    friction_share: float

    count_flag_batt_sat: int
    count_flag_eng_sat: int

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {
            "E_wheel_pos_MJ": self.E_wheel_pos_MJ,
            "E_wheel_neg_MJ": self.E_wheel_neg_MJ,
            "E_fric_MJ": self.E_fric_MJ,
            "E_aux_MJ": self.E_aux_MJ,
            "E_batt_dis_MJ": self.E_batt_dis_MJ,
            "E_batt_chg_MJ": self.E_batt_chg_MJ,
            "E_batt_net_MJ": self.E_batt_net_MJ,
            "E_eng_mech_MJ": self.E_eng_mech_MJ,
            "E_regen_to_batt_MJ": self.E_regen_to_batt_MJ,
            "E_engine_to_batt_MJ": self.E_engine_to_batt_MJ,
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
    Pfric = ts["P_brake_fric_W"].to_numpy(float) if "P_brake_fric_W" in ts.columns else np.zeros_like(Pw_del)

    Pw_pos = np.maximum(Pw_del, 0.0)
    Pw_neg = np.maximum(-Pw_del, 0.0)

    Pb_dis = np.maximum(Pb, 0.0)   # discharge
    Pb_chg = np.maximum(-Pb, 0.0)  # charge magnitude

    E_wheel_pos = _energy_MJ(Pw_pos, dt)
    E_wheel_neg = _energy_MJ(Pw_neg, dt)
    E_fric = _energy_MJ(np.maximum(Pfric, 0.0), dt)
    E_aux = _energy_MJ(np.maximum(Paux, 0.0), dt)
    E_batt_dis = _energy_MJ(Pb_dis, dt)
    E_batt_chg = _energy_MJ(Pb_chg, dt)
    E_batt_net = E_batt_dis - E_batt_chg
    E_eng_mech = _energy_MJ(np.maximum(Peng, 0.0), dt)

    E_hvac = None
    if "P_hvac_W" in ts.columns:
        E_hvac = _energy_MJ(np.maximum(ts["P_hvac_W"].to_numpy(float), 0.0), dt)

    # --- Phase B: prefer explicit split columns ---
    if "P_batt_chg_from_regen_W" in ts.columns and "P_batt_chg_from_engine_W" in ts.columns:
        Preg = ts["P_batt_chg_from_regen_W"].to_numpy(float)
        Peng2b = ts["P_batt_chg_from_engine_W"].to_numpy(float)
        E_regen_to_batt = _energy_MJ(np.maximum(Preg, 0.0), dt)
        E_engine_to_batt = _energy_MJ(np.maximum(Peng2b, 0.0), dt)
        # safety cap (should already hold if split is correct)
        E_regen_to_batt = min(E_regen_to_batt, E_wheel_neg)
    else:
        # Phase A fallback (heuristic)
        E_regen_to_batt = min(E_batt_chg, E_wheel_neg)
        E_engine_to_batt = max(E_batt_chg - E_regen_to_batt, 0.0)

    eps = 1e-9
    regen_util = float(E_regen_to_batt / max(E_wheel_neg, eps))
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
        E_engine_to_batt_MJ=E_engine_to_batt,
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
        (x0, y0),
        (cx0, y0), (cx1, y1), (x1, y1),
        (x1, y1 + h1),
        (cx1, y1 + h1), (cx0, y0 + h0), (x0, y0 + h0),
        (x0, y0),
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
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    Ew_pos, Ew_neg = es.E_wheel_pos_MJ, es.E_wheel_neg_MJ
    Efric, Eaux = es.E_fric_MJ, es.E_aux_MJ
    Eb_dis, Eb_chg = es.E_batt_dis_MJ, es.E_batt_chg_MJ
    Eeng = es.E_eng_mech_MJ

    # Phase B split（再導出しない）
    regen_to_batt = min(es.E_regen_to_batt_MJ, Ew_neg)
    engine_to_batt = max(es.E_engine_to_batt_MJ, 0.0)
    regen_not_stored = max(Ew_neg - regen_to_batt, 0.0)

    aux_from_batt = min(Eb_dis, Eaux)
    batt_to_wheel = max(Eb_dis - aux_from_batt, 0.0)

    wheel_need = max(Ew_pos - batt_to_wheel, 0.0)
    eng_to_wheel = min(Eeng, wheel_need)
    eng_unused = max(Eeng - eng_to_wheel - engine_to_batt, 0.0)

    # --- nodes ---
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
    if other > 1e-9:
        sinks.append(("Other", other, ENERGY_COLORS["Other"]))

    # --- layout params (MUST be defined BEFORE used) ---
    xL, xR = 0.05, 0.75
    boxW = 0.18
    y_top = 0.94
    y_bot = 0.06
    avail = y_top - y_bot

    gap = 0.012
    min_box_h = 0.010  # 0MJでも表示するための最低高さ

    vals_src = [v for _, v, _ in sources]
    vals_sink = [v for _, v, _ in sinks]

    def _need_height(scale: float, vals: list[float]) -> float:
        if not vals:
            return 0.0
        return sum(max(v * scale, min_box_h) for v in vals) + gap * (len(vals) - 1)

    total = max(sum(vals_src), sum(vals_sink), 1e-12)

    # binary search for max scale that fits both sides
    hi = avail / total
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        need = max(_need_height(mid, vals_src), _need_height(mid, vals_sink))
        if need <= avail + 1e-12:
            lo = mid
        else:
            hi = mid
    scale = lo

    # --- build boxes AFTER scale/min_box_h exist ---
    src_boxes = {}
    y = y_top
    for name, v, c in sources:
        h = max(v * scale, min_box_h)
        y -= h
        src_boxes[name] = (xL, y, boxW, h, c, v)
        y -= gap

    sink_boxes = {}
    y = y_top
    for name, v, c in sinks:
        h = max(v * scale, min_box_h)
        y -= h
        sink_boxes[name] = (xR, y, boxW, h, c, v)
        y -= gap

    # draw boxes + labels
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
        hh = v * scale  # ribbon thickness: TRUE magnitude
        y0b = y0 + src_off[src]
        y1b = y1 + sink_off[dst]
        _ribbon(ax, x0 + w0, y0b, hh, x1, y1b, hh, color=color, alpha=0.65)
        src_off[src] += hh
        sink_off[dst] += hh

    # flows
    connect("Engine_mech", "Wheel_traction", eng_to_wheel, ENERGY_COLORS["Engine_mech"])
    connect("Engine_mech", "Battery_charge", engine_to_batt, ENERGY_COLORS["Engine_mech"])
    if "Other" in sink_boxes:
        connect("Engine_mech", "Other", eng_unused, ENERGY_COLORS["Engine_mech"])

    connect("Battery_discharge", "Aux_load", aux_from_batt, ENERGY_COLORS["Battery_discharge"])
    connect("Battery_discharge", "Wheel_traction", batt_to_wheel, ENERGY_COLORS["Battery_discharge"])

    connect("Wheel_regen_available", "Battery_charge", regen_to_batt, ENERGY_COLORS["Wheel_regen_available"])
    if "Other" in sink_boxes:
        connect("Wheel_regen_available", "Other", regen_not_stored, ENERGY_COLORS["Wheel_regen_available"])

    ax.text(0.5, 0.02, NOTE_SANK, ha="center", va="bottom", fontsize=8)

def plot_loss_sankey(ax, budget: dict) -> None:
    """
    ②-B Fuel + Wheel_neg_source -> (Wheel_pos, Aux, Fric, Losses, ΔBatt, Residual)
    - 0MJでも箱を出す（min_box_h）
    - 箱は必ず y_bot..y_top に収まる（scaleを二分探索）
    - 右側ラベルは「外側」に出して非重複配置（leader line）
    """
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    def mj(k: str) -> float:
        try:
            return float(budget.get(k, 0.0) or 0.0)
        except Exception:
            return 0.0

    # --- sources ---
    sources = [
        ("Fuel_energy", max(mj("E_fuel_MJ"), 0.0), ENERGY_COLORS.get("Fuel_energy", "#8c564b")),
        ("Wheel_neg_source", max(mj("E_wheel_neg_source_MJ"), 0.0), ENERGY_COLORS.get("Wheel_regen_available", "#6baed6")),
    ]

    # --- sinks ---
    dE = mj("dE_batt_storage_MJ")
    resid = mj("E_residual_MJ")

    sinks = [
        ("Wheel_traction", max(mj("E_wheel_pos_MJ"), 0.0), ENERGY_COLORS.get("Wheel_traction", "#2ca02c")),
        ("Aux_load", max(mj("E_aux_MJ"), 0.0), ENERGY_COLORS.get("Aux_load", "#d62728")),
        ("Friction_brake_heat", max(mj("E_fric_MJ"), 0.0), ENERGY_COLORS.get("Friction_brake_heat", "#7f7f7f")),
        ("Engine_loss", max(mj("E_loss_engine_MJ"), 0.0), ENERGY_COLORS.get("Engine_loss", "#f2b6a0")),
        ("MG_loss", max(mj("E_loss_mg_MJ"), 0.0), ENERGY_COLORS.get("MG_loss", "#17becf")),
        ("INV_loss", max(mj("E_loss_inv_MJ"), 0.0), ENERGY_COLORS.get("INV_loss", "#9edae5")),
        ("Battery_loss", max(mj("E_loss_batt_MJ"), 0.0), ENERGY_COLORS.get("Battery_loss", "#bcbd22")),
        ("Delta_batt_storage", abs(dE), ENERGY_COLORS.get("Delta_batt_storage", "#9467bd")),
        ("Residual", abs(resid), ENERGY_COLORS.get("Residual", "#bdbdbd")),
    ]

    # --- layout params ---
    xL, xR = 0.06, 0.76
    boxW = 0.18

    y_top = 0.94
    y_bot = 0.06
    avail = y_top - y_bot

    gap = 0.012
    min_box_h = 0.010  # 0MJでも見せる

    vals_src = [v for _, v, _ in sources]
    vals_sink = [v for _, v, _ in sinks]

    def _need_height(scale: float, vals: list[float]) -> float:
        if not vals:
            return 0.0
        return sum(max(v * scale, min_box_h) for v in vals) + gap * (len(vals) - 1)

    total = max(sum(vals_src), sum(vals_sink), 1e-12)
    hi = avail / total
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        need = max(_need_height(mid, vals_src), _need_height(mid, vals_sink))
        if need <= avail + 1e-12:
            lo = mid
        else:
            hi = mid
    scale = lo

    # --- build boxes ---
    src_boxes = {}
    y = y_top
    for name, v, c in sources:
        h = max(v * scale, min_box_h)
        y -= h
        src_boxes[name] = (xL, y, boxW, h, c, v)
        y -= gap

    sink_boxes = {}
    y = y_top
    for name, v, c in sinks:
        h = max(v * scale, min_box_h)
        y -= h
        sink_boxes[name] = (xR, y, boxW, h, c, v)
        y -= gap

    # draw source boxes + labels（左は箱の近くでOK）
    for name, (x, y, w, h, c, v) in src_boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="none", alpha=0.85))
        ax.text(x + w + 0.01, y + h / 2, f"{name}\n{v:.1f} MJ", va="center", ha="left", fontsize=9)

    # draw sink boxes（ラベルは後で外側に非重複配置）
    for name, (x, y, w, h, c, v) in sink_boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="none", alpha=0.85))

    # --- non-overlap label placement for sinks (right side) ---
    # label text
    def _sink_label(name: str, v: float) -> str:
        if name == "Delta_batt_storage":
            return f"{name}\n{dE:+.1f} MJ"
        if name == "Residual":
            return f"{name}\n{resid:+.1f} MJ"
        return f"{name}\n{v:.1f} MJ"

    items = []
    for name, (x, y, w, h, c, v) in sink_boxes.items():
        yc = y + h / 2
        items.append((name, yc, _sink_label(name, v)))

    # sort top->bottom
    items.sort(key=lambda t: -t[1])

    # minimum label spacing in axis coords
    dy_min = 0.032  # ←ここを大きくするとさらに読みやすい（ただし詰め込み限界あり）
    y_targets = [it[1] for it in items]

    # push down to avoid overlap
    for i in range(1, len(y_targets)):
        y_targets[i] = min(y_targets[i], y_targets[i - 1] - dy_min)

    # if pushed too far below, shift up
    if y_targets:
        min_y = y_targets[-1]
        if min_y < (y_bot + 0.02):
            shift = (y_bot + 0.02) - min_y
            y_targets = [yy + shift for yy in y_targets]

        # clamp top if needed
        max_y = y_targets[0]
        if max_y > (y_top - 0.01):
            shift = max_y - (y_top - 0.01)
            y_targets = [yy - shift for yy in y_targets]

    x_text = 0.98
    x_line0 = xR + boxW
    x_line1 = x_text - 0.01

    for (name, yc, label), yt in zip(items, y_targets):
        # leader line
        ax.plot([x_line0, x_line1], [yc, yt], color="#666666", linewidth=0.6, alpha=0.7)
        ax.text(x_text, yt, label, va="center", ha="right", fontsize=9)

    # --- ribbons ---
    src_off = {k: 0.0 for k in src_boxes}
    sink_off = {k: 0.0 for k in sink_boxes}

    def connect(src: str, dst: str, v: float, color: str) -> None:
        if v <= 0:
            return
        x0, y0, w0, _, _, _ = src_boxes[src]
        x1, y1, _, _, _, _ = sink_boxes[dst]
        hh = v * scale
        y0b = y0 + src_off[src]
        y1b = y1 + sink_off[dst]
        _ribbon(ax, x0 + w0, y0b, hh, x1, y1b, hh, color=color, alpha=0.55)
        src_off[src] += hh
        sink_off[dst] += hh

    # allocation strategy:
    # wheel_neg first to regen-related sinks, then fill remaining by fuel
    wheel_neg = src_boxes["Wheel_neg_source"][5]
    remaining_wheel = wheel_neg

    priority = ["Delta_batt_storage", "Battery_loss", "MG_loss", "INV_loss", "Friction_brake_heat"]
    for dst in priority:
        v_dst = sink_boxes[dst][5]
        take = min(v_dst, remaining_wheel)
        connect("Wheel_neg_source", dst, take, src_boxes["Wheel_neg_source"][4])
        remaining_wheel -= take

    if remaining_wheel > 1e-9 and "Residual" in sink_boxes:
        take = min(remaining_wheel, sink_boxes["Residual"][5])
        connect("Wheel_neg_source", "Residual", take, src_boxes["Wheel_neg_source"][4])

    for dst in sink_boxes.keys():
        allocated = sink_off[dst] / max(scale, 1e-12)  # MJ already allocated
        need = max(sink_boxes[dst][5] - allocated, 0.0)
        connect("Fuel_energy", dst, need, src_boxes["Fuel_energy"][4])

    ax.text(0.5, 0.02, NOTE_SANK, ha="center", va="bottom", fontsize=8)

# ===== ① Overview =====
def plot_overview(ts: pd.DataFrame, outdir: Path, dpi: int = 160) -> None:
    _ensure_dir(outdir)
    t = ts["t_s"].to_numpy(float)
    phase = ts["phase"].astype(str).to_numpy() if "phase" in ts.columns else None
    mode = ts["mode"].astype(str).to_numpy() if "mode" in ts.columns else np.array(["NA"] * len(ts), dtype=str)

    fig = plt.figure(figsize=(13.5, 10.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.25, 1.25, 1.25], hspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    if phase is not None:
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

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if phase is not None:
        _shade_by_segments(ax2, t, phase, _phase_color, alpha=0.10)
    P_req = ts["P_wheel_req_W"].to_numpy(float) / 1000.0
    P_del = ts["P_wheel_deliv_W_dbg"].to_numpy(float) / 1000.0
    P_fric = ts["P_brake_fric_W"].to_numpy(float) / 1000.0 if "P_brake_fric_W" in ts.columns else np.zeros_like(P_del)

    ax2.plot(t, P_req, color=SIG_COL["wheel_req"], linestyle="--", linewidth=1.15, label="P_wheel_req")
    ax2.plot(t, P_del, color=SIG_COL["wheel_deliv"], linewidth=1.25, label="P_wheel_deliv")
    ax2.fill_between(t, 0.0, P_fric, color=SIG_COL["brake_fric"], alpha=0.25, label="P_brake_fric")
    ax2.axhline(0, linewidth=0.8, color="#000000", alpha=0.6)
    ax2.set_ylabel("Wheel power [kW]")
    ax2.set_ylim(_robust_sym_ylim(np.r_[P_req, P_del, P_fric]))
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    if phase is not None:
        _shade_by_segments(ax3, t, phase, _phase_color, alpha=0.10)
    soc = ts["soc_pct"].to_numpy(float) if "soc_pct" in ts.columns else ts.get("soc", pd.Series(np.nan, index=ts.index)).to_numpy(float)
    ax3.plot(t, soc, color=SIG_COL["soc"], linewidth=1.35, label="SOC")
    ax3.set_ylabel("SOC [%]")

    ax3b = ax3.twinx()
    P_batt = ts["P_batt_act_W"].to_numpy(float) / 1000.0
    lim_dis = ts["lim_batt_discharge_W"].to_numpy(float) / 1000.0 if "lim_batt_discharge_W" in ts.columns else np.zeros_like(P_batt)
    lim_chg = ts["lim_batt_charge_W"].to_numpy(float) / 1000.0 if "lim_batt_charge_W" in ts.columns else np.zeros_like(P_batt)

    ax3b.plot(t, P_batt, color=SIG_COL["batt_act"], linewidth=1.05, alpha=0.92, label="P_batt_act")
    ax3b.axhline(0, linewidth=0.8, color="#000000", alpha=0.6)
    ax3b.plot(t, +lim_dis, color=SIG_COL["lim_dis"], linewidth=0.9, linestyle="--", alpha=0.9, label="lim_dis")
    ax3b.plot(t, -lim_chg, color=SIG_COL["lim_chg"], linewidth=0.9, linestyle="--", alpha=0.9, label="lim_chg")
    ax3b.set_ylabel("P_batt_act [kW] (+dis / -chg)")

    sat = ts["flag_batt_sat"].to_numpy(int) if "flag_batt_sat" in ts.columns else np.zeros(len(ts), dtype=int)
    idx = np.where(sat > 0)[0]
    if idx.size > 0:
        ax3b.scatter(t[idx], P_batt[idx], marker="^", s=22, c="#d62728", alpha=0.9, linewidths=0.0, label="batt_sat")

    ax3b.set_ylim(_robust_sym_ylim(np.r_[P_batt, +lim_dis, -lim_chg]))
    h3, l3 = ax3.get_legend_handles_labels()
    h3b, l3b = ax3b.get_legend_handles_labels()
    ax3.legend(h3 + h3b, l3 + l3b, loc="upper left", fontsize=8, framealpha=0.9)

    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    if phase is not None:
        _shade_by_segments(ax4, t, phase, _phase_color, alpha=0.10)
    _draw_mode_band(ax4, t, mode, y0=0.83, y1=1.00, alpha=0.25)

    eng_rpm = ts["eng_rpm"].to_numpy(float)
    eng_tq = ts["eng_tq_Nm"].to_numpy(float)

    ax4.plot(t, eng_rpm, color=SIG_COL["eng_rpm"], linewidth=1.05, label="eng_rpm")
    ax4.set_ylabel("Engine speed [rpm]")

    ax4b = ax4.twinx()
    ax4b.plot(t, eng_tq, color=SIG_COL["eng_tq"], linewidth=1.0, alpha=0.9, label="eng_tq")
    ax4b.set_ylabel("Engine torque [Nm]")

    idx_sat = np.where(ts["flag_eng_sat"].to_numpy(int) > 0)[0] if "flag_eng_sat" in ts.columns else np.array([], dtype=int)
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
    Pfric = ts["P_brake_fric_W"].to_numpy(float) if "P_brake_fric_W" in ts.columns else np.zeros_like(Pw_del)
    Paux = ts["P_aux_W"].to_numpy(float)
    Pb = ts["P_batt_act_W"].to_numpy(float)
    Pb_dis = np.maximum(Pb, 0.0)
    Pb_chg = np.maximum(-Pb, 0.0)
    Peng = np.maximum(ts["P_eng_mech_W"].to_numpy(float), 0.0)

    df = pd.DataFrame({
        "phase": ts["phase"].astype(str),
        "E_wheel_pos_MJ": Pw_pos * dt / 1e6,
        "E_wheel_neg_MJ": Pw_neg * dt / 1e6,
        "E_fric_MJ": np.maximum(Pfric, 0.0) * dt / 1e6,
        "E_aux_MJ": np.maximum(Paux, 0.0) * dt / 1e6,
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
        "E_regen_to_batt_MJ","E_engine_to_batt_MJ",
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

    fig2, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["KPI", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    ax.set_title("Energy Flow KPIs (MJ / ratios / counts)")
    _savefig(outdir / "02_energy_kpis.png", dpi=dpi)

    budget = compute_loss_budget(ts)
    with open(outdir / "loss_budget.json","w") as f:
        json.dump(budget, f, indent=2)

    if REQUIRED_BUDGET_COLS.issubset(set(ts.columns)):
            budget = compute_loss_budget(ts)

            with open(outdir / "loss_budget.json", "w", encoding="utf-8") as f:
                json.dump(budget, f, indent=2)

            # ②-B Sankey-like budget plot
            fig3, ax3 = plt.subplots(figsize=(12.5, 6.8))
            plot_loss_sankey(ax3, budget)
            ax3.set_title("Energy Flow②-B: Fuel/Loss/ΔSOC Budget (MJ)")
            _savefig(outdir / "02_energy_loss_sankey.png", dpi=dpi, pad_inches=0.35)

            # ②-B Budget table
            order = [
                "E_fuel_MJ",
                "E_wheel_neg_source_MJ",
                "E_wheel_pos_MJ",
                "E_aux_MJ",
                "E_fric_MJ",
                "E_loss_engine_MJ",
                "E_loss_mg_MJ",
                "E_loss_inv_MJ",
                "E_loss_batt_MJ",
                "dE_batt_storage_MJ",
                "E_residual_MJ",
            ]
            rows3 = [[k, f"{float(budget.get(k, 0.0)):.3f}"] for k in order]

            fig4, ax4 = plt.subplots(figsize=(7.8, 4.8))
            ax4.axis("off")
            table2 = ax4.table(cellText=rows3, colLabels=["Budget term", "MJ"], loc="center")
            table2.auto_set_font_size(False)
            table2.set_fontsize(9)
            table2.scale(1.0, 1.2)
            ax4.set_title("Fuel/Loss/ΔSOC Budget Terms (MJ)")
            _savefig(outdir / "02_loss_kpis.png", dpi=dpi)

    else:
        missing = sorted(list(REQUIRED_BUDGET_COLS - set(ts.columns)))
        print(f"[WARN] Skip loss budget plots: missing columns: {missing}")

def run_all(timeseries: str, outdir: str, dpi: int = 160) -> None:
    out = Path(outdir)
    _ensure_dir(out)
    ts = pd.read_csv(timeseries)
    plot_overview(ts, out / "01_overview", dpi=dpi)
    plot_energy_flow(ts, out / "02_energy_flow", dpi=dpi)
    print(f"[OK] Saved dashboards to: {out.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="THS dashboards (Phase B aware)")
    ap.add_argument("--timeseries", required=True, help="Path to timeseries.csv")
    ap.add_argument("--outdir", default="plots_dashboards_B", help="Output directory")
    ap.add_argument("--dpi", type=int, default=160)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run_all(args.timeseries, args.outdir, dpi=args.dpi)


if __name__ == "__main__":
    main()
