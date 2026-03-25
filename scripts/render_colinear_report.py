from __future__ import annotations

import argparse
import base64
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_KEYS = [
    ("overall_pass", "Overall Pass"),
    ("A_pass", "A Pass"),
    ("B_pass", "B Pass"),
    ("duration_s", "Duration [s]"),
    ("EV_share_time", "EV Share Time"),
    ("count_eng_start", "Engine Starts"),
    ("fuel_on_time_ratio", "Fuel On Ratio"),
    ("soc_recon_resid_max_abs_pct", "SOC Recon Resid Max Abs [pct]"),
    ("bus_balance_resid_max_W", "Bus Balance Resid Max [W]"),
    ("TTW_eff", "TTW Efficiency"),
    ("regen_utilization", "Regen Utilization"),
    ("fuel_g_per_km", "Fuel [g/km]"),
    ("shortfall_step_ratio", "Shortfall Step Ratio"),
    ("count_mg1_overspeed", "MG1 Overspeed Count"),
    ("count_mg2_overspeed", "MG2 Overspeed Count"),
    ("E_fuel_MJ", "Fuel Energy [MJ]"),
    ("E_loss_total_MJ", "Total Loss [MJ]"),
    ("E_loss_engine_MJ", "Engine Loss [MJ]"),
    ("E_regen_to_batt_MJ", "Regen to Batt [MJ]"),
    ("E_engine_to_batt_MJ", "Engine to Batt [MJ]"),
]


@dataclass(frozen=True)
class RunFiles:
    run_dir: Path
    timeseries_csv: Path
    kpis_phaseB_json: Path | None
    gate_detail_json: Path | None
    budgets_phaseB_json: Path | None


@dataclass(frozen=True)
class SnapshotCard:
    label: str
    t_s: float
    mode: str
    engine_on: bool
    figure_b64: str
    meta_rows: list[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a minimal single-run co-linear visualization HTML report."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to a single Phase B run folder, e.g. artifacts/out_phaseB_suite_after_index_fix/B00_baseline",
    )
    parser.add_argument(
        "--out_html",
        default=None,
        help="Output HTML path. Defaults to <run_dir>/Viz/colinear_report.html",
    )
    parser.add_argument(
        "--max_snapshots",
        type=int,
        default=5,
        help="Maximum number of representative snapshots to render (default: 5)",
    )
    return parser.parse_args()


def resolve_run_files(run_dir: Path) -> RunFiles:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    timeseries_candidates = list(run_dir.rglob("timeseries.csv"))
    if not timeseries_candidates:
        raise FileNotFoundError(f"timeseries.csv not found under: {run_dir}")
    timeseries_csv = timeseries_candidates[0]

    return RunFiles(
        run_dir=run_dir,
        timeseries_csv=timeseries_csv,
        kpis_phaseB_json=_find_optional(run_dir, "kpis_phaseB.json"),
        gate_detail_json=_find_optional(run_dir, "gate_detail.json"),
        budgets_phaseB_json=_find_optional(run_dir, "budgets_phaseB.json"),
    )


def _find_optional(run_dir: Path, filename: str) -> Path | None:
    candidates = list(run_dir.rglob(filename))
    return candidates[0] if candidates else None


def load_timeseries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "t_s" not in df.columns:
        raise ValueError(f"Required column 't_s' is missing in {csv_path}")
    return df


def load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "veh_spd_mps" in out.columns:
        out["veh_spd_kph"] = out["veh_spd_mps"] * 3.6
    else:
        out["veh_spd_kph"] = pd.NA

    if "eng_rpm" in out.columns:
        out["engine_on"] = out["eng_rpm"].fillna(0.0) > 0.0
    else:
        out["engine_on"] = False

    if "P_batt_chg_from_regen_W" in out.columns:
        out["regen_active"] = out["P_batt_chg_from_regen_W"].fillna(0.0) > 0.0
    else:
        out["regen_active"] = False

    prev_engine_on = out["engine_on"].shift(1, fill_value=False)
    out["engine_start"] = (~prev_engine_on) & out["engine_on"]

    if "fuel_cut" in out.columns:
        out["fuel_cut_active"] = out["fuel_cut"].fillna(0) == 1
    else:
        out["fuel_cut_active"] = False

    out["snapshot_label"] = ""
    return out


def build_summary(
    df: pd.DataFrame,
    kpis_phaseB: dict[str, Any],
    gate_detail: dict[str, Any],
    budgets_phaseB: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    merged = {}
    merged.update(budgets_phaseB)
    merged.update(gate_detail)
    merged.update(kpis_phaseB)

    for key, _label in SUMMARY_KEYS:
        if key in merged:
            summary[key] = merged[key]

    if "duration_s" not in summary and not df.empty:
        summary["duration_s"] = float(df["t_s"].iloc[-1] - df["t_s"].iloc[0])

    if "EV_share_time" not in summary and "mode" in df.columns:
        mode_series = df["mode"].fillna("")
        summary["EV_share_time"] = float((mode_series == "EV").mean())

    if "count_eng_start" not in summary and "engine_start" in df.columns:
        summary["count_eng_start"] = int(df["engine_start"].sum())

    if "fuel_on_time_ratio" not in summary and "engine_on" in df.columns:
        summary["fuel_on_time_ratio"] = float(df["engine_on"].mean())

    if "overall_pass" not in summary:
        if "PASS" in gate_detail:
            summary["overall_pass"] = bool(gate_detail["PASS"])
        elif kpis_phaseB:
            summary["overall_pass"] = bool(kpis_phaseB.get("OVERALL_PASS", False))

    if "A_pass" not in summary:
        a_gate = gate_detail.get("A", {}) if isinstance(gate_detail.get("A", {}), dict) else {}
        if "PASS" in a_gate:
            summary["A_pass"] = bool(a_gate["PASS"])
        elif kpis_phaseB:
            summary["A_pass"] = bool(kpis_phaseB.get("A_PASS", False))

    if "B_pass" not in summary:
        b_gate = gate_detail.get("B", {}) if isinstance(gate_detail.get("B", {}), dict) else {}
        if "PASS" in b_gate:
            summary["B_pass"] = bool(b_gate["PASS"])
        elif kpis_phaseB:
            summary["B_pass"] = bool(kpis_phaseB.get("B_PASS", False))

    return summary


def select_snapshot_indices(df: pd.DataFrame, max_snapshots: int = 5) -> list[int]:
    if df.empty:
        return []

    picks: list[int] = []

    def add_idx(idx: int | None) -> None:
        if idx is None:
            return
        if idx < 0 or idx >= len(df):
            return
        if idx in picks:
            return
        t = float(df.iloc[idx]["t_s"])
        for existing in picks:
            t_existing = float(df.iloc[existing]["t_s"])
            if abs(t - t_existing) < 5.0:
                return
        picks.append(idx)

    add_idx(0)
    add_idx(_first_true_index(df, "engine_start"))

    regen_idx = _first_true_index(df, "regen_active") if "regen_active" in df.columns else None
    if regen_idx is None and "mode" in df.columns:
        regen_idx = _first_true_index_bool(df["mode"].fillna("") == "REGEN")
    add_idx(regen_idx)

    if "veh_spd_mps" in df.columns and df["veh_spd_mps"].notna().any():
        add_idx(int(df["veh_spd_mps"].astype(float).idxmax()))

    add_idx(len(df) - 1)

    if len(picks) < max_snapshots:
        for frac in [0.25, 0.50, 0.75]:
            add_idx(int(round((len(df) - 1) * frac)))
            if len(picks) >= max_snapshots:
                break

    picks = sorted(picks, key=lambda i: float(df.iloc[i]["t_s"]))
    return picks[:max_snapshots]


def _first_true_index(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    return _first_true_index_bool(df[col].fillna(False))


def _first_true_index_bool(series: pd.Series) -> int | None:
    true_indices = series[series.astype(bool)].index.tolist()
    return int(true_indices[0]) if true_indices else None


def render_timeseries_figure(df: pd.DataFrame, snapshot_indices: list[int]) -> str:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes
    t = _series(df, "t_s")

    if "veh_spd_kph" in df.columns and df["veh_spd_kph"].notna().any():
        ax1.plot(t, df["veh_spd_kph"], label="Vehicle speed [kph]")
    ax1.set_ylabel("Speed [kph]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    if "soc_pct" in df.columns and df["soc_pct"].notna().any():
        ax2.plot(t, df["soc_pct"], label="SOC [%]")
    ax2.set_ylabel("SOC [%]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    for col, label in [
        ("eng_rpm", "Ne = eng_rpm"),
        ("mg1_rpm", "Ng = mg1_rpm"),
        ("ring_rpm", "Np = ring_rpm"),
        ("mg2_rpm", "Nm = mg2_rpm"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax3.plot(t, df[col], label=label)
    ax3.set_ylabel("Speed [rpm]")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", ncol=2)

    if "P_batt_chem_W" in df.columns and df["P_batt_chem_W"].notna().any():
        ax4.plot(t, df["P_batt_chem_W"], label="P_batt_chem_W")
    ax4.set_ylabel("Power [W]")
    ax4.set_xlabel("Time [s]")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    for ax in axes:
        _draw_state_bands(ax, df, "engine_on", "Engine ON")
        _draw_state_bands(ax, df, "regen_active", "Regen")
        _draw_snapshot_markers(ax, df, snapshot_indices)

    fig.tight_layout()
    return fig_to_base64(fig)


def _draw_state_bands(ax: plt.Axes, df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        return
    intervals = _find_true_intervals(df, col)
    ylim = ax.get_ylim()
    for i, (t0, t1) in enumerate(intervals):
        ax.axvspan(t0, t1, alpha=0.08, label=label if i == 0 else None)
    ax.set_ylim(ylim)


def _draw_snapshot_markers(ax: plt.Axes, df: pd.DataFrame, snapshot_indices: list[int]) -> None:
    for idx in snapshot_indices:
        ax.axvline(float(df.iloc[idx]["t_s"]), linestyle="--", linewidth=1.0, alpha=0.7)


def _find_true_intervals(df: pd.DataFrame, col: str) -> list[tuple[float, float]]:
    vals = df[col].fillna(False).astype(bool).tolist()
    times = df["t_s"].astype(float).tolist()
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for i, active in enumerate(vals):
        if active and start is None:
            start = times[i]
        if (not active) and start is not None:
            intervals.append((start, times[i]))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))
    return intervals


def render_colinear_snapshot(row: pd.Series, label: str) -> str:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(top=0.72, right=0.80)

    rpm_eps = 1e-6
    rho_vis = _infer_rho_vis(row)
    grm_vis = _infer_grm_vis(row)

    x_mg1 = 0.0
    x_engine = 1.0
    x_ring = x_engine + rho_vis
    x_fixed = x_ring + 1.0
    x_mg2 = x_fixed + grm_vis

    Ng_raw = _float_or_nan(row.get("mg1_rpm"))
    Ne_raw = _float_or_nan(row.get("eng_rpm"))
    Np_raw = _float_or_nan(row.get("ring_rpm"))
    Nm_raw = _float_or_nan(row.get("mg2_rpm"))

    Ne_vis = 0.0 if (not math.isnan(Ne_raw) and abs(Ne_raw) < rpm_eps) else Ne_raw
    Np_vis = Np_raw
    Ng_vis = float("nan")
    Nm_vis = float("nan")
    if not math.isnan(Ne_vis) and not math.isnan(Np_vis) and rho_vis > 0.0:
        Ng_vis = ((1.0 + rho_vis) * Ne_vis - Np_vis) / rho_vis
    if not math.isnan(Np_vis) and grm_vis > 0.0:
        Nm_vis = -grm_vis * Np_vis

    Tg = _float_or_nan(row.get("mg1_tq_Nm"))
    Te = _float_or_nan(row.get("eng_tq_Nm"))
    Tp = _float_or_nan(row.get("T_ring_deliv_Nm"))
    Tm = _float_or_nan(row.get("mg2_tq_Nm"))
    torques = [Tg, Te, Tp, Tm]

    ax.axhline(0.0, color="0.55", linewidth=2.0, alpha=0.9)

    ax.text(
        (x_mg1 + x_ring) / 2.0,
        1.25,
        "Front planetary",
        color="tab:red",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        (x_ring + x_mg2) / 2.0,
        1.25,
        "Rear planetary",
        color="tab:blue",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontsize=12,
        fontweight="bold",
    )

    shaft_specs = [
        (x_mg1, "MG1\nFront Sun\n(Ng)"),
        (x_engine, "Engine\nFront Carrier\n(Ne)"),
        (x_ring, "Ring\nShared Node\n(Np)"),
        (x_mg2, "MG2\nRear Sun\n(Nm)"),
    ]
    for x, shaft in shaft_specs:
        ax.axvline(x, linestyle=":", linewidth=1.0, alpha=0.4, color="0.45")
        ax.text(
            x,
            1.04,
            shaft,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            linespacing=1.25,
        )

    ax.axvline(x_fixed, linestyle=":", linewidth=1.0, alpha=0.35, color="0.45")
    ax.scatter([x_fixed], [0.0], s=90, color="black", zorder=6)
    ax.text(x_fixed + 0.06, 0.0, "Rear Carrier\nFixed", va="bottom", ha="left", fontsize=10)

    if not any(math.isnan(v) for v in [Ng_vis, Ne_vis, Np_vis]):
        ax.plot([x_mg1, x_engine, x_ring], [Ng_vis, Ne_vis, Np_vis], color="tab:red", linewidth=2.4, zorder=3)
    if not any(math.isnan(v) for v in [Np_vis, Nm_vis]):
        ax.plot([x_ring, x_fixed, x_mg2], [Np_vis, 0.0, Nm_vis], color="tab:blue", linewidth=2.4, zorder=3)

    concept_points = [
        (x_mg1, Ng_vis, "Ng"),
        (x_engine, Ne_vis, "Ne"),
        (x_ring, Np_vis, "Np"),
        (x_mg2, Nm_vis, "Nm"),
    ]
    for x, s_val, tag in concept_points:
        if math.isnan(s_val):
            continue
        ax.scatter([x], [s_val], s=28, color="black", zorder=7)
        ax.text(x - 0.05, s_val, tag, ha="right", va="center", fontsize=10)

    raw_points = [Ng_raw, Ne_raw, Np_raw, Nm_raw]
    raw_x = [x_mg1, x_engine, x_ring, x_mg2]
    vis_points = [Ng_vis, Ne_vis, Np_vis, Nm_vis]
    for x, raw_val, vis_val in zip(raw_x, raw_points, vis_points):
        if math.isnan(raw_val):
            continue
        if math.isnan(vis_val) or abs(raw_val - vis_val) > 1e-6:
            ax.scatter([x], [raw_val], s=48, facecolors="none", edgecolors="0.4", linewidths=1.2, zorder=5)

    arrow_scale = _torque_arrow_scale(torques)
    for x, tq in zip([x_mg1, x_engine, x_ring, x_mg2], torques):
        if math.isnan(tq):
            continue
        dy = tq * arrow_scale
        ax.arrow(
            x,
            0.0,
            0.0,
            dy,
            width=0.03,
            head_width=0.12,
            head_length=max(abs(dy) * 0.15, 20.0),
            length_includes_head=True,
            color="black",
            alpha=0.9,
            zorder=4,
        )

    mode_label = _display_mode_label(row)
    t_s = _float_or_nan(row.get("t_s"))
    engine_on = bool(row.get("engine_on", False))
    fig.suptitle(
        f"{label} | t={t_s:.1f}s | mode={mode_label} | engine_on={engine_on}",
        y=0.985,
        fontsize=13,
    )
    ax.set_xticks([])
    ax.set_ylabel("Speed [rpm]")
    ax.grid(True, axis="y", alpha=0.3)

    line_note = "line=planetary constraint"
    if any((not math.isnan(r) and not math.isnan(v) and abs(r - v) > 1e-6) for r, v in zip(raw_points, vis_points)):
        line_note = "line=planetary constraint | hollow marker=raw speed"

    meta_lines = [
        f"Ne(raw)={_fmt_num(Ne_raw, 0)} rpm | Te={_fmt_num(row.get('eng_tq_Nm'), 1)} Nm",
        f"Ng(raw)={_fmt_num(Ng_raw, 0)} rpm | Tg={_fmt_num(row.get('mg1_tq_Nm'), 1)} Nm",
        f"Np(raw)={_fmt_num(Np_raw, 0)} rpm | Tp={_fmt_num(row.get('T_ring_deliv_Nm'), 1)} Nm",
        f"Nm(raw)={_fmt_num(Nm_raw, 0)} rpm | Tm={_fmt_num(row.get('mg2_tq_Nm'), 1)} Nm",
        f"rho≈{rho_vis:.3f} | grm≈{grm_vis:.3f}",
        line_note,
        _mode_explanation(row),
    ]
    ax.text(
        1.02,
        0.98,
        "\n".join(meta_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    finite_speeds = [s for s in [Ng_vis, Ne_vis, Np_vis, Nm_vis, Ng_raw, Ne_raw, Np_raw, Nm_raw] if not math.isnan(s)] + [0.0]
    y_min = min(finite_speeds)
    y_max = max(finite_speeds)
    pad = max(40.0, 0.12 * max(abs(y_min), abs(y_max), 1.0))
    ax.set_ylim(y_min - pad, y_max + pad)

    return fig_to_base64(fig)


def _display_mode_label(row: pd.Series) -> str:
    mode = str(row.get("mode", "N/A"))
    eng_rpm = _float_or_nan(row.get("eng_rpm"))
    if mode.lower() == "regen":
        if math.isnan(eng_rpm) or abs(eng_rpm) < 1e-6:
            return "Regen (engine stop)"
        return "Regen (engine spin)"
    return mode


def _mode_explanation(row: pd.Series) -> str:
    mode = str(row.get("mode", "N/A"))
    mode_lower = mode.lower()
    engine_on = bool(row.get("engine_on", False))
    fuel_cut = bool(row.get("fuel_cut_active", False))

    if mode_lower == "ev":
        if not engine_on:
            return "EV note: engine stopped"
        if fuel_cut:
            return "EV note: EV-labelled state with engine spinning under fuel cut"
        return "EV note: EV-labelled state while engine speed is nonzero"

    if mode_lower == "regen":
        if not engine_on:
            return "Regen note: regenerative decel with engine stopped"
        if fuel_cut:
            return "Regen note: regenerative decel with engine spinning under fuel cut"
        return "Regen note: regenerative decel with engine spinning"

    if fuel_cut and engine_on:
        return "State note: engine spinning with fuel cut"

    return "State note: raw mode and engine_on are shown separately"


def _infer_rho_vis(row: pd.Series) -> float:
    Ng = _float_or_nan(row.get("mg1_rpm"))
    Ne = _float_or_nan(row.get("eng_rpm"))
    Np = _float_or_nan(row.get("ring_rpm"))
    if math.isnan(Ng) or math.isnan(Ne) or math.isnan(Np):
        return 1.0
    denom = Ne - Ng
    if abs(denom) < 1e-9:
        return 1.0
    rho = (Np - Ne) / denom
    if not math.isfinite(rho) or rho <= 0.0:
        return 1.0
    return float(rho)


def _infer_grm_vis(row: pd.Series) -> float:
    Np = _float_or_nan(row.get("ring_rpm"))
    Nm = _float_or_nan(row.get("mg2_rpm"))
    if math.isnan(Np) or math.isnan(Nm):
        return 1.0
    if abs(Np) < 1e-9:
        return 1.0
    grm = -Nm / Np
    if not math.isfinite(grm) or grm <= 0.0:
        return 1.0
    return float(grm)


def _torque_arrow_scale(torques: list[float]) -> float:
    finite = [abs(x) for x in torques if not math.isnan(x)]
    if not finite:
        return 1.0
    max_abs = max(finite)
    if max_abs < 1e-9:
        return 1.0
    return 120.0 / max_abs


def build_snapshot_cards(df: pd.DataFrame, indices: list[int]) -> list[SnapshotCard]:
    cards: list[SnapshotCard] = []
    for order, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        label = f"S{order}"
        fig_b64 = render_colinear_snapshot(row, label)
        meta_rows = [
            ("Mode", str(row.get("mode", "N/A"))),
            ("Engine ON", str(bool(row.get("engine_on", False)))),
            ("Fuel cut", str(bool(row.get("fuel_cut_active", False)))),
            ("Vehicle speed [kph]", _fmt_num(row.get("veh_spd_kph"), 1)),
            ("SOC [%]", _fmt_num(row.get("soc_pct"), 2)),
            ("P_batt_chem_W", _fmt_num(row.get("P_batt_chem_W"), 1)),
            ("P_batt_chg_from_regen_W", _fmt_num(row.get("P_batt_chg_from_regen_W"), 1)),
            ("P_batt_chg_from_engine_W", _fmt_num(row.get("P_batt_chg_from_engine_W"), 1)),
            ("T_ring_req_Nm", _fmt_num(row.get("T_ring_req_Nm"), 1)),
            ("T_ring_deliv_Nm", _fmt_num(row.get("T_ring_deliv_Nm"), 1)),
            ("shortfall_tq_Nm", _fmt_num(row.get("shortfall_tq_Nm"), 1)),
            ("State note", _mode_explanation(row)),
        ]
        cards.append(
            SnapshotCard(
                label=label,
                t_s=float(row["t_s"]),
                mode=str(row.get("mode", "N/A")),
                engine_on=bool(row.get("engine_on", False)),
                figure_b64=fig_b64,
                meta_rows=meta_rows,
            )
        )
    return cards


def build_html(
    run_dir: Path,
    source_files: RunFiles,
    summary: dict[str, Any],
    timeseries_b64: str,
    snapshot_cards: list[SnapshotCard],
    warnings: list[str],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_cards = "\n".join(_render_summary_cards(summary))
    snapshot_html = "\n".join(_render_snapshot_card(card) for card in snapshot_cards)
    warnings_html = "".join(f"<li>{_html_escape(w)}</li>" for w in warnings)

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Co-linear Report - {run_dir.name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; background: #fff; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ color: #555; margin-bottom: 20px; }}
    .section {{ margin-top: 28px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; background: #fafafa; }}
    .card-title {{ font-size: 12px; color: #666; margin-bottom: 6px; text-transform: uppercase; }}
    .card-value {{ font-size: 22px; font-weight: 700; }}
    .figure {{ width: 100%; max-width: 1200px; border: 1px solid #ddd; border-radius: 10px; }}
    .snapshots {{ display: flex; flex-direction: column; gap: 20px; }}
    .snapshot-card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; }}
    .snapshot-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }}
    .snapshot-title {{ font-size: 18px; font-weight: 700; }}
    .snapshot-meta {{ color: #555; font-size: 14px; }}
    .snapshot-body {{ display: grid; grid-template-columns: minmax(600px, 1fr) 320px; gap: 18px; align-items: start; }}
    .meta-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .meta-table td {{ padding: 5px 0; border-bottom: 1px solid #eee; vertical-align: top; }}
    .meta-table td:first-child {{ color: #666; width: 55%; }}
    .warn {{ border: 1px solid #f0c36d; background: #fff8e6; border-radius: 10px; padding: 12px 14px; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Co-linear Visualization Report</h1>
  <div class=\"meta\">
    <div><strong>Run:</strong> {run_dir.name}</div>
    <div><strong>Source folder:</strong> <code>{_html_escape(str(run_dir))}</code></div>
    <div><strong>Timeseries file:</strong> <code>{_html_escape(str(source_files.timeseries_csv))}</code></div>
    <div><strong>Generated:</strong> {generated_at}</div>
  </div>

  {f'<div class="warn"><strong>Warnings</strong><ul>{warnings_html}</ul></div>' if warnings else ''}

  <div class=\"section\">
    <h2>Summary</h2>
    <div class=\"summary-grid\">{summary_cards}</div>
  </div>

  <div class=\"section\">
    <h2>Time Series</h2>
    <img class=\"figure\" src=\"data:image/png;base64,{timeseries_b64}\" alt=\"time series\" />
  </div>

  <div class=\"section\">
    <h2>Representative Co-linear Snapshots</h2>
    <div class=\"snapshots\">{snapshot_html}</div>
  </div>
</body>
</html>
"""


def _render_summary_cards(summary: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    for key, label in SUMMARY_KEYS:
        if key not in summary:
            continue
        cards.append(
            f'<div class="card"><div class="card-title">{_html_escape(label)}</div><div class="card-value">{_html_escape(_fmt_summary_value(summary[key]))}</div></div>'
        )
    if not cards:
        cards.append('<div class="card"><div class="card-title">Summary</div><div class="card-value">No JSON summary found</div></div>')
    return cards


def _render_snapshot_card(card: SnapshotCard) -> str:
    meta_table_rows = "".join(
        f"<tr><td>{_html_escape(k)}</td><td>{_html_escape(v)}</td></tr>" for k, v in card.meta_rows
    )
    return f"""
    <div class=\"snapshot-card\">
      <div class=\"snapshot-head\">
        <div class=\"snapshot-title\">{_html_escape(card.label)} @ t={card.t_s:.1f}s</div>
        <div class=\"snapshot-meta\">mode={_html_escape(card.mode)} | engine_on={card.engine_on}</div>
      </div>
      <div class=\"snapshot-body\">
        <img class=\"figure\" src=\"data:image/png;base64,{card.figure_b64}\" alt=\"snapshot\" />
        <table class=\"meta-table\">{meta_table_rows}</table>
      </div>
    </div>
    """


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([pd.NA] * len(df))


def _float_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _fmt_num(value: Any, digits: int) -> str:
    num = _float_or_nan(value)
    if math.isnan(num):
        return "N/A"
    return f"{num:.{digits}f}"


def _fmt_summary_value(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_files = resolve_run_files(run_dir)

    warnings: list[str] = []
    df = add_derived_columns(load_timeseries(run_files.timeseries_csv))

    kpis_phaseB = load_optional_json(run_files.kpis_phaseB_json)
    gate_detail = load_optional_json(run_files.gate_detail_json)
    budgets_phaseB = load_optional_json(run_files.budgets_phaseB_json)

    if not run_files.kpis_phaseB_json:
        warnings.append("kpis_phaseB.json not found; summary will use timeseries fallbacks where possible.")
    if not run_files.gate_detail_json:
        warnings.append("gate_detail.json not found; gate-related summary fields may be missing.")
    if not run_files.budgets_phaseB_json:
        warnings.append("budgets_phaseB.json not found; energy-loss summary fields may be missing.")

    summary = build_summary(df, kpis_phaseB, gate_detail, budgets_phaseB)
    snapshot_indices = select_snapshot_indices(df, max_snapshots=max(1, args.max_snapshots))
    if not snapshot_indices:
        warnings.append("No snapshot indices were selected because the timeseries is empty.")

    timeseries_b64 = render_timeseries_figure(df, snapshot_indices)
    snapshot_cards = build_snapshot_cards(df, snapshot_indices)

    html = build_html(
        run_dir=run_dir,
        source_files=run_files,
        summary=summary,
        timeseries_b64=timeseries_b64,
        snapshot_cards=snapshot_cards,
        warnings=warnings,
    )

    out_html = Path(args.out_html).resolve() if args.out_html else run_dir / "Viz" / "colinear_report.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote report: {out_html}")


if __name__ == "__main__":
    main()
