from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_timeseries_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("timeseries*.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No timeseries*.csv found in: {run_dir}")


def _to_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "on"}
    if pd.isna(x):
        return False
    return bool(float(x) != 0.0)


def _to_num(x: Any) -> float | None:
    if x is None or pd.isna(x):
        return None
    return float(x)


def _median_positive(values: list[float], fallback: float = 1.0) -> float:
    clean = [v for v in values if np.isfinite(v) and v > 0.0]
    if not clean:
        return fallback
    return float(np.median(np.asarray(clean, dtype=float)))


def _round_up_nice(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1000.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    scaled = value / base
    if scaled <= 1.0:
        nice = 1.0
    elif scaled <= 1.5:
        nice = 1.5
    elif scaled <= 2.0:
        nice = 2.0
    elif scaled <= 3.0:
        nice = 3.0
    elif scaled <= 5.0:
        nice = 5.0
    elif scaled <= 6.0:
        nice = 6.0
    else:
        nice = 10.0
    return float(nice * base)


def _infer_colinear_ratios(ts: pd.DataFrame) -> dict[str, float]:
    rho_samples: list[float] = []
    grm_samples: list[float] = []

    if {"mg1_rpm", "eng_rpm", "ring_rpm"}.issubset(ts.columns):
        for _, row in ts.iterrows():
            Ng = _to_num(row.get("mg1_rpm"))
            Ne = _to_num(row.get("eng_rpm"))
            Np = _to_num(row.get("ring_rpm"))
            if Ng is None or Ne is None or Np is None:
                continue
            denom = Ne - Ng
            if abs(denom) < 1e-9:
                continue
            rho = (Np - Ne) / denom
            if np.isfinite(rho) and rho > 0.0:
                rho_samples.append(float(rho))

    if {"mg2_rpm", "ring_rpm"}.issubset(ts.columns):
        for _, row in ts.iterrows():
            Nm = _to_num(row.get("mg2_rpm"))
            Np = _to_num(row.get("ring_rpm"))
            if Nm is None or Np is None or abs(Np) < 1e-9:
                continue
            grm = -Nm / Np
            if np.isfinite(grm) and grm > 0.0:
                grm_samples.append(float(grm))

    return {
        "rho": _median_positive(rho_samples, fallback=1.0),
        "grm": _median_positive(grm_samples, fallback=1.0),
    }


def _mode_events(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "mode" not in df.columns or "t_s" not in df.columns:
        return []
    mode = df["mode"].astype(str).to_numpy()
    t_s = df["t_s"].to_numpy(float)
    if len(mode) == 0:
        return []
    out = [{"idx": 0, "t_s": float(t_s[0]), "kind": "mode", "value": mode[0]}]
    for i in range(1, len(mode)):
        if mode[i] != mode[i - 1]:
            out.append({"idx": int(i), "t_s": float(t_s[i]), "kind": "mode", "value": mode[i]})
    return out


def _build_snapshots(df: pd.DataFrame) -> list[dict[str, Any]]:
    n = len(df)
    if n == 0:
        return []
    t_s = df["t_s"].to_numpy(float) if "t_s" in df.columns else np.arange(n, dtype=float)
    speed = df["veh_spd_mps"].to_numpy(float) if "veh_spd_mps" in df.columns else np.zeros(n)
    engine_on = (df["eng_rpm"].to_numpy(float) > 1.0) if "eng_rpm" in df.columns else np.zeros(n, dtype=bool)
    regen_active = (df["P_batt_chem_W"].to_numpy(float) < -1.0) if "P_batt_chem_W" in df.columns else np.zeros(n, dtype=bool)

    candidates: list[tuple[str, int, str]] = [("Start", 0, "Initial frame")]
    eng_idx = np.where(engine_on)[0]
    if eng_idx.size > 0:
        candidates.append(("First engine start", int(eng_idx[0]), "First frame with engine on"))
    regen_idx = np.where(regen_active)[0]
    if regen_idx.size > 0:
        candidates.append(("First regen", int(regen_idx[0]), "First frame with battery charging from regen"))
    candidates.append(("Max speed", int(np.nanargmax(speed)), "Peak vehicle speed"))
    candidates.append(("End", n - 1, "Final frame"))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for label, idx, reason in candidates:
        if label in seen:
            continue
        seen.add(label)
        out.append({"label": label, "idx": int(idx), "t_s": float(t_s[idx]), "reason": reason})
    return out


def _compute_rpm_axis_max(ts: pd.DataFrame) -> float:
    vals: list[float] = []
    for col in ("eng_rpm", "mg1_rpm", "ring_rpm", "mg2_rpm"):
        if col in ts.columns:
            arr = pd.to_numeric(ts[col], errors="coerce").to_numpy(dtype=float)
            vals.extend([abs(v) for v in arr if np.isfinite(v)])
    if not vals:
        return 1000.0
    return _round_up_nice(max(vals))


def _compute_torque_axis_max(ts: pd.DataFrame) -> float:
    vals: list[float] = []
    for col in ("eng_tq_Nm", "mg1_tq_Nm", "T_ring_deliv_Nm", "mg2_tq_Nm"):
        if col in ts.columns:
            arr = pd.to_numeric(ts[col], errors="coerce").to_numpy(dtype=float)
            vals.extend([abs(v) for v in arr if np.isfinite(v)])
    if not vals:
        return 100.0
    return _round_up_nice(max(vals))


def build_viewer_payload(run_dir: Path, ts: pd.DataFrame) -> dict[str, Any]:
    run_dir_abs = run_dir.resolve()
    ratios = _infer_colinear_ratios(ts)
    rpm_axis_max = _compute_rpm_axis_max(ts)
    torque_axis_max = _compute_torque_axis_max(ts)

    frames: list[dict[str, Any]] = []
    for i, row in ts.iterrows():
        veh_spd_mps = _to_num(row.get("veh_spd_mps"))
        frames.append(
            {
                "idx": int(i),
                "time_s": _to_num(row.get("t_s")),
                "mode": str(row.get("mode", "")),
                "engine_on": _to_bool((row.get("eng_rpm", 0.0) or 0.0) > 1.0),
                "fuel_cut": _to_bool(row.get("fuel_cut", False)),
                "regen_active": _to_bool((row.get("P_batt_chem_W", 0.0) or 0.0) < -1.0),
                "vehicle_speed_kph": (veh_spd_mps * 3.6) if veh_spd_mps is not None else None,
                "soc_pct": _to_num(row.get("soc_pct")),
                "eng_rpm": _to_num(row.get("eng_rpm")),
                "mg1_rpm": _to_num(row.get("mg1_rpm")),
                "ring_rpm": _to_num(row.get("ring_rpm")),
                "mg2_rpm": _to_num(row.get("mg2_rpm")),
                "eng_tq_Nm": _to_num(row.get("eng_tq_Nm")),
                "mg1_tq_Nm": _to_num(row.get("mg1_tq_Nm")),
                "T_ring_deliv_Nm": _to_num(row.get("T_ring_deliv_Nm")),
                "mg2_tq_Nm": _to_num(row.get("mg2_tq_Nm")),
                "P_batt_chem_W": _to_num(row.get("P_batt_chem_W")),
                "flag_shortfall": _to_bool((row.get("shortfall_power_W", 0.0) or 0.0) > 1e-6),
                "flag_eng_sat": _to_bool(row.get("flag_eng_sat", False)),
                "flag_mg1_sat": _to_bool(row.get("flag_mg1_sat", False)),
                "flag_mg2_sat": _to_bool(row.get("flag_mg2_sat", False)),
                "flag_batt_sat": _to_bool(row.get("flag_batt_sat", False)),
            }
        )

    duration_s = float(ts["t_s"].iloc[-1]) if "t_s" in ts.columns and len(ts) else 0.0
    max_speed_kph = float(np.nanmax(ts["veh_spd_mps"].to_numpy(float) * 3.6)) if "veh_spd_mps" in ts.columns and len(ts) else 0.0

    return {
        "meta": {"run_dir": run_dir_abs.name, "run_id": run_dir_abs.name},
        "summary": {
            "num_frames": len(frames),
            "duration_s": duration_s,
            "max_speed_kph": max_speed_kph,
            "rpm_axis_max": rpm_axis_max,
            "torque_axis_max": torque_axis_max,
        },
        "ratios": ratios,
        "frames": frames,
        "snapshots": _build_snapshots(ts),
        "events": _mode_events(ts),
    }


def render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>THS Co-linear Viewer</title>
<style>
:root{{color-scheme:light;font-family:Arial,sans-serif;}}
body{{margin:14px;color:#0f172a;background:#f8fafc;}}
.viewer{{max-width:1280px;margin:0 auto;display:grid;grid-template-rows:56px 56px 520px 398px;gap:10px;}}
.panel{{background:#fff;border:1px solid #dbe2ea;border-radius:10px;padding:8px 10px;box-sizing:border-box;overflow:hidden;}}
.status-row{{display:grid;grid-template-columns:140px 180px 1fr;align-items:stretch;height:56px;}}
.status-cell{{min-width:0;padding:4px 10px;border-right:1px solid #e2e8f0;}}
.status-cell:last-child{{border-right:none;}}
.status-label,.summary-label{{font-size:11px;color:#64748b;margin-bottom:2px;}}
.status-value,.summary-value{{font-size:14px;font-weight:600;line-height:1.25;}}
.clamp-2{{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}}
.summary-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;height:56px;align-items:stretch;}}
.summary-card{{min-width:0;padding:4px 10px;}}
.summary-value{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.main-row{{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:10px;height:520px;}}
.diagram-panel{{height:520px;padding:8px 10px 12px 10px;}}
svg{{width:100%;height:500px;background:#fff;display:block;}}
.support-panel{{display:grid;grid-template-rows:110px 1fr;gap:10px;height:520px;}}
.support-panel h4{{font-size:13px;margin:0 0 6px 0;color:#334155;}}
.badge-panel{{height:110px;}}
.badge-grid{{display:flex;flex-wrap:wrap;gap:6px;max-height:64px;overflow:hidden;}}
.badge{{display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;background:#e2e8f0;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:125px;}}
.badge.alert{{background:#fee2e2;color:#991b1b;}}
.badge.ok{{background:#dcfce7;color:#166534;}}
.detail-table{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px;}}
.detail-table th{{width:88px;text-align:left;color:#64748b;font-weight:600;padding:4px 0;vertical-align:top;}}
.detail-table td{{padding:4px 0;color:#0f172a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.bottom-row{{display:grid;grid-template-rows:330px 40px 28px;gap:8px;height:398px;}}
.trend-panel{{display:grid;grid-template-rows:70px 60px 100px 80px;gap:6px;height:330px;padding:8px;}}
.trend-panel canvas{{width:100%;display:block;border:1px solid #e5e7eb;border-radius:6px;background:#fff;}}
.slider-row{{display:grid;grid-template-columns:50px 1fr 120px;align-items:center;gap:10px;height:40px;}}
.slider-readout{{font-size:12px;color:#475569;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.button-row{{display:block;height:28px;overflow:hidden;}}
.button-strip{{display:flex;gap:6px;overflow:hidden;white-space:nowrap;}}
.button-strip button{{height:20px;padding:0 8px;font-size:11px;border:1px solid #cbd5e1;border-radius:999px;background:#f8fafc;cursor:pointer;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.button-strip button.active{{background:#dbeafe;border-color:#60a5fa;color:#1d4ed8;}}
@media (max-width: 1024px) {{
  .viewer{{grid-template-rows:56px 56px 500px 420px;}}
  .main-row{{grid-template-columns:1fr;grid-template-rows:500px 240px;height:auto;}}
  .support-panel{{grid-template-rows:110px 120px;height:auto;}}
  .status-row{{grid-template-columns:120px 150px 1fr;}}
}}
</style>
</head>
<body>
<div class="viewer">
  <div class="panel status-row">
    <div class="status-cell">
      <div class="status-label">Time</div>
      <div id="statusTime" class="status-value"></div>
    </div>
    <div class="status-cell">
      <div class="status-label">Action</div>
      <div id="statusAction" class="status-value"></div>
    </div>
    <div class="status-cell status-explanation">
      <div class="status-label">Explanation</div>
      <div id="statusExplanation" class="status-value clamp-2"></div>
    </div>
  </div>
  <div class="panel summary-row">
    <div class="summary-card"><div class="summary-label">Run</div><div id="sumRun" class="summary-value"></div></div>
    <div class="summary-card"><div class="summary-label">Frames</div><div id="sumFrames" class="summary-value"></div></div>
    <div class="summary-card"><div class="summary-label">Duration</div><div id="sumDuration" class="summary-value"></div></div>
    <div class="summary-card"><div class="summary-label">Max speed</div><div id="sumMaxSpeed" class="summary-value"></div></div>
  </div>
  <div class="main-row">
    <div class="panel diagram-panel"><svg id="diag" viewBox="0 0 1000 500"></svg></div>
    <div class="support-panel">
      <div class="panel badge-panel"><h4>State</h4><div id="badges" class="badge-grid"></div></div>
      <div class="panel detail-panel">
        <h4>Selected frame details</h4>
        <table class="detail-table"><tbody>
          <tr><th>Mode</th><td id="dMode"></td></tr>
          <tr><th>Speed</th><td id="dSpeed"></td></tr>
          <tr><th>SOC</th><td id="dSoc"></td></tr>
          <tr><th>RPM</th><td id="dRpm"></td></tr>
          <tr><th>Torque</th><td id="dTorque"></td></tr>
          <tr><th>Battery</th><td id="dBatt"></td></tr>
          <tr><th>Ratios</th><td id="dRatios"></td></tr>
          <tr><th>Residual</th><td id="dResidual"></td></tr>
        </tbody></table>
      </div>
    </div>
  </div>
  <div class="bottom-row">
    <div class="panel trend-panel">
      <canvas id="speedCanvas" width="1200" height="70"></canvas>
      <canvas id="socCanvas" width="1200" height="60"></canvas>
      <canvas id="rpmCanvas" width="1200" height="100"></canvas>
      <canvas id="powerCanvas" width="1200" height="80"></canvas>
    </div>
    <div class="panel slider-row">
      <label for="timeSlider">Frame</label>
      <input id="timeSlider" type="range" min="0" step="1" />
      <div id="sliderReadout" class="slider-readout"></div>
    </div>
    <div class="panel button-row">
      <div id="snapshots" class="button-strip"></div>
    </div>
  </div>
</div>
<script id="viewer-data" type="application/json">{payload_json}</script>
<script>
const data = JSON.parse(document.getElementById('viewer-data').textContent);
const frames = data.frames || [];
const snapshots = data.snapshots || [];
const events = data.events || [];
let selectedIndex = 0;

const slider = document.getElementById('timeSlider');
const readout = document.getElementById('sliderReadout');
const badges = document.getElementById('badges');
const statusTime = document.getElementById('statusTime');
const statusAction = document.getElementById('statusAction');
const statusExplanation = document.getElementById('statusExplanation');
const sumRun = document.getElementById('sumRun');
const sumFrames = document.getElementById('sumFrames');
const sumDuration = document.getElementById('sumDuration');
const sumMaxSpeed = document.getElementById('sumMaxSpeed');
const dMode = document.getElementById('dMode');
const dSpeed = document.getElementById('dSpeed');
const dSoc = document.getElementById('dSoc');
const dRpm = document.getElementById('dRpm');
const dTorque = document.getElementById('dTorque');
const dBatt = document.getElementById('dBatt');
const dRatios = document.getElementById('dRatios');
const dResidual = document.getElementById('dResidual');
const snapshotsDiv = document.getElementById('snapshots');
const svg = document.getElementById('diag');

const speedCanvas = document.getElementById('speedCanvas');
const socCanvas = document.getElementById('socCanvas');
const rpmCanvas = document.getElementById('rpmCanvas');
const powerCanvas = document.getElementById('powerCanvas');
const speedCtx = speedCanvas.getContext('2d');
const socCtx = socCanvas.getContext('2d');
const rpmCtx = rpmCanvas.getContext('2d');
const powerCtx = powerCanvas.getContext('2d');

function fmt(v, d=1){{ return (v===null || v===undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d); }}
function shortenExplanation(text){{ const maxLen = 44; if (!text) return '-'; return text.length > maxLen ? text.slice(0, maxLen - 1) + '…' : text; }}

function header(){{
  sumRun.textContent = data.meta.run_id;
  sumFrames.textContent = String(data.summary.num_frames);
  sumDuration.textContent = `${{fmt(data.summary.duration_s,1)}} s`;
  sumMaxSpeed.textContent = `${{fmt(data.summary.max_speed_kph,1)}} km/h`;
}}

function makeSnapshots(){{
  snapshotsDiv.innerHTML = '';
  snapshots.forEach((s) => {{
    const btn = document.createElement('button');
    btn.textContent = `${{s.label}} · ${{fmt(s.t_s,1)}}s`;
    btn.title = s.reason;
    btn.onclick = () => setIndex(s.idx);
    btn.dataset.idx = String(s.idx);
    snapshotsDiv.appendChild(btn);
  }});
}}

function computeTrendX(frameIndex, width, padL, padR){{
  const tMin = frames[0]?.time_s ?? 0;
  const tMax = frames[frames.length - 1]?.time_s ?? 1;
  const t = frames[frameIndex]?.time_s ?? frameIndex;
  return padL + (width - padL - padR) * ((t - tMin) / Math.max(1e-9, tMax - tMin));
}}

function drawEventBands(ctx, canvasHeight, padT, padB, padL, padR){{
  ctx.save();
  ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
  events.forEach((ev) => {{
    const x = computeTrendX(ev.idx, ctx.canvas.width, padL, padR);
    ctx.fillRect(x - 3, padT, 6, canvasHeight - padT - padB);
  }});
  ctx.restore();
}}

function drawCurrentMarker(ctx, canvasHeight, padT, padB, padL, padR){{
  const x = computeTrendX(selectedIndex, ctx.canvas.width, padL, padR);
  ctx.save();
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, padT);
  ctx.lineTo(x, canvasHeight - padB);
  ctx.stroke();
  ctx.restore();
}}

function drawSeriesCanvas(ctx, options){{
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const pad = {{l: 46, r: 12, t: 10, b: options.showXAxis ? 18 : 8}};
  ctx.clearRect(0, 0, W, H);
  if (!frames.length) return;

  drawEventBands(ctx, H, pad.t, pad.b, pad.l, pad.r);

  const xs = frames.map((f) => f.time_s ?? f.idx);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const xpix = (x) => pad.l + (W - pad.l - pad.r) * ((x - xmin) / Math.max(1e-9, xmax - xmin));

  let yMin = options.yMin;
  let yMax = options.yMax;
  if (yMin === undefined || yMax === undefined) {{
    const vals = [];
    options.series.forEach((s) => {{
      frames.forEach((f) => {{
        const y = s.value(f);
        if (Number.isFinite(y)) vals.push(y);
      }});
    }});
    if (!vals.length) {{
      yMin = 0;
      yMax = 1;
    }} else {{
      yMin = Math.min(...vals);
      yMax = Math.max(...vals);
      if (options.includeZero) {{
        yMin = Math.min(yMin, 0);
        yMax = Math.max(yMax, 0);
      }}
      const span = Math.max(1e-6, yMax - yMin);
      const margin = span * (options.marginFrac ?? 0.08);
      yMin -= margin;
      yMax += margin;
    }}
  }}
  const ypix = (y) => H - pad.b - (H - pad.t - pad.b) * ((y - yMin) / Math.max(1e-9, yMax - yMin));

  ctx.strokeStyle = '#cbd5e1';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();

  if (options.includeZero && yMin < 0 && yMax > 0) {{
    const y0 = ypix(0);
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(W - pad.r, y0); ctx.stroke();
  }}

  options.series.forEach((s) => {{
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width ?? 1.5;
    ctx.beginPath();
    let started = false;
    frames.forEach((f, i) => {{
      const y = s.value(f);
      if (!Number.isFinite(y)) {{
        started = false;
        return;
      }}
      const x = xpix(xs[i]);
      const yp = ypix(y);
      if (!started) {{
        ctx.moveTo(x, yp);
        started = true;
      }} else {{
        ctx.lineTo(x, yp);
      }}
    }});
    ctx.stroke();
  }});

  drawCurrentMarker(ctx, H, pad.t, pad.b, pad.l, pad.r);

  ctx.fillStyle = '#334155';
  ctx.font = '11px Arial';
  ctx.fillText(options.label, pad.l + 6, pad.t + 11);

  if (options.legend && options.legend.length) {{
    let lx = W - pad.r - 6;
    const ly = pad.t + 11;
    ctx.textAlign = 'right';
    for (let i = options.legend.length - 1; i >= 0; i--) {{
      const item = options.legend[i];
      const textWidth = ctx.measureText(item.text).width;
      lx -= textWidth;
      ctx.fillStyle = item.color;
      ctx.fillText(item.text, lx + textWidth, ly);
      lx -= 14;
    }}
    ctx.textAlign = 'left';
  }}

  ctx.fillStyle = '#64748b';
  ctx.font = '10px Arial';
  ctx.fillText(fmt(yMax, 0), 4, pad.t + 8);
  ctx.fillText(fmt(yMin, 0), 4, H - pad.b);

  if (options.showXAxis) {{
    ctx.fillText('Time [s]', W / 2 - 18, H - 4);
  }}
}}

function drawTrends(){{
  drawSeriesCanvas(speedCtx, {{
    label: 'Vehicle speed [km/h]',
    series: [{{ color: '#2563eb', value: (f) => f.vehicle_speed_kph, width: 1.6 }}],
    marginFrac: 0.10,
    showXAxis: false,
  }});

  drawSeriesCanvas(socCtx, {{
    label: 'SOC [%]',
    series: [{{ color: '#2563eb', value: (f) => f.soc_pct, width: 1.5 }}],
    marginFrac: 0.05,
    showXAxis: false,
  }});

  drawSeriesCanvas(rpmCtx, {{
    label: 'Motor speed [rpm]',
    series: [
      {{ color: '#2563eb', value: (f) => f.eng_rpm, width: 1.4 }},
      {{ color: '#f97316', value: (f) => f.mg1_rpm, width: 1.4 }},
      {{ color: '#16a34a', value: (f) => f.ring_rpm, width: 1.4 }},
      {{ color: '#dc2626', value: (f) => Number.isFinite(f.mg2_rpm) ? -Math.abs(f.mg2_rpm) : null, width: 1.4 }},
    ],
    includeZero: true,
    marginFrac: 0.08,
    showXAxis: false,
    legend: [
      {{ text: 'Ne', color: '#2563eb' }},
      {{ text: 'Ng', color: '#f97316' }},
      {{ text: 'Np', color: '#16a34a' }},
      {{ text: 'Nm', color: '#dc2626' }},
    ],
  }});

  drawSeriesCanvas(powerCtx, {{
    label: 'Battery power [W]',
    series: [{{ color: '#2563eb', value: (f) => f.P_batt_chem_W, width: 1.4 }}],
    includeZero: true,
    marginFrac: 0.08,
    showXAxis: true,
  }});
}}

function computeFrontX(left, splitX, rho){{
  const width = splitX - left;
  return {{ xNg: left, xNe: left + width * (1 / (1 + rho)), xNp: splitX }};
}}

function computeRearX(splitX, right, grm){{
  const width = right - splitX;
  return {{ xNp: splitX, xFix: splitX + width * (1 / (1 + grm)), xNm: right }};
}}

function torqueArrowSvg(x, baseY, torque, scalePx, label){{
  if (!Number.isFinite(torque) || Math.abs(torque) < 1e-6) return '';
  const sign = torque >= 0 ? -1 : 1;
  const len = Math.abs(torque) * scalePx;
  const tipY = baseY + sign * len;
  const shaftEndY = tipY - sign * 8;
  const headHalf = 5;
  const labelY = tipY + (sign < 0 ? -6 : 16);
  return `
    <line x1="${{x}}" y1="${{baseY}}" x2="${{x}}" y2="${{shaftEndY}}" stroke="#111827" stroke-width="2.2" stroke-linecap="round" />
    <polygon points="${{x}},${{tipY}} ${{x - headHalf}},${{tipY - sign * 8}} ${{x + headHalf}},${{tipY - sign * 8}}" fill="#111827" />
    <text x="${{x}}" y="${{labelY}}" font-size="12" text-anchor="middle" fill="#111827">${{label}}</text>`;
}}

function drawDiag(frame){{
  const W = 1000;
  const H = 500;
  const centerY = H * 0.52;
  const left = 90, right = 900, splitX = 470;
  const axisLeft = 56;
  const axisRight = 944;
  const rho = Number.isFinite(data.ratios?.rho) && data.ratios.rho > 0 ? data.ratios.rho : 1.0;
  const grm = Number.isFinite(data.ratios?.grm) && data.ratios.grm > 0 ? data.ratios.grm : 1.0;
  const axisMax = Number.isFinite(data.summary?.rpm_axis_max) && data.summary.rpm_axis_max > 0 ? data.summary.rpm_axis_max : 3000;
  const torqueAxisMax = Number.isFinite(data.summary?.torque_axis_max) && data.summary.torque_axis_max > 0 ? data.summary.torque_axis_max : 100;

  const frontX = computeFrontX(left, splitX, rho);
  const rearX = computeRearX(splitX, right, grm);
  const Ng = frame.mg1_rpm ?? 0;
  const Ne = frame.eng_rpm ?? 0;
  const Np = frame.ring_rpm ?? 0;
  const NmRaw = frame.mg2_rpm ?? 0;
  const NmDisplay = -Math.abs(NmRaw);
  const scaleY = 170 / axisMax;
  const yMap = (v) => centerY - v * scaleY;
  const yNg = yMap(Ng), yNe = yMap(Ne), yNp = yMap(Np), yNm = yMap(NmDisplay), y0 = centerY;
  const Tg = frame.mg1_tq_Nm ?? 0, Te = frame.eng_tq_Nm ?? 0, Tp = frame.T_ring_deliv_Nm ?? 0, Tm = frame.mg2_tq_Nm ?? 0;
  const tqScalePx = 52 / torqueAxisMax;
  const resFront = rho * Ng - (1 + rho) * Ne + Np;
  const resRear = NmDisplay + grm * Np;

  const tickVals = [axisMax, axisMax / 2, 0, -axisMax / 2, -axisMax];
  const tickSvg = tickVals.map((v) => {{
    const y = yMap(v);
    const strong = Math.abs(v) < 1e-9;
    return `
      <line x1="${{axisLeft}}" y1="${{y}}" x2="940" y2="${{y}}" stroke="${{strong ? '#94a3b8' : '#e2e8f0'}}" stroke-width="${{strong ? '1.5' : '1'}}" />
      <line x1="${{axisLeft - 6}}" y1="${{y}}" x2="${{axisLeft}}" y2="${{y}}" stroke="#64748b" stroke-width="1" />
      <text x="${{axisLeft - 10}}" y="${{y + 4}}" font-size="11" text-anchor="end" fill="#64748b">${{Math.round(v)}}</text>`;
  }}).join('');

  const torqueTickVals = [torqueAxisMax, torqueAxisMax / 2, 0, -torqueAxisMax / 2, -torqueAxisMax];
  const torqueTickSvg = torqueTickVals.map((v) => {{
    const y = centerY - v * tqScalePx;
    return `
      <line x1="${{axisRight}}" y1="${{y}}" x2="${{axisRight + 6}}" y2="${{y}}" stroke="#64748b" stroke-width="1" />
      <text x="${{axisRight + 10}}" y="${{y + 4}}" font-size="11" text-anchor="start" fill="#64748b">${{Math.round(v)}}</text>`;
  }}).join('');

  svg.innerHTML = `
    <line x1="${{axisLeft}}" y1="24" x2="${{axisLeft}}" y2="476" stroke="#64748b" stroke-width="1.2" />
    <line x1="${{axisRight}}" y1="${{centerY - 52}}" x2="${{axisRight}}" y2="${{centerY + 52}}" stroke="#64748b" stroke-width="1.2" />
    ${{tickSvg}}
    ${{torqueTickSvg}}
    <text x="18" y="20" font-size="12" fill="#334155">Speed</text>
    <text x="18" y="34" font-size="12" fill="#334155">[rpm]</text>
    <text x="954" y="20" font-size="12" fill="#334155">Torque</text>
    <text x="954" y="34" font-size="12" fill="#334155">[Nm]</text>

    <polyline points="${{frontX.xNg}},${{yNg}} ${{frontX.xNe}},${{yNe}} ${{frontX.xNp}},${{yNp}}" fill="none" stroke="#1d4ed8" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
    <polyline points="${{rearX.xNp}},${{yNp}} ${{rearX.xFix}},${{y0}} ${{rearX.xNm}},${{yNm}}" fill="none" stroke="#15803d" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />

    <circle cx="${{frontX.xNg}}" cy="${{yNg}}" r="4.5" fill="#ffffff" stroke="#93c5fd" stroke-width="1.4" />
    <circle cx="${{frontX.xNe}}" cy="${{yNe}}" r="4.8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.4" />
    <circle cx="${{frontX.xNp}}" cy="${{yNp}}" r="5.0" fill="#ffffff" stroke="#fcd34d" stroke-width="1.4" />
    <circle cx="${{rearX.xNm}}" cy="${{yNm}}" r="4.5" fill="#ffffff" stroke="#86efac" stroke-width="1.4" />

    <circle cx="${{frontX.xNg}}" cy="${{yNg}}" r="5.5" fill="#60a5fa" stroke="#1d4ed8" stroke-width="1.2" />
    <circle cx="${{frontX.xNe}}" cy="${{yNe}}" r="6" fill="#fca5a5" stroke="#b91c1c" stroke-width="1.4" />
    <circle cx="${{frontX.xNp}}" cy="${{yNp}}" r="8.5" fill="#fef08a" stroke="#854d0e" stroke-width="2" />
    <circle cx="${{rearX.xFix}}" cy="${{y0}}" r="17" fill="#0f766e" fill-opacity="0.15" />
    <circle cx="${{rearX.xFix}}" cy="${{y0}}" r="9.5" fill="#0f766e" stroke="#134e4a" stroke-width="2" />
    <circle cx="${{rearX.xNm}}" cy="${{yNm}}" r="5.5" fill="#86efac" stroke="#15803d" stroke-width="1.2" />
    <line x1="${{rearX.xFix}}" y1="${{y0 - 24}}" x2="${{rearX.xFix}}" y2="${{y0 + 24}}" stroke="#0f766e" stroke-width="2" stroke-opacity="0.55" />

    ${{torqueArrowSvg(frontX.xNg, y0, Tg, tqScalePx, 'Tg')}}
    ${{torqueArrowSvg(frontX.xNe, y0, Te, tqScalePx, 'Te')}}
    ${{torqueArrowSvg(frontX.xNp, y0, Tp, tqScalePx, 'Tp')}}
    ${{torqueArrowSvg(rearX.xNm, y0, Tm, tqScalePx, 'Tm')}}

    <text x="${{frontX.xNg-30}}" y="40" font-size="13" fill="#1e40af">Ng (MG1)</text>
    <text x="${{frontX.xNe-30}}" y="40" font-size="13" fill="#991b1b">Ne (Engine)</text>
    <text x="${{frontX.xNp-12}}" y="40" font-size="13" fill="#854d0e">Np</text>
    <text x="${{rearX.xFix-46}}" y="40" font-size="13" fill="#115e59">Rear carrier fixed (0)</text>
    <text x="${{rearX.xNm-45}}" y="40" font-size="13" fill="#166534">Nm (MG2, negative)</text>

    <text x="${{frontX.xNg - 20}}" y="${{centerY + 94}}" font-size="13" fill="#1e3a8a">Front planetary concept: Ng - Ne - Np</text>
    <text x="${{frontX.xNp - 20}}" y="${{centerY + 94}}" font-size="13" fill="#14532d">Rear planetary concept: Np - fixed(0) - Nm</text>

    <text x="64" y="56" font-size="12" fill="#334155">concept lines are primary; circles are supporting raw points; arrows show torque</text>
    <text x="64" y="${{H - 36}}" font-size="12" fill="#475569">RPM axis ±${{Math.round(axisMax)}} · Torque axis ±${{Math.round(torqueAxisMax)}} · rho=${{rho.toFixed(3)}} · grm=${{grm.toFixed(3)}}</text>
    <text x="64" y="${{H - 18}}" font-size="12" fill="#475569">front resid=${{resFront.toFixed(1)}} · rear resid=${{resRear.toFixed(1)}}</text>`;
}}

function classifyAction(frame){{
  const speed = frame.vehicle_speed_kph ?? 0;
  const engOn = !!frame.engine_on;
  const regen = !!frame.regen_active;
  const battW = frame.P_batt_chem_W ?? 0;
  const engTq = frame.eng_tq_Nm ?? 0;
  const mg2Tq = frame.mg2_tq_Nm ?? 0;
  const mg1rpm = Math.abs(frame.mg1_rpm ?? 0);
  if (regen && !engOn) return ['Regen decel', 'eng off · rear fixed · MG2 charge'];
  if (regen && engOn && Math.abs(engTq) < 10) return ['Regen + engine spin', 'regen · engine spin · low fuel tq'];
  if (!engOn && speed > 1 && mg2Tq > 2) return ['EV drive', 'eng off · MG2 drive · front balance'];
  if (engOn && battW < -50) return ['Engine charging', 'engine on · net charge · electrical path'];
  if (engOn && mg2Tq > 2 && engTq > 10) return ['Hybrid drive', 'engine + MG2 · wheel drive'];
  if (engOn && Math.abs(mg2Tq) <= 2 && speed > 1) return ['Engine drive only', 'engine dominant · e-wheel tq low'];
  if (!engOn && speed <= 1 && mg1rpm < 50) return ['Idle/stop', 'vehicle stop · eng off · low speed'];
  return ['Transition', `mode ${{frame.mode || '-'}}`];
}}

function render(){{
  if (!frames.length) return;
  const f = frames[selectedIndex];
  const [action, explanation] = classifyAction(f);
  statusTime.textContent = `${{fmt(f.time_s,1)}} s`;
  statusAction.textContent = action;
  statusExplanation.textContent = shortenExplanation(explanation);
  readout.textContent = `idx=${{selectedIndex}} / ${{frames.length-1}} · t=${{fmt(f.time_s,2)}} s`;
  drawTrends();
  drawDiag(f);

  const badgeItems = [
    `<span class="badge">${{action}}</span>`,
    `<span class="badge ${{f.engine_on ? 'ok' : ''}}">engine_on:${{f.engine_on}}</span>`,
    `<span class="badge">fuel_cut:${{f.fuel_cut}}</span>`,
    `<span class="badge ${{f.regen_active ? 'ok' : ''}}">regen:${{f.regen_active}}</span>`,
    f.flag_shortfall ? '<span class="badge alert">shortfall</span>' : '',
    f.flag_eng_sat ? '<span class="badge alert">eng_sat</span>' : '',
    f.flag_mg1_sat ? '<span class="badge alert">mg1_sat</span>' : '',
    f.flag_mg2_sat ? '<span class="badge alert">mg2_sat</span>' : '',
    f.flag_batt_sat ? '<span class="badge alert">batt_sat</span>' : ''
  ].filter(Boolean);
  badges.innerHTML = badgeItems.slice(0, 6).join(' ');

  const rho = Number.isFinite(data.ratios?.rho) && data.ratios.rho > 0 ? data.ratios.rho : 1.0;
  const grm = Number.isFinite(data.ratios?.grm) && data.ratios.grm > 0 ? data.ratios.grm : 1.0;
  const Ng = f.mg1_rpm ?? 0;
  const Ne = f.eng_rpm ?? 0;
  const Np = f.ring_rpm ?? 0;
  const NmDisplay = -Math.abs(f.mg2_rpm ?? 0);
  const resFront = rho * Ng - (1 + rho) * Ne + Np;
  const resRear = NmDisplay + grm * Np;

  dMode.textContent = f.mode ?? '-';
  dSpeed.textContent = `${{fmt(f.vehicle_speed_kph,1)}} km/h`;
  dSoc.textContent = `${{fmt(f.soc_pct,2)}} %`;
  dRpm.textContent = `eng ${{fmt(f.eng_rpm,0)}} / mg1 ${{fmt(f.mg1_rpm,0)}} / ring ${{fmt(f.ring_rpm,0)}} / mg2 ${{fmt(NmDisplay,0)}}`;
  dTorque.textContent = `Tg ${{fmt(f.mg1_tq_Nm,1)}} / Te ${{fmt(f.eng_tq_Nm,1)}} / Tp ${{fmt(f.T_ring_deliv_Nm,1)}} / Tm ${{fmt(f.mg2_tq_Nm,1)}}`;
  dBatt.textContent = `${{fmt(f.P_batt_chem_W,0)}} W`;
  dRatios.textContent = `rho ${{rho.toFixed(3)}} / grm ${{grm.toFixed(3)}}`;
  dResidual.textContent = `F ${{resFront.toFixed(1)}} / R ${{resRear.toFixed(1)}}`;

  [...snapshotsDiv.querySelectorAll('button')].forEach((b)=>{{
    b.classList.toggle('active', Number(b.dataset.idx) === selectedIndex);
  }});
}}

function setIndex(i){{
  selectedIndex = Math.max(0, Math.min(frames.length - 1, Number(i)));
  slider.value = String(selectedIndex);
  render();
}}

header();
makeSnapshots();
slider.max = String(Math.max(0, frames.length - 1));
slider.value = '0';
slider.oninput = (e) => setIndex(e.target.value);
render();
</script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render self-contained single-run co-linear viewer HTML")
    p.add_argument("--run_dir", type=Path, required=True, help="Run directory")
    p.add_argument("--out_html", type=Path, default=None, help="Optional output HTML path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ts_path = _find_timeseries_csv(run_dir)
    ts = pd.read_csv(ts_path)
    payload = build_viewer_payload(run_dir=run_dir, ts=ts)
    html = render_html(payload)
    out_html = args.out_html if args.out_html is not None else run_dir / "Viz" / "colinear_viewer.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] wrote viewer: {out_html}")
    print(f"[INFO] frames={payload['summary']['num_frames']} snapshots={len(payload['snapshots'])}")


if __name__ == "__main__":
    main()
