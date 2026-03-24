from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_timeseries_csv(run_dir: Path) -> Path:
    signals_dir = run_dir / "signals"
    if not signals_dir.exists():
        raise FileNotFoundError(f"signals directory not found under run_dir: {signals_dir}")

    derived = sorted(signals_dir.glob("timeseries_phaseB_derived_*.csv"))
    if derived:
        return derived[-1]

    canonical = sorted(signals_dir.glob("timeseries_phaseB_*.csv"))
    canonical = [p for p in canonical if "derived" not in p.name]
    if canonical:
        return canonical[-1]

    raise FileNotFoundError(f"No timeseries_phaseB*.csv found in: {signals_dir}")


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


def _mode_events(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "mode" not in df.columns or "t_s" not in df.columns:
        return []

    out: list[dict[str, Any]] = []
    mode = df["mode"].astype(str).to_numpy()
    t_s = df["t_s"].to_numpy(float)
    if len(mode) == 0:
        return out

    out.append({"idx": 0, "t_s": float(t_s[0]), "kind": "mode", "value": mode[0]})
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
    engine_on = (
        df["eng_rpm"].to_numpy(float) > 1.0
        if "eng_rpm" in df.columns
        else np.zeros(n, dtype=bool)
    )
    regen_active = (
        df["P_batt_chem_W"].to_numpy(float) < -1.0
        if "P_batt_chem_W" in df.columns
        else np.zeros(n, dtype=bool)
    )

    candidates: list[tuple[str, int, str]] = []
    candidates.append(("Start", 0, "Initial frame"))

    eng_idx = np.where(engine_on)[0]
    if eng_idx.size > 0:
        i = int(eng_idx[0])
        candidates.append(("First engine start", i, "First frame with engine on"))

    regen_idx = np.where(regen_active)[0]
    if regen_idx.size > 0:
        i = int(regen_idx[0])
        candidates.append(("First regen", i, "First frame with battery charging from regen"))

    if n > 0:
        i = int(np.nanargmax(speed))
        candidates.append(("Max speed", i, "Peak vehicle speed"))

    candidates.append(("End", n - 1, "Final frame"))

    snapshots: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for label, idx, reason in candidates:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        snapshots.append(
            {
                "label": label,
                "idx": int(idx),
                "t_s": float(t_s[idx]),
                "reason": reason,
            }
        )
    return snapshots


def build_viewer_payload(run_dir: Path, ts: pd.DataFrame) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames: list[dict[str, Any]] = []
    for i, row in ts.iterrows():
        frame = {
            "idx": int(i),
            "time_s": _to_num(row.get("t_s")),
            "mode": str(row.get("mode", "")),
            "engine_on": _to_bool(row.get("eng_rpm", 0.0) > 1.0),
            "fuel_cut": _to_bool(row.get("fuel_cut", False)),
            "regen_active": _to_bool((row.get("P_batt_chem_W", 0.0) or 0.0) < -1.0),
            "vehicle_speed_kph": _to_num((row.get("veh_spd_mps", np.nan) or np.nan) * 3.6),
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
            "flag_shortfall": _to_bool(row.get("shortfall_power_W", 0.0) > 1e-6),
            "flag_eng_sat": _to_bool(row.get("flag_eng_sat", False)),
            "flag_mg1_sat": _to_bool(row.get("flag_mg1_sat", False)),
            "flag_mg2_sat": _to_bool(row.get("flag_mg2_sat", False)),
            "flag_batt_sat": _to_bool(row.get("flag_batt_sat", False)),
        }
        frames.append(frame)

    duration_s = float(ts["t_s"].iloc[-1]) if "t_s" in ts.columns and len(ts) else 0.0
    max_speed_kph = float(np.nanmax(ts["veh_spd_mps"].to_numpy(float) * 3.6)) if "veh_spd_mps" in ts.columns and len(ts) else 0.0

    return {
        "meta": {
            "run_dir": str(run_dir),
            "run_id": manifest.get("run_id", run_dir.name),
            "source_timeseries": manifest.get("files", {}).get("timeseries", ""),
        },
        "summary": {
            "num_frames": len(frames),
            "duration_s": duration_s,
            "max_speed_kph": max_speed_kph,
        },
        "frames": frames,
        "snapshots": _build_snapshots(ts),
        "events": _mode_events(ts),
    }


def render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>THS Co-linear Viewer</title>
<style>
body{{font-family:Arial,sans-serif;margin:16px;color:#1f2937;background:#f8fafc;}}
.panel{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:12px;}}
.row{{display:grid;grid-template-columns:1fr 320px;gap:12px;}}
#snapshots button{{margin:4px;padding:6px 8px;border:1px solid #d1d5db;border-radius:6px;background:#f9fafb;cursor:pointer;}}
#snapshots button.active{{background:#dbeafe;border-color:#60a5fa;}}
.meta{{font-size:12px;color:#475569;line-height:1.5;}}
svg{{width:100%;height:340px;background:#fff;}}
canvas{{width:100%;height:220px;border:1px solid #e5e7eb;border-radius:6px;}}
.badge{{display:inline-block;padding:2px 6px;border-radius:999px;font-size:11px;margin-right:4px;background:#e2e8f0;}}
.badge.alert{{background:#fee2e2;color:#991b1b;}}
.badge.ok{{background:#dcfce7;color:#166534;}}
.controls{{display:flex;gap:8px;align-items:center;}}
</style>
</head>
<body>
<div class=\"panel\">
  <h2 style=\"margin:0 0 8px 0\">Single-run Co-linear Viewer</h2>
  <div id=\"summary\" class=\"meta\"></div>
</div>
<div class=\"panel\">
  <h3 style=\"margin-top:0\">Time series</h3>
  <canvas id=\"tsCanvas\" width=\"1000\" height=\"220\"></canvas>
</div>
<div class=\"panel controls\">
  <label for=\"timeSlider\">Selected frame</label>
  <input id=\"timeSlider\" type=\"range\" min=\"0\" step=\"1\" style=\"flex:1\" />
  <div id=\"sliderReadout\" class=\"meta\"></div>
</div>
<div class=\"row\">
  <div class=\"panel\">
    <h3 style=\"margin-top:0\">Main co-linear diagram</h3>
    <svg id=\"diag\" viewBox=\"0 0 800 340\"></svg>
  </div>
  <div class=\"panel\">
    <h3 style=\"margin-top:0\">Snapshots</h3>
    <div id=\"snapshots\"></div>
    <h3>Frame details</h3>
    <div id=\"details\" class=\"meta\"></div>
  </div>
</div>
<script id=\"viewer-data\" type=\"application/json\">{payload_json}</script>
<script>
const data = JSON.parse(document.getElementById('viewer-data').textContent);
const frames = data.frames || [];
const snapshots = data.snapshots || [];
let selectedIndex = 0;

const slider = document.getElementById('timeSlider');
const readout = document.getElementById('sliderReadout');
const details = document.getElementById('details');
const summary = document.getElementById('summary');
const snapshotsDiv = document.getElementById('snapshots');
const canvas = document.getElementById('tsCanvas');
const ctx = canvas.getContext('2d');
const svg = document.getElementById('diag');

function fmt(v, d=1){{ return (v===null || v===undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d); }}

function header(){{
  summary.innerHTML = `Run: <b>${{data.meta.run_id}}</b> · Frames: ${{data.summary.num_frames}} · Duration: ${{fmt(data.summary.duration_s,1)}} s · Max speed: ${{fmt(data.summary.max_speed_kph,1)}} km/h`;
}}

function makeSnapshots(){{
  snapshotsDiv.innerHTML = '';
  snapshots.forEach((s) => {{
    const btn = document.createElement('button');
    btn.textContent = `${{s.label}} @ ${{fmt(s.t_s,1)}} s`;
    btn.title = s.reason;
    btn.onclick = () => setIndex(s.idx);
    btn.dataset.idx = String(s.idx);
    snapshotsDiv.appendChild(btn);
  }});
}}

function drawTimeSeries(){{
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  if (!frames.length) return;
  const pad = {{l:50,r:16,t:12,b:26}};
  const xs = frames.map(f => f.time_s ?? f.idx);
  const ys = frames.map(f => f.vehicle_speed_kph ?? 0);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(1, ...ys);
  const xpix = (x) => pad.l + (W - pad.l - pad.r) * ((x - xmin) / Math.max(1e-9, xmax - xmin));
  const ypix = (y) => H - pad.b - (H - pad.t - pad.b) * (y / ymax);

  ctx.strokeStyle = '#111827';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ys.forEach((y,i)=>{{
    const x = xpix(xs[i]), yp = ypix(y);
    if (i===0) ctx.moveTo(x,yp); else ctx.lineTo(x,yp);
  }});
  ctx.stroke();

  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();

  const sx = xpix(xs[selectedIndex]);
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(sx,pad.t); ctx.lineTo(sx,H-pad.b); ctx.stroke();

  ctx.fillStyle = '#334155';
  ctx.font = '12px Arial';
  ctx.fillText('vehicle speed (km/h)', pad.l + 8, pad.t + 12);
}}

function drawDiag(frame){{
  const W=800, H=340;
  const centerY = H/2;
  const xFront = [120, 280, 440];
  const xRear = [120, 280, 440];
  const yVals = [frame.mg1_rpm ?? 0, frame.eng_rpm ?? 0, frame.ring_rpm ?? 0, frame.mg2_rpm ?? 0, 0];
  const maxAbs = Math.max(500, ...yVals.map(v => Math.abs(v)));
  const yMap = (v) => centerY - (v / maxAbs) * 120;

  const yNg = yMap(frame.mg1_rpm ?? 0);
  const yNe = yMap(frame.eng_rpm ?? 0);
  const yNp = yMap(frame.ring_rpm ?? 0);
  const yNm = yMap(frame.mg2_rpm ?? 0);
  const y0 = yMap(0);

  svg.innerHTML = `
    <line x1="50" y1="${{centerY}}" x2="760" y2="${{centerY}}" stroke="#cbd5e1" stroke-width="1" />
    <line x1="${{xFront[0]}}" y1="${{yNg}}" x2="${{xFront[2]}}" y2="${{yNp}}" stroke="#2563eb" stroke-width="2" />
    <line x1="${{xRear[0]+280}}" y1="${{yNp}}" x2="${{xRear[2]+280}}" y2="${{yNm}}" stroke="#16a34a" stroke-width="2" />
    <circle cx="${{xFront[0]}}" cy="${{yNg}}" r="6" fill="#2563eb" />
    <circle cx="${{xFront[1]}}" cy="${{yNe}}" r="6" fill="#ef4444" />
    <circle cx="${{xFront[2]}}" cy="${{yNp}}" r="6" fill="#2563eb" />
    <circle cx="${{xRear[0]+280}}" cy="${{yNp}}" r="6" fill="#16a34a" />
    <circle cx="${{xRear[1]+280}}" cy="${{y0}}" r="6" fill="#64748b" />
    <circle cx="${{xRear[2]+280}}" cy="${{yNm}}" r="6" fill="#16a34a" />
    <text x="${{xFront[0]-16}}" y="24" font-size="12">Ng (MG1)</text>
    <text x="${{xFront[1]-16}}" y="24" font-size="12">Ne (ENG)</text>
    <text x="${{xFront[2]-16}}" y="24" font-size="12">Np (Ring)</text>
    <text x="${{xRear[0]+265}}" y="24" font-size="12">Np</text>
    <text x="${{xRear[1]+265}}" y="24" font-size="12">0</text>
    <text x="${{xRear[2]+265}}" y="24" font-size="12">Nm (MG2)</text>
    <text x="100" y="320" font-size="12" fill="#1e293b">Front planetary concept: Ng-Ne-Np</text>
    <text x="430" y="320" font-size="12" fill="#1e293b">Rear planetary concept: Np-0-Nm</text>
  `;
}}

function render(){{
  if (!frames.length) return;
  const f = frames[selectedIndex];
  readout.textContent = `idx=${{selectedIndex}} / ${{frames.length-1}}, t=${{fmt(f.time_s,2)}} s`;
  drawTimeSeries();
  drawDiag(f);

  const badges = [
    `<span class="badge ${{f.engine_on ? 'ok' : ''}}">engine_on:${{f.engine_on}}</span>`,
    `<span class="badge">fuel_cut:${{f.fuel_cut}}</span>`,
    `<span class="badge ${{f.regen_active ? 'ok' : ''}}">regen:${{f.regen_active}}</span>`,
    f.flag_shortfall ? '<span class="badge alert">shortfall</span>' : '',
    (f.flag_eng_sat||f.flag_mg1_sat||f.flag_mg2_sat||f.flag_batt_sat) ? '<span class="badge alert">saturation</span>' : ''
  ].join(' ');

  details.innerHTML = `
    ${{badges}}<br/>
    mode=${{f.mode}}<br/>
    speed=${{fmt(f.vehicle_speed_kph,1)}} km/h · soc=${{fmt(f.soc_pct,2)}} %<br/>
    RPM: eng=${{fmt(f.eng_rpm,0)}}, mg1=${{fmt(f.mg1_rpm,0)}}, ring=${{fmt(f.ring_rpm,0)}}, mg2=${{fmt(f.mg2_rpm,0)}}<br/>
    Torque (Nm): eng=${{fmt(f.eng_tq_Nm,1)}}, mg1=${{fmt(f.mg1_tq_Nm,1)}}, ring=${{fmt(f.T_ring_deliv_Nm,1)}}, mg2=${{fmt(f.mg2_tq_Nm,1)}}<br/>
    P_batt_chem=${{fmt(f.P_batt_chem_W,0)}} W
  `;

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
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render self-contained single-run co-linear viewer HTML")
    p.add_argument("--run_dir", type=Path, required=True, help="Run directory (contains signals/, Viz/, manifest.json)")
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
