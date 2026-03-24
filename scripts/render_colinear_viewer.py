from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_timeseries_csv(run_dir: Path) -> Path:
    signals_dir = run_dir
    if not signals_dir.exists():
        raise FileNotFoundError(f"signals directory not found under run_dir: {signals_dir}")

    derived = sorted(signals_dir.glob("timeseries*.csv"))
    if derived:
        return derived[-1]

    canonical = sorted(signals_dir.glob("timeseries*.csv"))
    canonical = [p for p in canonical if "derived" not in p.name]
    if canonical:
        return canonical[-1]

    raise FileNotFoundError(f"No timeseries.csv found in: {signals_dir}")


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
    run_dir_abs = run_dir.resolve()
    manifest_path = run_dir_abs / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames: list[dict[str, Any]] = []
    for i, row in ts.iterrows():
        veh_spd_mps = _to_num(row.get("veh_spd_mps"))
        frame = {
            "idx": int(i),
            "time_s": _to_num(row.get("t_s")),
            "mode": str(row.get("mode", "")),
            "engine_on": _to_bool(row.get("eng_rpm", 0.0) > 1.0),
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
            "run_dir": run_dir_abs.name,
            "run_id": manifest.get("run_id", run_dir_abs.name),
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
:root{{color-scheme:light;font-family:Arial,sans-serif;}}
body{{margin:14px;color:#0f172a;background:#f8fafc;}}
.viewer{{max-width:1280px;margin:0 auto;}}
.panel{{background:#fff;border:1px solid #dbe2ea;border-radius:10px;padding:10px 12px;}}
.status{{margin-bottom:10px;display:flex;justify-content:space-between;gap:8px;align-items:baseline;}}
.status-main{{font-size:16px;font-weight:700;line-height:1.3;}}
.status-sub{{font-size:12px;color:#475569;text-align:right;}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:10px;align-items:stretch;}}
.diagram-wrap{{padding:8px 10px 12px 10px;}}
svg{{width:100%;height:500px;background:#fff;display:block;}}
.support{{display:flex;flex-direction:column;gap:10px;}}
.support h4{{font-size:13px;margin:0 0 6px 0;color:#334155;}}
.meta{{font-size:12px;color:#475569;line-height:1.45;}}
.badge{{display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;margin:2px 4px 2px 0;background:#e2e8f0;color:#334155;}}
.badge.alert{{background:#fee2e2;color:#991b1b;}}
.badge.ok{{background:#dcfce7;color:#166534;}}
.timeline{{margin-top:10px;display:grid;grid-template-rows:auto auto auto;gap:8px;}}
canvas{{width:100%;height:96px;border:1px solid #e5e7eb;border-radius:8px;background:#fff;}}
.controls{{display:flex;gap:10px;align-items:center;}}
.controls label{{font-size:12px;color:#334155;white-space:nowrap;}}
.controls input[type="range"]{{flex:1;}}
.button-strip{{display:flex;gap:6px;flex-wrap:wrap;}}
.button-strip button{{padding:4px 8px;border:1px solid #cbd5e1;border-radius:999px;background:#f8fafc;cursor:pointer;font-size:11px;color:#334155;}}
.button-strip button.active{{background:#dbeafe;border-color:#60a5fa;color:#1d4ed8;}}
.button-strip button.event{{background:#f1f5f9;}}
@media (max-width: 1024px) {{
  .layout{{grid-template-columns:1fr;}}
  .status{{flex-direction:column;align-items:flex-start;}}
  .status-sub{{text-align:left;}}
  svg{{height:440px;}}
}}
</style>
</head>
<body>
<div class=\"viewer\">
  <div class=\"panel status\">
    <div id=\"statusLine\" class=\"status-main\"></div>
    <div id=\"summary\" class=\"status-sub\"></div>
  </div>
  <div class=\"layout\">
    <div class=\"panel diagram-wrap\">
      <svg id=\"diag\" viewBox=\"0 0 1000 500\"></svg>
    </div>
    <div class=\"support\">
      <div class=\"panel\">
        <h4>State badges</h4>
        <div id=\"badges\"></div>
      </div>
      <div class=\"panel\">
        <h4>Selected frame details</h4>
        <div id=\"details\" class=\"meta\"></div>
      </div>
    </div>
  </div>
  <div class=\"timeline\">
    <canvas id=\"tsCanvas\" width=\"1200\" height=\"96\"></canvas>
    <div class=\"panel controls\">
      <label for=\"timeSlider\">Frame</label>
      <input id=\"timeSlider\" type=\"range\" min=\"0\" step=\"1\" />
      <div id=\"sliderReadout\" class=\"meta\"></div>
    </div>
    <div class=\"panel\">
      <div class=\"button-strip\" id=\"snapshots\"></div>
      <div class=\"button-strip\" id=\"events\" style=\"margin-top:6px\"></div>
    </div>
  </div>
</div>
<script id=\"viewer-data\" type=\"application/json\">{payload_json}</script>
<script>
const data = JSON.parse(document.getElementById('viewer-data').textContent);
const frames = data.frames || [];
const snapshots = data.snapshots || [];
const events = data.events || [];
let selectedIndex = 0;

const slider = document.getElementById('timeSlider');
const readout = document.getElementById('sliderReadout');
const details = document.getElementById('details');
const badges = document.getElementById('badges');
const summary = document.getElementById('summary');
const statusLine = document.getElementById('statusLine');
const snapshotsDiv = document.getElementById('snapshots');
const eventsDiv = document.getElementById('events');
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
    btn.textContent = `${{s.label}} · ${{fmt(s.t_s,1)}}s`;
    btn.title = s.reason;
    btn.onclick = () => setIndex(s.idx);
    btn.dataset.idx = String(s.idx);
    snapshotsDiv.appendChild(btn);
  }});
}}

function makeEvents(){{
  eventsDiv.innerHTML = '';
  events.forEach((ev) => {{
    const btn = document.createElement('button');
    btn.className = 'event';
    btn.textContent = `${{ev.value}} @ ${{fmt(ev.t_s,1)}}s`;
    btn.title = 'Mode transition';
    btn.onclick = () => setIndex(ev.idx);
    btn.dataset.idx = String(ev.idx);
    eventsDiv.appendChild(btn);
  }});
}}

function drawTimeSeries(){{
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  if (!frames.length) return;
  const pad = {{l:40,r:14,t:10,b:20}};
  const xs = frames.map(f => f.time_s ?? f.idx);
  const ys = frames.map(f => f.vehicle_speed_kph);
  const validYs = ys.filter((y) => Number.isFinite(y));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(1, ...(validYs.length ? validYs : [0]));
  const xpix = (x) => pad.l + (W - pad.l - pad.r) * ((x - xmin) / Math.max(1e-9, xmax - xmin));
  const ypix = (y) => H - pad.b - (H - pad.t - pad.b) * (y / ymax);

  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  let started = false;
  ys.forEach((y,i)=>{{
    if (!Number.isFinite(y)) {{
      started = false;
      return;
    }}
    const x = xpix(xs[i]), yp = ypix(y);
    if (!started) {{
      ctx.moveTo(x,yp);
      started = true;
    }} else {{
      ctx.lineTo(x,yp);
    }}
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
  ctx.font = '11px Arial';
  ctx.fillText('Vehicle speed (km/h)', pad.l + 6, pad.t + 10);
}}

function drawDiag(frame){{
  const W=1000, H=500;
  const centerY = H*0.52;
  const left = 110;
  const spacing = 160;
  const xFront = [left, left + spacing, left + spacing * 2];
  const rearStart = left + spacing * 3;
  const xRear = [rearStart, rearStart + spacing, rearStart + spacing * 2];
  const yVals = [frame.mg1_rpm ?? 0, frame.eng_rpm ?? 0, frame.ring_rpm ?? 0, frame.mg2_rpm ?? 0, 0];
  const maxAbs = Math.max(500, ...yVals.map(v => Math.abs(v)));
  const yMap = (v) => centerY - (v / maxAbs) * 170;

  const yNg = yMap(frame.mg1_rpm ?? 0);
  const yNe = yMap(frame.eng_rpm ?? 0);
  const yNp = yMap(frame.ring_rpm ?? 0);
  const yNm = yMap(frame.mg2_rpm ?? 0);
  const y0 = yMap(0);

  svg.innerHTML = `
    <line x1="60" y1="${{centerY}}" x2="940" y2="${{centerY}}" stroke="#d1d5db" stroke-width="1.5" />
    <polyline points="${{xFront[0]}},${{yNg}} ${{xFront[1]}},${{yNe}} ${{xFront[2]}},${{yNp}}" fill="none" stroke="#1d4ed8" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
    <polyline points="${{xRear[0]}},${{yNp}} ${{xRear[1]}},${{y0}} ${{xRear[2]}},${{yNm}}" fill="none" stroke="#15803d" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />

    <line x1="${{xFront[0]}}" y1="${{centerY + 70}}" x2="${{xFront[2]}}" y2="${{centerY + 70}}" stroke="#bfdbfe" stroke-width="1.5" stroke-dasharray="5 4" />
    <line x1="${{xRear[0]}}" y1="${{centerY + 70}}" x2="${{xRear[2]}}" y2="${{centerY + 70}}" stroke="#bbf7d0" stroke-width="1.5" stroke-dasharray="5 4" />
    <circle cx="${{xFront[0]}}" cy="${{yNg}}" r="5.5" fill="#60a5fa" stroke="#1d4ed8" stroke-width="1.5" />
    <circle cx="${{xFront[1]}}" cy="${{yNe}}" r="6" fill="#fca5a5" stroke="#b91c1c" stroke-width="1.5" />
    <circle cx="${{xFront[2]}}" cy="${{yNp}}" r="5.5" fill="#60a5fa" stroke="#1d4ed8" stroke-width="1.5" />
    <circle cx="${{xRear[0]}}" cy="${{yNp}}" r="5.5" fill="#86efac" stroke="#15803d" stroke-width="1.5" />
    <circle cx="${{xRear[1]}}" cy="${{y0}}" r="17" fill="#0f766e" fill-opacity="0.15" />
    <circle cx="${{xRear[1]}}" cy="${{y0}}" r="9.5" fill="#0f766e" stroke="#134e4a" stroke-width="2" />
    <circle cx="${{xRear[2]}}" cy="${{yNm}}" r="5.5" fill="#86efac" stroke="#15803d" stroke-width="1.5" />
    <line x1="${{xRear[1]}}" y1="${{y0 - 24}}" x2="${{xRear[1]}}" y2="${{y0 + 24}}" stroke="#0f766e" stroke-width="2" stroke-opacity="0.55" />

    <text x="${{xFront[0]-30}}" y="${{centerY - 205}}" font-size="13" fill="#1e40af">Ng (MG1)</text>
    <text x="${{xFront[1]-30}}" y="${{centerY - 205}}" font-size="13" fill="#991b1b">Ne (Engine)</text>
    <text x="${{xFront[2]-30}}" y="${{centerY - 205}}" font-size="13" fill="#1e40af">Np (Ring)</text>
    <text x="${{xRear[0]-16}}" y="${{centerY - 205}}" font-size="13" fill="#166534">Np</text>
    <text x="${{xRear[1]-46}}" y="${{centerY - 205}}" font-size="13" fill="#115e59">Rear carrier fixed (0)</text>
    <text x="${{xRear[2]-25}}" y="${{centerY - 205}}" font-size="13" fill="#166534">Nm (MG2)</text>

    <text x="${{xFront[0] - 20}}" y="${{centerY + 94}}" font-size="13" fill="#1e3a8a">Front planetary concept: Ng - Ne - Np</text>
    <text x="${{xRear[0] - 20}}" y="${{centerY + 94}}" font-size="13" fill="#14532d">Rear planetary concept: Np - fixed(0) - Nm</text>

    <text x="64" y="34" font-size="12" fill="#334155">concept lines are primary; circles are supporting raw points</text>
    <text x="64" y="${{H - 22}}" font-size="12" fill="#475569">RPM scale ±${{Math.round(maxAbs)}}</text>
  `;
}}

function classifyAction(frame){{
  const speed = frame.vehicle_speed_kph ?? 0;
  const engOn = !!frame.engine_on;
  const regen = !!frame.regen_active;
  const battW = frame.P_batt_chem_W ?? 0;
  const engTq = frame.eng_tq_Nm ?? 0;
  const mg2Tq = frame.mg2_tq_Nm ?? 0;
  const mg1rpm = Math.abs(frame.mg1_rpm ?? 0);

  if (regen && !engOn) return ['Regen decel', 'Engine stopped, rear carrier fixed, MG2 charging battery'];
  if (regen && engOn && Math.abs(engTq) < 10) return ['Regen with engine spin', 'Vehicle decelerating while engine spins with low fuel torque'];
  if (!engOn && speed > 1 && mg2Tq > 2) return ['EV drive', 'Engine stopped, MG2 propels vehicle and front planetary balances speed'];
  if (engOn && battW < -50) return ['Engine charging', 'Engine running and electrical path net-charging battery'];
  if (engOn && mg2Tq > 2 && engTq > 10) return ['Hybrid drive', 'Engine and MG2 both contribute wheel-side torque'];
  if (engOn && Math.abs(mg2Tq) <= 2 && speed > 1) return ['Engine drive only', 'Engine dominates propulsion while electric wheel torque is near zero'];
  if (!engOn && speed <= 1 && mg1rpm < 50) return ['Idle/stop', 'Vehicle near standstill with engine off and low machine speeds'];
  return ['Transition', `System transitioning (mode=${{frame.mode || '-'}})`];
}}

function render(){{
  if (!frames.length) return;
  const f = frames[selectedIndex];
  const [action, explanation] = classifyAction(f);
  statusLine.textContent = `${{fmt(f.time_s,1)}} s | ${{action}} | ${{explanation}}`;
  readout.textContent = `idx=${{selectedIndex}} / ${{frames.length-1}} · t=${{fmt(f.time_s,2)}} s`;
  drawTimeSeries();
  drawDiag(f);

  const badgeHtml = [
    `<span class="badge">${{action}}</span>`,
    `<span class="badge ${{f.engine_on ? 'ok' : ''}}">engine_on:${{f.engine_on}}</span>`,
    `<span class="badge">fuel_cut:${{f.fuel_cut}}</span>`,
    `<span class="badge ${{f.regen_active ? 'ok' : ''}}">regen:${{f.regen_active}}</span>`,
    f.flag_shortfall ? '<span class="badge alert">shortfall</span>' : '',
    (f.flag_eng_sat||f.flag_mg1_sat||f.flag_mg2_sat||f.flag_batt_sat) ? '<span class="badge alert">saturation</span>' : ''
  ].join(' ');
  badges.innerHTML = badgeHtml;

  details.innerHTML = `
    mode=${{f.mode}}<br/>
    speed=${{fmt(f.vehicle_speed_kph,1)}} km/h · soc=${{fmt(f.soc_pct,2)}} %<br/>
    RPM: eng=${{fmt(f.eng_rpm,0)}}, mg1=${{fmt(f.mg1_rpm,0)}}, ring=${{fmt(f.ring_rpm,0)}}, mg2=${{fmt(f.mg2_rpm,0)}}<br/>
    Torque (Nm): eng=${{fmt(f.eng_tq_Nm,1)}}, mg1=${{fmt(f.mg1_tq_Nm,1)}}, ring=${{fmt(f.T_ring_deliv_Nm,1)}}, mg2=${{fmt(f.mg2_tq_Nm,1)}}<br/>
    P_batt_chem=${{fmt(f.P_batt_chem_W,0)}} W
  `;

  [...snapshotsDiv.querySelectorAll('button'), ...eventsDiv.querySelectorAll('button')].forEach((b)=>{{
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
makeEvents();
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
