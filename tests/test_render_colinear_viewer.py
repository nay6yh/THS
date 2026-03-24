from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.render_colinear_viewer import _build_snapshots, build_viewer_payload


def _sample_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "mode": ["EV", "EV", "HybridDrive", "Regen", "EV"],
            "veh_spd_mps": [0.0, 3.0, 8.0, 5.0, 1.0],
            "eng_rpm": [0.0, 0.0, 1100.0, 900.0, 0.0],
            "mg1_rpm": [200.0, 300.0, 400.0, 350.0, 100.0],
            "ring_rpm": [0.0, 100.0, 220.0, 180.0, 30.0],
            "mg2_rpm": [0.0, 140.0, 260.0, 210.0, 20.0],
            "fuel_cut": [1, 1, 0, 1, 1],
            "soc_pct": [60.0, 59.9, 59.7, 59.8, 59.8],
            "eng_tq_Nm": [0.0, 0.0, 90.0, 70.0, 0.0],
            "mg1_tq_Nm": [10.0, 11.0, -20.0, -18.0, 8.0],
            "T_ring_deliv_Nm": [50.0, 60.0, 80.0, -25.0, 10.0],
            "mg2_tq_Nm": [45.0, 54.0, 72.0, -15.0, 9.0],
            "P_batt_chem_W": [3000.0, 2500.0, 2000.0, -1500.0, 500.0],
            "shortfall_power_W": [0.0, 0.0, 0.0, 0.0, 0.0],
            "flag_eng_sat": [0, 0, 0, 0, 0],
            "flag_mg1_sat": [0, 0, 0, 0, 0],
            "flag_mg2_sat": [0, 0, 0, 0, 0],
            "flag_batt_sat": [0, 0, 0, 0, 0],
        }
    )


def test_build_snapshots_has_expected_labels() -> None:
    snaps = _build_snapshots(_sample_timeseries())
    labels = {s["label"] for s in snaps}
    assert "Start" in labels
    assert "First engine start" in labels
    assert "First regen" in labels
    assert "Max speed" in labels
    assert "End" in labels


def test_payload_contains_frames_and_snapshots() -> None:
    ts = _sample_timeseries()
    payload = build_viewer_payload(run_dir=Path("."), ts=ts)

    assert payload["summary"]["num_frames"] == 5
    assert payload["frames"][2]["engine_on"] is True
    assert payload["frames"][3]["regen_active"] is True
    assert len(payload["snapshots"]) >= 4
