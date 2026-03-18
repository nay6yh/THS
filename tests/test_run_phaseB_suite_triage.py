import json
import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import unittest

import scripts.run_phaseB_suite as suite


class TestPhaseBSuiteTriage(unittest.TestCase):
    def test_classify_traction_shortfall(self):
        gate_detail = {
            "A": {
                "A_shortfall_step_ratio": {"ok": False, "value": 0.2, "thr": 0.001},
                "A_E_short_over_E_trac": {"ok": False, "value": 0.02, "thr": 0.001},
            },
            "B": {"PASS": True},
            "PASS": False,
        }
        reason, note = suite._classify_gate_failure(gate_detail)
        self.assertEqual(reason, "traction_shortfall")
        self.assertIn("A_shortfall_step_ratio", note)

    def test_expected_fail_detected_interpretation(self):
        gate_detail = {
            "A": {
                "A_shortfall_step_ratio": {"ok": True},
                "A_E_short_over_E_trac": {"ok": False},
            }
        }
        ok, trigger = suite._expected_fail_detected(gate_detail)
        self.assertTrue(ok)
        self.assertEqual(trigger, "A_E_short_over_E_trac")


    def test_classify_failure_detail_includes_primary_and_secondary(self):
        gate_detail = {
            "A": {
                "A_psd_speed_resid_max_rpm": {"ok": False, "value": 301.0, "thr": 300.0},
            },
            "B": {
                "B_soc_recon_resid_max_abs_pct": {"ok": False, "value": 30.7, "thr": 0.01},
            },
            "PASS": False,
        }
        detail = suite._classify_gate_failure_detail(gate_detail)
        self.assertEqual(detail["reason"], "residual_limit_exceeded")
        self.assertEqual(detail["primary_failed_gate"], "A_psd_speed_resid_max_rpm")
        self.assertEqual(detail["secondary_failed_gates"], "B_soc_recon_resid_max_abs_pct")
        self.assertIn("value=301.0", detail["note"])

    def test_summary_row_exposes_triage_columns(self):
        row = suite._new_summary_row(
            variant="B00_baseline",
            passed=False,
            gating_mode="blocking",
            primary_failure_reason="residual_limit_exceeded",
            note="B_soc_recon_resid_max_abs_pct failed (value=30.7, thr=0.01)",
            blocking=True,
            primary_failed_gate="B_soc_recon_resid_max_abs_pct",
            primary_failed_value=30.7,
            primary_failed_thr=0.01,
            secondary_failed_gates="",
        )
        self.assertEqual(row["primary_failed_gate"], "B_soc_recon_resid_max_abs_pct")
        self.assertEqual(row["primary_failed_value"], 30.7)
        self.assertEqual(row["primary_failed_thr"], 0.01)
        self.assertEqual(row["secondary_failed_gates"], "")


    def test_summary_json_uses_null_for_missing_triage_values(self):
        row = suite._new_summary_row(
            variant="B00b_compare_vs_baseline",
            passed=True,
            gating_mode="informational",
            primary_failure_reason="pass",
            note="all gating checks passed",
            blocking=False,
        )
        payload = json.dumps([row], allow_nan=False)
        self.assertIn('"primary_failed_value": null', payload)
        self.assertIn('"primary_failed_thr": null', payload)
        self.assertNotIn('NaN', payload)

    def test_informational_row_is_non_blocking(self):
        row = suite._new_summary_row(
            variant="B00b_compare_vs_baseline",
            passed=False,
            gating_mode="informational",
            primary_failure_reason="informational_compare_regression",
            note="non-gating baseline comparison row",
            blocking=False,
        )
        self.assertFalse(row["PASS"])
        self.assertEqual(row["gating_mode"], "informational")
        self.assertFalse(row["blocking"])

    def test_determinism_compare_uses_kpis_even_when_runs_fail(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            r1 = out_dir / "B09_determinism_run1"
            r2 = out_dir / "B09_determinism_run2"
            r1.mkdir(parents=True)
            r2.mkdir(parents=True)
            (r1 / "kpis_phaseB.json").write_text('{"fuel_g_per_km": 1.0}', encoding="utf-8")
            (r2 / "kpis_phaseB.json").write_text('{"fuel_g_per_km": 1.0}', encoding="utf-8")

            result = suite._evaluate_determinism_compare(str(out_dir))
            self.assertTrue(result["PASS"])
            self.assertEqual(result["primary_failure_reason"], "pass")

    def test_determinism_compare_reports_missing_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            r1 = out_dir / "B09_determinism_run1"
            r1.mkdir(parents=True)
            (r1 / "kpis_phaseB.json").write_text('{"fuel_g_per_km": 1.0}', encoding="utf-8")

            result = suite._evaluate_determinism_compare(str(out_dir))
            self.assertFalse(result["PASS"])
            self.assertEqual(result["primary_failure_reason"], "missing_artifact")
            self.assertIn("B09_determinism_run2/kpis_phaseB.json", result["note"])


if __name__ == "__main__":
    unittest.main()
