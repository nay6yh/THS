import sys
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


if __name__ == "__main__":
    unittest.main()
