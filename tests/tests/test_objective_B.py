import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.objective_B import score_candidate_B, ObjectiveWeightsB


class DummyBase:
    shortfall_tq_Nm = 0.0
    J_over = 0.0
    soc_next = 0.55
    eng_rpm = 2000.0
    mg1_rpm = 0.0


class TestObjectiveB(unittest.TestCase):
    def test_objective_is_stub(self):
        with self.assertRaises(NotImplementedError):
            score_candidate_B(
                base=DummyBase(),
                P_fuel_W=1000.0,
                loss_total_W=100.0,
                soc_target=0.55,
                soc_band=0.05,
                weights=ObjectiveWeightsB(),
            )


if __name__ == "__main__":
    unittest.main()
