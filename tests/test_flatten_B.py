import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.battery_B import BattChemResult
from ths_zero.fuel_B import FuelAccount
from ths_zero.losses_B import LossAccount
from ths_zero.step_B import StepResultB, ChargeSplit
from ths_zero.flatten_B import flatten_phase_b_extensions
from ths_zero.schema_B import PHASE_B_EXT_COLUMNS


class TestFlattenB(unittest.TestCase):
    def test_flatten_contains_all_ext_columns(self):
        sol = StepResultB(
            base=None,  # base not used by flatten
            batt=BattChemResult(P_chem_W=10.0, P_term_W=9.0, loss_batt_W=1.0, I_batt_A=0.0),
            fuel=FuelAccount(mdot_fuel_gps=0.1, P_fuel_W=4300.0, loss_engine_W=1000.0),
            losses=LossAccount(
                loss_mg1_W=1.0, loss_mg2_W=2.0,
                loss_inv1_W=0.0, loss_inv2_W=0.0,
                loss_engine_W=1000.0,
            ),
            split=ChargeSplit(P_batt_chg_from_regen_W=3.0, P_batt_chg_from_engine_W=4.0),
        )
        d = flatten_phase_b_extensions(sol)
        self.assertEqual(set(d.keys()), set(PHASE_B_EXT_COLUMNS))
        # sanity: fuel fields present
        self.assertIn("P_fuel_W", d)
        self.assertIn("mdot_fuel_gps", d)


if __name__ == "__main__":
    unittest.main()
