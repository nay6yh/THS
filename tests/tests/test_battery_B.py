import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.battery_B import battery_B0, battery_B1, BatteryConfigB


class TestBatteryB(unittest.TestCase):
    def test_battery_B0_discharge(self):
        cfg = BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.90)
        res = battery_B0(P_term_W=1000.0, cfg=cfg)
        self.assertEqual(res.P_term_W, 1000.0)
        self.assertGreater(res.P_chem_W, 1000.0)
        self.assertGreater(res.loss_batt_W, 0.0)
        self.assertEqual(res.I_batt_A, 0.0)
        self.assertAlmostEqual(res.P_chem_W, res.P_term_W + res.loss_batt_W, places=9)

    def test_battery_B0_charge(self):
        cfg = BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.90)
        res = battery_B0(P_term_W=-1000.0, cfg=cfg)
        self.assertEqual(res.P_term_W, -1000.0)
        # chemical magnitude smaller -> less negative than P_term
        self.assertGreater(res.P_chem_W, -1000.0)
        self.assertLess(res.P_chem_W, 0.0)
        self.assertGreater(res.loss_batt_W, 0.0)
        self.assertAlmostEqual(abs(res.P_chem_W), abs(res.P_term_W) - res.loss_batt_W, places=9)

    def test_battery_B0_idle(self):
        cfg = BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.90)
        res = battery_B0(P_term_W=0.0, cfg=cfg)
        self.assertEqual(res.P_term_W, 0.0)
        self.assertEqual(res.P_chem_W, 0.0)
        self.assertEqual(res.loss_batt_W, 0.0)

    def test_battery_config_invalid_efficiency(self):
        with self.assertRaises(ValueError):
            BatteryConfigB(model="B0", eta_charge=1.5, eta_discharge=0.9)
        with self.assertRaises(ValueError):
            BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.0)

    def test_battery_B1_is_stub(self):
        with self.assertRaises(NotImplementedError):
            battery_B1(P_term_W=1000.0, soc=0.5, cfg=BatteryConfigB(model="B1"))


if __name__ == "__main__":
    unittest.main()
