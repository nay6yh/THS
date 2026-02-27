import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import unittest

from ths_zero.fuel_B import calc_fuel_account


class TestFuelB(unittest.TestCase):
    def test_zero_rpm_zero_fuel(self):
        bsfc_map = lambda rpm, tq: 250.0
        res = calc_fuel_account(eng_rpm=0.0, eng_tq_Nm=100.0, bsfc_map=bsfc_map, lhv_J_per_g=43_000)
        self.assertEqual(res.mdot_fuel_gps, 0.0)
        self.assertEqual(res.P_fuel_W, 0.0)
        self.assertEqual(res.loss_engine_W, 0.0)

    def test_zero_torque_zero_fuel(self):
        bsfc_map = lambda rpm, tq: 250.0
        res = calc_fuel_account(eng_rpm=2000.0, eng_tq_Nm=0.0, bsfc_map=bsfc_map)
        self.assertEqual(res.mdot_fuel_gps, 0.0)
        self.assertEqual(res.P_fuel_W, 0.0)
        self.assertEqual(res.loss_engine_W, 0.0)

    def test_positive_power_produces_positive_fuel_and_loss(self):
        bsfc_map = lambda rpm, tq: 250.0  # g/kWh
        eng_rpm = 3000.0
        eng_tq = 100.0
        res = calc_fuel_account(eng_rpm=eng_rpm, eng_tq_Nm=eng_tq, bsfc_map=bsfc_map, lhv_J_per_g=43_000)
        self.assertGreater(res.mdot_fuel_gps, 0.0)
        self.assertGreater(res.P_fuel_W, 0.0)
        self.assertGreater(res.loss_engine_W, 0.0)

        P_mech = (eng_rpm * 2.0 * math.pi / 60.0) * eng_tq
        # energy identity
        self.assertAlmostEqual(res.P_fuel_W, P_mech + res.loss_engine_W, delta=1e-6 * max(1.0, res.P_fuel_W))

    def test_negative_rpm_raises(self):
        bsfc_map = lambda rpm, tq: 250.0
        with self.assertRaises(ValueError):
            calc_fuel_account(eng_rpm=-10.0, eng_tq_Nm=50.0, bsfc_map=bsfc_map)

    def test_bsfc_map_failure_raises(self):
        def bad_bsfc(_rpm: float, _tq: float) -> float:
            raise RuntimeError("lookup failed")
        with self.assertRaises(ValueError):
            calc_fuel_account(eng_rpm=2000.0, eng_tq_Nm=50.0, bsfc_map=bad_bsfc)


if __name__ == "__main__":
    unittest.main()
