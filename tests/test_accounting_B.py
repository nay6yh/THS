import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.accounting_B import split_charging_origin, bus_residual_W, integrate_trapz_J


class TestAccountingB(unittest.TestCase):
    def test_split_charging_origin_sign_logic(self):
        # charging: P_batt_term_W < 0
        # mg2 generates (regen), mg1 generates (engine charge)
        P_batt_term_W = -300.0
        P_mg1_bus_W = -100.0
        P_mg2_bus_W = -250.0
        P_regen, P_eng = split_charging_origin(
            P_batt_term_W=P_batt_term_W,
            P_mg1_bus_W=P_mg1_bus_W,
            P_mg2_bus_W=P_mg2_bus_W,
        )
        self.assertGreaterEqual(P_regen, 0.0)
        self.assertGreaterEqual(P_eng, 0.0)
        self.assertAlmostEqual(P_regen + P_eng, 300.0, places=6)
        # regen prioritized, limited by mg2 generation magnitude (250)
        self.assertAlmostEqual(P_regen, 250.0, places=6)
        self.assertAlmostEqual(P_eng, 50.0, places=6)

    def test_split_no_charge(self):
        P_regen, P_eng = split_charging_origin(P_batt_term_W=10.0, P_mg1_bus_W=-10.0, P_mg2_bus_W=-10.0)
        self.assertEqual(P_regen, 0.0)
        self.assertEqual(P_eng, 0.0)

    def test_integrate_trapz_uniform_dt(self):
        P_W = [100.0, 200.0, 150.0]
        dt_s = [1.0, 1.0]
        E_J = integrate_trapz_J(P_W=P_W, dt_s=dt_s)
        self.assertAlmostEqual(E_J, 325.0, places=9)

    def test_integrate_trapz_variable_dt(self):
        P_W = [0.0, 100.0, 50.0]
        dt_s = [2.0, 3.0]
        E_J = integrate_trapz_J(P_W=P_W, dt_s=dt_s)
        self.assertAlmostEqual(E_J, 325.0, places=9)

    def test_integrate_trapz_empty(self):
        self.assertEqual(integrate_trapz_J(P_W=[1.0], dt_s=[]), 0.0)

    def test_integrate_trapz_negative_dt_raises(self):
        with self.assertRaises(ValueError):
            integrate_trapz_J(P_W=[0.0, 1.0], dt_s=[-1.0])

    def test_bus_residual(self):
        resid = bus_residual_W(P_batt_term_W=100.0, P_mg1_bus_W=40.0, P_mg2_bus_W=50.0, P_aux_W=10.0)
        self.assertAlmostEqual(resid, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
