import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.losses_B import mg_loss_from_mech_and_bus, inverter_losses, SimConfigB


class TestLossesB(unittest.TestCase):
    def test_mg_loss_motoring(self):
        loss = mg_loss_from_mech_and_bus(P_mech_W=100.0, P_bus_elec_W=120.0)
        self.assertAlmostEqual(loss, 20.0, places=9)

    def test_mg_loss_generating(self):
        loss = mg_loss_from_mech_and_bus(P_mech_W=-100.0, P_bus_elec_W=-80.0)
        self.assertAlmostEqual(loss, 20.0, places=9)

    def test_inverter_losses_embedded(self):
        cfg = SimConfigB(inv_loss_mode="embedded_in_mg_eta")
        l1, l2 = inverter_losses(cfg=cfg, P_mg1_bus_W=-100.0, P_mg2_bus_W=-200.0)
        self.assertEqual(l1, 0.0)
        self.assertEqual(l2, 0.0)

    def test_inverter_losses_separate_mode_placeholder(self):
        cfg = SimConfigB(inv_loss_mode="separate")
        l1, l2 = inverter_losses(cfg=cfg, P_mg1_bus_W=-1000.0, P_mg2_bus_W=2000.0)
        self.assertAlmostEqual(l1, 20.0, places=9)
        self.assertAlmostEqual(l2, 40.0, places=9)


if __name__ == "__main__":
    unittest.main()
