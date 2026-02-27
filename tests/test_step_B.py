import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest
from types import SimpleNamespace

from ths_zero.step_B import solve_step_B, StepInputsB, enrich_with_phase_b_extensions
from ths_zero.battery_B import BatteryConfigB
from ths_zero.losses_B import SimConfigB


class TestStepB(unittest.TestCase):
    def test_solve_step_B_is_stub(self):
        def dummy_bsfc(_rpm: float, _tq: float) -> float:
            return 240.0
        def dummy_eta(_rpm: float, _tq: float) -> float:
            return 0.92
        def dummy_tqmax(_rpm: float) -> float:
            return 140.0

        x = StepInputsB(
            t_s=0.0, dt_s=1.0, phase="Low",
            ring_rpm=1000.0, ring_omega_radps=100.0,
            T_ring_req_Nm=10.0, P_wheel_req_W=1000.0,
            soc=0.55, E_batt_Wh=400.0,
            E_usable_Wh=800.0, Emin_Wh=240.0, Emax_Wh=640.0,
            P_aux_W=300.0, Tamb_C=20.0, Tbatt_C=20.0,
            P_dis_max_W=30000.0, P_chg_max_W=25000.0,
            mg1_rpm_max=10000.0, mg2_rpm_max=12000.0,
            mg1_tq_max_Nm=80.0, mg2_tq_max_Nm=220.0,
            eng_rpm_min=1200.0, eng_rpm_max=5200.0,
            alpha=0.277, beta=0.723,
        )

        with self.assertRaises(NotImplementedError):
            solve_step_B(
                x,
                bsfc_map=dummy_bsfc,
                eta_mg1_map=dummy_eta,
                eta_mg2_map=dummy_eta,
                eng_tq_max_map=dummy_tqmax,
                eng_drag_min_map=None,
                batt_cfg=BatteryConfigB(),
                sim_cfg=SimConfigB(),
            )

    def test_enrich_with_phase_b_extensions_computes_accounts(self):
        # minimal StepResult-like object (do NOT import step_A)
        base = SimpleNamespace(
            eng_rpm=3000.0,
            eng_tq_Nm=100.0,
            P_batt_act_W=-1000.0,  # charging
            P_mg1_mech_W=-600.0,
            P_mg1_elec_W=-500.0,   # generating to bus
            P_mg2_mech_W=-900.0,
            P_mg2_elec_W=-700.0,   # regen to bus
            soc_next=0.55,
        )

        bsfc_map = lambda rpm, tq: 250.0

        sol_b = enrich_with_phase_b_extensions(
            base=base,
            bsfc_map=bsfc_map,
            batt_cfg=BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.90),
            sim_cfg=SimConfigB(inv_loss_mode="embedded_in_mg_eta"),
        )

        # fuel
        self.assertGreater(sol_b.fuel.mdot_fuel_gps, 0.0)
        self.assertGreater(sol_b.fuel.P_fuel_W, 0.0)
        self.assertGreater(sol_b.fuel.loss_engine_W, 0.0)

        # battery (B0)
        self.assertEqual(sol_b.batt.P_term_W, -1000.0)
        self.assertAlmostEqual(sol_b.batt.P_chem_W, -950.0, places=6)
        self.assertAlmostEqual(sol_b.batt.loss_batt_W, 50.0, places=6)

        # mg losses are non-negative
        self.assertGreaterEqual(sol_b.losses.loss_mg1_W, 0.0)
        self.assertGreaterEqual(sol_b.losses.loss_mg2_W, 0.0)

        # inverter losses embedded => 0
        self.assertEqual(sol_b.losses.loss_inv1_W, 0.0)
        self.assertEqual(sol_b.losses.loss_inv2_W, 0.0)
        self.assertEqual(sol_b.losses.loss_inv_W, 0.0)

        # split: charge magnitude 1000, regen prioritized up to 700
        self.assertAlmostEqual(sol_b.split.P_batt_chg_from_regen_W, 700.0, places=6)
        self.assertAlmostEqual(sol_b.split.P_batt_chg_from_engine_W, 300.0, places=6)


if __name__ == "__main__":
    unittest.main()
