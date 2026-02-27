import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest
from types import SimpleNamespace

import pandas as pd

from ths_zero.sim_grid_B import default_output_filenames, simulate_ths_grid_B
from ths_zero.battery_B import BatteryConfigB
from ths_zero.losses_B import SimConfigB
from ths_zero.schema_B import PHASE_B_OUTPUT_COLUMN_ORDER, validate_column_order


class TestSimGridB(unittest.TestCase):
    def test_default_output_filenames(self):
        ts, cs = default_output_filenames("20250101_000000")
        self.assertTrue(ts.startswith("timeseries_phaseB_"))
        self.assertTrue(ts.endswith(".csv"))
        self.assertTrue(cs.startswith("constraints_phaseB_"))
        self.assertTrue(cs.endswith(".csv"))

    def test_simulate_requires_solver_injection(self):
        wltc = pd.DataFrame([
            {"t_s": 0.0, "dt_s": 1.0, "veh_spd_mps": 0.0, "veh_acc_mps2": 0.0, "phase": "Low"}
        ])
        with self.assertRaises(NotImplementedError):
            simulate_ths_grid_B(
                wltc,
                batt_cfg=BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.90),
                sim_cfg=SimConfigB(inv_loss_mode="embedded_in_mg_eta"),
                bsfc_map=lambda rpm, tq: 250.0,
                solve_step_base_fn=None,
            )

    def test_simulate_with_mock_solver(self):
        # minimal WLTC (3 steps)
        wltc = pd.DataFrame([
            {"t_s": 0.0, "dt_s": 1.0, "veh_spd_mps": 10.0, "veh_acc_mps2": 0.0, "phase": "Low"},
            {"t_s": 1.0, "dt_s": 1.0, "veh_spd_mps": 10.0, "veh_acc_mps2": 0.0, "phase": "Low"},
            {"t_s": 2.0, "dt_s": 1.0, "veh_spd_mps": 10.0, "veh_acc_mps2": 0.0, "phase": "Low"},
        ])

        # minimal config stubs for derived computations
        common = SimpleNamespace(g=9.80665)
        veh = SimpleNamespace(
            mass_kg=1250.0,
            Crr=0.010,
            CdA=0.62,
            tire_radius_m=0.31,
            final_drive=3.7,
            driveline_eff=0.97,
            Zs=30,
            Zr=78,
            eng_tq_max_Nm=120.0,
            mg1_tq_max_Nm=80.0,
            mg2_tq_max_Nm=220.0,
            mg1_rpm_max=10000.0,
            mg2_rpm_max=12000.0,
        )
        batt = SimpleNamespace(
            E_usable_Wh=800.0,
            soc_min=0.30,
            soc_max=0.80,
            P_aux_base_W=300.0,
            P_discharge_nom_W=30000.0,
            P_charge_nom_W=25000.0,
        )
        env = SimpleNamespace(Tamb_C=20.0, p_amb_Pa=101325.0, cabin_setpoint_C=23.0)
        init = SimpleNamespace(soc0=0.55)

        def mock_solver(x, **_kwargs):
            # Provide minimal StepResult-like attributes used by enrich + default row builder
            # Choose charging case: P_batt_act_W < 0
            return SimpleNamespace(
                mode="HybridDrive",
                fuel_cut=0,
                eng_rpm=3000.0,
                eng_tq_Nm=100.0,
                mg1_rpm=800.0,
                mg1_tq_Nm=-2.0,
                mg2_rpm=1200.0,
                mg2_tq_Nm=-5.0,

                P_eng_mech_W=(3000.0 * 2 * 3.141592653589793 / 60.0) * 100.0,
                P_mg1_mech_W=-25.0,
                P_mg2_mech_W=-35.0,
                P_mg1_elec_W=-200.0,  # engine generation source
                P_mg2_elec_W=-300.0,  # regen source
                P_aux_W=400.0,
                P_batt_req_W=-500.0,
                P_batt_act_W=-500.0,

                P_brake_fric_W=0.0,
                shortfall_tq_Nm=0.0,
                shortfall_power_W=0.0,

                excess_tq_Nm=0.0,
                J_over=0.0,

                E_batt_next_Wh=0.55 * 800.0,
                soc_next=0.55,

                J_total=1.0,
                J_fuel=1.0,
                J_soc=0.0,
                J_fric=0.0,
                J_short=0.0,
                J_spin=0.0,
                J_smooth=0.0,
                J_charge=0.0,

                stats=SimpleNamespace(n_total=10, n_kept=2),
            )

        ts, cons = simulate_ths_grid_B(
            wltc,
            common=common,
            veh=veh,
            batt=batt,
            init=init,
            env=env,
            batt_cfg=BatteryConfigB(model="B0", eta_charge=0.95, eta_discharge=0.90),
            sim_cfg=SimConfigB(inv_loss_mode="embedded_in_mg_eta"),
            bsfc_map=lambda rpm, tq: 250.0,
            solve_step_base_fn=mock_solver,
        )

        self.assertEqual(list(ts.columns), PHASE_B_OUTPUT_COLUMN_ORDER)
        validate_column_order(ts, strict=True)
        self.assertTrue((ts["sim_phase"] == "B").all())

        # Split sanity (v10): non-braking mode -> regen attribution must be 0 to avoid regen_utilization > 1
        self.assertAlmostEqual(float(ts.loc[0, "P_batt_chg_from_regen_W"]), 0.0, places=6)
        self.assertAlmostEqual(float(ts.loc[0, "P_batt_chg_from_engine_W"]), 500.0, places=6)

        # Extension columns exist
        for c in (
            "P_batt_chem_W", "loss_batt_W", "I_batt_A",
            "mdot_fuel_gps", "P_fuel_W", "loss_engine_W",
            "loss_inv1_W", "loss_inv2_W", "loss_inv_W",
        ):
            self.assertIn(c, ts.columns)

        self.assertTrue(cons.empty)


if __name__ == "__main__":
    unittest.main()
