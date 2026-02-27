import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

import numpy as np
import pandas as pd

from ths_zero.audit_B import compute_soc_reconstruction_residual_B, compute_audit_outputs_B, validate_audit_outputs_B


class TestAuditB(unittest.TestCase):
    def _make_ts(self):
        # 3 samples, 2 effective integration intervals for reconstruction update
        dt_s = [1.0, 1.0, 1.0]
        E_usable = [1000.0, 1000.0, 1000.0]

        # Use P_batt_chem = 3600 W for 1s -> 1 Wh discharge on first interval
        P_batt_chem = [3600.0, 0.0, 0.0]
        E_batt = [500.0, 499.0, 499.0]
        soc_pct = [50.0, 49.9, 49.9]

        # Perfect fuel balance: P_fuel = P_eng + loss
        P_eng = [1000.0, 1000.0, 0.0]
        loss_engine = [2000.0, 2000.0, 0.0]
        P_fuel = [3000.0, 3000.0, 0.0]

        # Perfect bus balance
        P_batt_act = [100.0, 100.0, -50.0]
        P_mg1 = [30.0, 30.0, -10.0]
        P_mg2 = [50.0, 50.0, -20.0]
        P_aux = [20.0, 20.0, -20.0]

        # Wheel delivered: include a braking sample to exercise regen metrics
        P_wheel_deliv = [500.0, -400.0, 0.0]

        # Charging split powers (>=0). Charging occurs on sample 1 (wheel negative).
        P_regen_to_batt = [0.0, 200.0, 0.0]
        P_engine_to_batt = [0.0, 0.0, 0.0]

        ts = pd.DataFrame({
            "phase": ["X", "X", "X"],
            "t_s": [0.0, 1.0, 2.0],
            "dt_s": dt_s,
            "veh_spd_mps": [0.0, 0.0, 0.0],
            "veh_acc_mps2": [0.0, 0.0, 0.0],
            "Tamb_C": [25.0, 25.0, 25.0],
            "rho_air_kgpm3": [1.2, 1.2, 1.2],
            "Tbatt_C": [25.0, 25.0, 25.0],
            "F_roll_N": [0.0, 0.0, 0.0],
            "F_aero_N": [0.0, 0.0, 0.0],
            "F_iner_N": [0.0, 0.0, 0.0],
            "F_total_N": [0.0, 0.0, 0.0],
            "wheel_omega_radps": [0.0, 0.0, 0.0],
            "ring_omega_radps": [0.0, 0.0, 0.0],
            "ring_rpm": [0.0, 0.0, 0.0],
            "T_wheel_req_Nm": [0.0, 0.0, 0.0],
            "P_wheel_req_W": [0.0, 0.0, 0.0],
            "T_ring_req_Nm": [0.0, 0.0, 0.0],
            "mode": ["EV", "EV", "EV"],
            "fuel_cut": [1, 1, 1],
            "eng_rpm": [0.0, 0.0, 0.0],
            "eng_tq_Nm": [0.0, 0.0, 0.0],
            "mg1_rpm": [0.0, 0.0, 0.0],
            "mg1_tq_Nm": [0.0, 0.0, 0.0],
            "mg2_rpm": [0.0, 0.0, 0.0],
            "mg2_tq_Nm": [0.0, 0.0, 0.0],
            "P_eng_mech_W": P_eng,
            "P_mg1_mech_W": [0.0, 0.0, 0.0],
            "P_mg2_mech_W": [0.0, 0.0, 0.0],
            "P_mg1_elec_W": P_mg1,
            "P_mg2_elec_W": P_mg2,
            "P_hvac_W": [0.0, 0.0, 0.0],
            "P_aux_W": P_aux,
            "P_batt_req_W": P_batt_act,
            "P_batt_act_W": P_batt_act,
            "P_brake_fric_W": [0.0, 0.0, 0.0],
            "shortfall_tq_Nm": [0.0, 0.0, 0.0],
            "shortfall_power_W": [0.0, 0.0, 0.0],
            "soc_pct": soc_pct,
            "E_batt_Wh": E_batt,
            "lim_batt_discharge_W": [0.0, 0.0, 0.0],
            "lim_batt_charge_W": [0.0, 0.0, 0.0],
            "J_total": [0.0, 0.0, 0.0],
            "J_fuel": [0.0, 0.0, 0.0],
            "J_soc": [0.0, 0.0, 0.0],
            "J_fric": [0.0, 0.0, 0.0],
            "J_short": [0.0, 0.0, 0.0],
            "J_spin": [0.0, 0.0, 0.0],
            "J_smooth": [0.0, 0.0, 0.0],
            "J_charge": [0.0, 0.0, 0.0],
            "excess_tq_Nm": [0.0, 0.0, 0.0],
            "J_over": [0.0, 0.0, 0.0],
            "n_grid_total": [0, 0, 0],
            "n_grid_kept": [0, 0, 0],
            "batt_E_usable_Wh": E_usable,
            "batt_Emin_Wh": [0.0, 0.0, 0.0],
            "batt_Emax_Wh": [1000.0, 1000.0, 1000.0],
            "flag_eng_sat": [0, 0, 0],
            "flag_mg1_sat": [0, 0, 0],
            "flag_mg2_sat": [0, 0, 0],
            "flag_batt_sat": [0, 0, 0],
            "flag_mg1_overspeed": [0, 0, 0],
            "flag_mg2_overspeed": [0, 0, 0],
            "T_ring_deliv_Nm": [0.0, 0.0, 0.0],
            "resid_ring_torque_Nm_dbg": [0.0, 0.0, 0.0],
            "P_wheel_deliv_W_dbg": P_wheel_deliv,
            "resid_wheel_power_W_dbg": [0.0, 0.0, 0.0],
            "wheel_omega_radps_dbg": [0.0, 0.0, 0.0],
            "ring_omega_radps_dbg": [0.0, 0.0, 0.0],
            "T_wheel_req_Nm_dbg": [0.0, 0.0, 0.0],
            "T_ring_req_Nm_dbg": [0.0, 0.0, 0.0],

            # Phase B ext columns needed by audit
            "P_batt_chem_W": P_batt_chem,
            "loss_batt_W": [0.0, 0.0, 0.0],
            "I_batt_A": [0.0, 0.0, 0.0],
            "mdot_fuel_gps": [0.0, 0.0, 0.0],
            "P_fuel_W": P_fuel,
            "loss_engine_W": loss_engine,
            "loss_mg1_W": [0.0, 0.0, 0.0],
            "loss_mg2_W": [0.0, 0.0, 0.0],
            "loss_inv1_W": [0.0, 0.0, 0.0],
            "loss_inv2_W": [0.0, 0.0, 0.0],
            "loss_inv_W": [0.0, 0.0, 0.0],
            "P_batt_chg_from_regen_W": P_regen_to_batt,
            "P_batt_chg_from_engine_W": P_engine_to_batt,
            "sim_phase": ["B", "B", "B"],
        })
        return ts

    def test_soc_recon_residual_near_zero(self):
        ts = self._make_ts()
        resid = compute_soc_reconstruction_residual_B(ts)
        self.assertEqual(len(resid), len(ts))
        self.assertTrue(np.all(np.abs(resid.to_numpy()) < 1e-9))

    def test_compute_audit_outputs(self):
        ts = self._make_ts()
        cons = pd.DataFrame({"t_s": ts["t_s"]})
        cons2, kpis, budgets = compute_audit_outputs_B(ts, cons)

        # residual columns added
        self.assertIn("fuel_balance_resid_W", cons2.columns)
        self.assertIn("bus_balance_resid_W", cons2.columns)
        self.assertIn("soc_recon_resid_pct", cons2.columns)

        # KPI sanity
        self.assertIn("TTW_eff", kpis)
        self.assertIn("regen_utilization", kpis)
        self.assertGreaterEqual(kpis["regen_utilization"], 0.0)
        self.assertLessEqual(kpis["regen_utilization"], 1.0 + 1e-12)

        # budgets sanity
        self.assertIn("E_fuel_MJ", budgets)
        self.assertGreaterEqual(budgets["E_fuel_MJ"], 0.0)


    def test_validate_audit_outputs_passes(self):
        ts = self._make_ts()
        cons = pd.DataFrame({"t_s": ts["t_s"]})
        cons2, kpis, budgets = compute_audit_outputs_B(ts, cons)

        violations = validate_audit_outputs_B(kpis, budgets, cons2, strict=False)
        # All checks should pass for this synthetic perfect case
        self.assertTrue(all(v["passed"] for v in violations.values()))

        # strict=True should not raise
        validate_audit_outputs_B(kpis, budgets, cons2, strict=True)

    def test_validate_audit_outputs_raises_on_violation(self):
        ts = self._make_ts()
        cons = pd.DataFrame({"t_s": ts["t_s"]})
        cons2, kpis, budgets = compute_audit_outputs_B(ts, cons)
        kpis_bad = dict(kpis)
        kpis_bad["regen_utilization"] = 1.2  # out of [0,1]
        with self.assertRaises(ValueError):
            validate_audit_outputs_B(kpis_bad, budgets, cons2, strict=True)


if __name__ == "__main__":
    unittest.main()
