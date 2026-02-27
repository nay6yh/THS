import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.schema_B import (
    PHASE_A_TIMESERIES_COLUMN_ORDER,
    PHASE_A_CANONICAL_COLUMN_ORDER,
    PHASE_B_EXT_COLUMNS,
    PHASE_B_OUTPUT_COLUMN_ORDER,
    PHASE_B_CANONICAL_OUTPUT_ORDER,
    extract_canonical_view,
    SIM_PHASE_VALUE,
)


class TestSchemaB(unittest.TestCase):
    def test_phase_a_timeseries_column_order_frozen(self):
        expected = [
            'phase', 't_s', 'dt_s', 'veh_spd_mps', 'veh_acc_mps2', 'Tamb_C', 'rho_air_kgpm3', 'Tbatt_C',
            'F_roll_N', 'F_aero_N', 'F_iner_N', 'F_total_N', 'wheel_omega_radps', 'ring_omega_radps', 'ring_rpm',
            'T_wheel_req_Nm', 'P_wheel_req_W', 'T_ring_req_Nm', 'mode', 'fuel_cut', 'eng_rpm', 'eng_tq_Nm',
            'mg1_rpm', 'mg1_tq_Nm', 'mg2_rpm', 'mg2_tq_Nm', 'P_eng_mech_W', 'P_mg1_mech_W', 'P_mg2_mech_W',
            'P_mg1_elec_W', 'P_mg2_elec_W', 'P_hvac_W', 'P_aux_W', 'P_batt_req_W', 'P_batt_act_W',
            'P_brake_fric_W', 'shortfall_tq_Nm', 'shortfall_power_W', 'soc_pct', 'E_batt_Wh',
            'lim_batt_discharge_W', 'lim_batt_charge_W', 'J_total', 'J_fuel', 'J_soc', 'J_fric', 'J_short',
            'J_spin', 'J_smooth', 'J_charge', 'excess_tq_Nm', 'J_over', 'n_grid_total', 'n_grid_kept',
            'batt_E_usable_Wh', 'batt_Emin_Wh', 'batt_Emax_Wh', 'flag_eng_sat', 'flag_mg1_sat', 'flag_mg2_sat',
            'flag_batt_sat', 'flag_mg1_overspeed', 'flag_mg2_overspeed', 'T_ring_deliv_Nm',
            'resid_ring_torque_Nm_dbg', 'P_wheel_deliv_W_dbg', 'resid_wheel_power_W_dbg',
            'wheel_omega_radps_dbg', 'ring_omega_radps_dbg', 'T_wheel_req_Nm_dbg', 'T_ring_req_Nm_dbg'
        ]
        self.assertEqual(PHASE_A_TIMESERIES_COLUMN_ORDER, expected)
        self.assertEqual(len(PHASE_A_TIMESERIES_COLUMN_ORDER), 71)

    def test_phase_b_ext_columns_frozen(self):
        expected_ext = [
            'P_batt_chem_W', 'loss_batt_W', 'I_batt_A',
            'mdot_fuel_gps', 'P_fuel_W', 'loss_engine_W',
            'loss_mg1_W', 'loss_mg2_W',
            'loss_inv1_W', 'loss_inv2_W', 'loss_inv_W',
            'P_batt_chg_from_regen_W', 'P_batt_chg_from_engine_W',
            'sim_phase'
        ]
        self.assertEqual(PHASE_B_EXT_COLUMNS, expected_ext)
        self.assertEqual(len(PHASE_B_EXT_COLUMNS), 14)

    def test_phase_b_output_order(self):
        self.assertEqual(SIM_PHASE_VALUE, 'B')
        self.assertEqual(PHASE_B_OUTPUT_COLUMN_ORDER, PHASE_A_TIMESERIES_COLUMN_ORDER + PHASE_B_EXT_COLUMNS)
        self.assertEqual(len(PHASE_B_OUTPUT_COLUMN_ORDER), 85)
        self.assertEqual(PHASE_B_OUTPUT_COLUMN_ORDER[-1], 'sim_phase')

    def test_phase_a_canonical_subset_is_subset(self):
        for c in PHASE_A_CANONICAL_COLUMN_ORDER:
            self.assertIn(c, PHASE_A_TIMESERIES_COLUMN_ORDER)




    def test_phase_b_canonical_output_order(self):
        self.assertEqual(PHASE_B_CANONICAL_OUTPUT_ORDER, PHASE_A_CANONICAL_COLUMN_ORDER + PHASE_B_EXT_COLUMNS)
        self.assertEqual(len(PHASE_B_CANONICAL_OUTPUT_ORDER), len(PHASE_A_CANONICAL_COLUMN_ORDER) + len(PHASE_B_EXT_COLUMNS))

    def test_extract_canonical_view(self):
        import pandas as pd
        # build minimal df with all required columns
        cols = PHASE_B_OUTPUT_COLUMN_ORDER
        df = pd.DataFrame({c: [0.0] for c in cols})
        view = extract_canonical_view(df, strict=True)
        self.assertEqual(list(view.columns), PHASE_B_CANONICAL_OUTPUT_ORDER)


if __name__ == '__main__':
    unittest.main()
