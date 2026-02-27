# ths_zero/sim_grid_A.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig
from .environment import air_density_kgpm3, hvac_power_W, battery_temp_C, batt_power_limits_W
from .step_A import StepInputs, StepWeights, solve_step_A

# solve_step_A 側（あなたの置き場に合わせて import）
from .step_A import (
    StepInputs, StepWeights, solve_step_A,
    EtaMap, BsfcMap, EngTqMaxMap, EngDragMinMap,
)

def constant_eta(_rpm: float, _tq: float) -> float:
    return 0.92

def constant_bsfc(_rpm: float, _tq: float) -> float:
    # 仮：実BSFCマップに置換する
    return 240.0

def eng_tq_max_flat(_rpm: float) -> float:
    return 140.0

def eng_drag_min_simple(rpm: float) -> float:
    # Fuel-cut drag（仮）：rpmに応じて負トルク下限
    # まずは小さめでOK（後で温度依存に拡張）
    return -10.0 - 0.002 * max(rpm, 0.0)

def simulate_ths_grid_A(
    wltc: pd.DataFrame,
    common: CommonConfig,
    veh: VehicleConfig,
    batt: BatteryConfig,
    init: InitialState,
    env: EnvironmentConfig,
    weights: StepWeights | None = None,
    bsfc_map: BsfcMap = constant_bsfc,
    eta_mg1: EtaMap = constant_eta,
    eta_mg2: EtaMap = constant_eta,
    eng_tq_max: EngTqMaxMap = eng_tq_max_flat,
    eng_drag_min: EngDragMinMap | None = eng_drag_min_simple,
    eng_rpm_step: float = 100.0,
    eng_tq_step: float = 5.0,
    soc_target: float = 0.55,
    soc_band: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      timeseries_df: main signals
      constraints_df: residuals/audit
    """
    if weights is None:
        weights = StepWeights()

    # PSD constants
    alpha = veh.Zs / (veh.Zs + veh.Zr)
    beta  = veh.Zr / (veh.Zs + veh.Zr)

    # state
    soc = float(init.soc0)
    E_batt_Wh = soc * batt.E_usable_Wh

    Emin_Wh = batt.soc_min * batt.E_usable_Wh
    Emax_Wh = batt.soc_max * batt.E_usable_Wh

    # unpack WLTC columns
    t = wltc["t_s"].to_numpy(dtype=float)
    dt = wltc["dt_s"].to_numpy(dtype=float)
    v = wltc["veh_spd_mps"].to_numpy(dtype=float)
    a = wltc["veh_acc_mps2"].to_numpy(dtype=float)
    phase = wltc["phase"].astype(str).to_numpy()

    rows = []
    cres = []

    prev_eng_rpm = None
    prev_eng_tq = None

    for k in range(len(t)):
        # --- environment
        Tamb_C = float(env.Tamb_C)  # Level1: constant
        rho = air_density_kgpm3(Tamb_C, env.p_amb_Pa, common)
        P_hvac = hvac_power_W(Tamb_C, env)
        P_aux = batt.P_aux_base_W + P_hvac

        Tbatt_C = battery_temp_C(Tamb_C, env)
        P_dis_max, P_chg_max = batt_power_limits_W(Tbatt_C, batt)

        # --- vehicle longitudinal demand
        F_roll = veh.mass_kg * common.g * veh.Crr
        F_aero = 0.5 * rho * veh.CdA * (v[k] ** 2)
        F_iner = veh.mass_kg * a[k]
        F_total = F_roll + F_aero + F_iner

        T_wheel_req = F_total * veh.tire_radius_m
        P_wheel_req = F_total * v[k]

        # --- driveline to ring
        wheel_omega = v[k] / max(veh.tire_radius_m, 1e-9)
        ring_omega = wheel_omega * veh.final_drive
        ring_rpm = ring_omega * 60.0 / (2.0 * np.pi)

        # “要求→リング換算”に driveline_eff を含める（監査しやすい）
        T_ring_req = T_wheel_req / max(veh.final_drive * veh.driveline_eff, 1e-9)

        # --- build step input
        xin = StepInputs(
            t_s=float(t[k]),
            dt_s=float(dt[k]),
            phase=str(phase[k]),
            ring_rpm=float(ring_rpm),
            ring_omega_radps=float(ring_omega),
            T_ring_req_Nm=float(T_ring_req),
            P_wheel_req_W=float(P_wheel_req),
            soc=float(soc),
            E_batt_Wh=float(E_batt_Wh),
            P_aux_W=float(P_aux),
            Tamb_C=float(Tamb_C),
            Tbatt_C=float(Tbatt_C),
            P_dis_max_W=float(P_dis_max),
            P_chg_max_W=float(P_chg_max),
            mg1_rpm_max=float(veh.mg1_rpm_max),
            mg2_rpm_max=float(veh.mg2_rpm_max),
            mg1_tq_max_Nm=float(veh.mg1_tq_max_Nm),
            mg2_tq_max_Nm=float(veh.mg2_tq_max_Nm),
            eng_rpm_min=float(veh.eng_rpm_min),
            eng_rpm_max=float(veh.eng_rpm_max),
            alpha=float(alpha),
            beta=float(beta),
            prev_eng_rpm=prev_eng_rpm,
            prev_eng_tq_Nm=prev_eng_tq,
            # battery constants (NEW)
            E_usable_Wh=float(batt.E_usable_Wh),
            Emin_Wh=float(Emin_Wh),
            Emax_Wh=float(Emax_Wh),
        )

        # --- solve 1 step
        sol = solve_step_A(
            xin,
            weights=weights,
            bsfc_map=bsfc_map,
            eng_tq_max_map=eng_tq_max,
            eta_mg1_map=eta_mg1,
            eta_mg2_map=eta_mg2,
            eng_drag_min_map=eng_drag_min,
            eng_rpm_step=eng_rpm_step,
            eng_tq_step=eng_tq_step,
            soc_target=soc_target,
            soc_band=soc_band,
        )

        # update state
        soc = sol.soc_next
        E_batt_Wh = sol.E_batt_next_Wh
        prev_eng_rpm = sol.eng_rpm
        prev_eng_tq = sol.eng_tq_Nm
        # wheel delivered power from ring torque (driveline efficiency)
        T_ring_deliv = xin.beta * sol.eng_tq_Nm + sol.mg2_tq_Nm
        P_wheel_deliv_W = T_ring_deliv * xin.ring_omega_radps * veh.driveline_eff

        # save timeseries
        rows.append({
            "phase": xin.phase,
            "t_s": xin.t_s,
            "dt_s": xin.dt_s,
            "veh_spd_mps": float(v[k]),
            "veh_acc_mps2": float(a[k]),
            "Tamb_C": xin.Tamb_C,
            "rho_air_kgpm3": float(rho),
            "Tbatt_C": xin.Tbatt_C,

            "F_roll_N": float(F_roll),
            "F_aero_N": float(F_aero),
            "F_iner_N": float(F_iner),
            "F_total_N": float(F_total),

            "wheel_omega_radps": float(wheel_omega),
            "ring_omega_radps": xin.ring_omega_radps,
            "ring_rpm": xin.ring_rpm,

            "T_wheel_req_Nm": float(T_wheel_req),
            "P_wheel_req_W": xin.P_wheel_req_W,
            "T_ring_req_Nm": xin.T_ring_req_Nm,

            "mode": sol.mode,
            "fuel_cut": int(sol.fuel_cut),

            "eng_rpm": sol.eng_rpm,
            "eng_tq_Nm": sol.eng_tq_Nm,
            "mg1_rpm": sol.mg1_rpm,
            "mg1_tq_Nm": sol.mg1_tq_Nm,
            "mg2_rpm": sol.mg2_rpm,
            "mg2_tq_Nm": sol.mg2_tq_Nm,

            "P_eng_mech_W": sol.P_eng_mech_W,
            "P_mg1_mech_W": sol.P_mg1_mech_W,
            "P_mg2_mech_W": sol.P_mg2_mech_W,
            "P_mg1_elec_W": sol.P_mg1_elec_W,
            "P_mg2_elec_W": sol.P_mg2_elec_W,

            "P_hvac_W": float(P_hvac),
            "P_aux_W": sol.P_aux_W,
            "P_batt_req_W": sol.P_batt_req_W,
            "P_batt_act_W": sol.P_batt_act_W,

            "P_brake_fric_W": sol.P_brake_fric_W,
            "shortfall_tq_Nm": sol.shortfall_tq_Nm,
            "shortfall_power_W": sol.shortfall_power_W,

            "soc_pct": sol.soc_next * 100.0,
            "E_batt_Wh": sol.E_batt_next_Wh,

            "lim_batt_discharge_W": xin.P_dis_max_W,
            "lim_batt_charge_W": xin.P_chg_max_W,

            "J_total": sol.J_total,
            "J_fuel": sol.J_fuel,
            "J_soc": sol.J_soc,
            "J_fric": sol.J_fric,
            "J_short": sol.J_short,
            "J_spin": sol.J_spin,
            "J_smooth": sol.J_smooth,
            "J_charge": sol.J_charge,
            "excess_tq_Nm": sol.excess_tq_Nm,   # StepResultに入れる場合
            "J_over": sol.J_over,

            "n_grid_total": sol.stats.n_total,
            "n_grid_kept": sol.stats.n_kept,
            "batt_E_usable_Wh": batt.E_usable_Wh,
            "batt_Emin_Wh": batt.soc_min * batt.E_usable_Wh,
            "batt_Emax_Wh": batt.soc_max * batt.E_usable_Wh,
            # flags (for audit.py)
            "flag_eng_sat": int(sol.fuel_cut == 0 and abs(sol.eng_tq_Nm) >= (veh.eng_tq_max_Nm - 1e-9)),
            "flag_mg1_sat": int(abs(sol.mg1_tq_Nm) >= (veh.mg1_tq_max_Nm - 1e-9)),
            "flag_mg2_sat": int(abs(sol.mg2_tq_Nm) >= (veh.mg2_tq_max_Nm - 1e-9)),
            "flag_batt_sat": int(abs(sol.P_batt_req_W - sol.P_batt_act_W) > 1e-6),
            "flag_mg1_overspeed": int(abs(sol.mg1_rpm) > (veh.mg1_rpm_max + 1e-6)),
            "flag_mg2_overspeed": int(abs(sol.mg2_rpm) > (veh.mg2_rpm_max + 1e-6)),
            # torque balance visibility
            "T_ring_deliv_Nm": T_ring_deliv,  # beta*Teng + Tmg2
            "resid_ring_torque_Nm_dbg": float(xin.T_ring_req_Nm - T_ring_deliv - sol.shortfall_tq_Nm),

            # wheel power terms
            "P_wheel_deliv_W_dbg": float(P_wheel_deliv_W),
            "resid_wheel_power_W_dbg": float(xin.P_wheel_req_W - P_wheel_deliv_W + sol.P_brake_fric_W),

            # also log mapping components to confirm base-plane
            "wheel_omega_radps_dbg": float(wheel_omega),
            "ring_omega_radps_dbg": float(xin.ring_omega_radps),
            "T_wheel_req_Nm_dbg": float(T_wheel_req),
            "T_ring_req_Nm_dbg": float(xin.T_ring_req_Nm),
        })

        # save constraints (audit residuals)
        cres.append({
            "t_s": xin.t_s,
            "phase": xin.phase,
            "resid_psd_speed_rpm": sol.resid_psd_speed_rpm,
            "resid_elec_power_W": sol.resid_elec_power_W,
            "resid_ring_torque_Nm": sol.resid_ring_torque_Nm,
            # SOC recon residual is computed later by full-run audit (re-integration)
            "resid_wheel_power_W": float(xin.P_wheel_req_W - P_wheel_deliv_W + sol.P_brake_fric_W),
        })

    return pd.DataFrame(rows), pd.DataFrame(cres)