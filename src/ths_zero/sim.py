from __future__ import annotations

import numpy as np
import pandas as pd

from .configs import CommonConfig, VehicleConfig, BatteryConfig, InitialState, EnvironmentConfig
from .environment import air_density_kgpm3, hvac_power_W, battery_temp_C, batt_power_limits_W
from .engine_maps import pick_engine_point_min_bsfc


def mech_to_elec_power(P_mech_W: float, eta: float) -> float:
    """
    Returns electrical power (positive = consumes electrical, negative = generates electrical).
    """
    if P_mech_W >= 0:
        return float(P_mech_W / max(eta, 1e-9))
    return float(P_mech_W * max(eta, 1e-9))


def decide_engine_on(P_wheel_req_W: float, v_mps: float, soc: float, common: CommonConfig) -> bool:
    soc_low = common.soc_target - common.soc_band
    if v_mps < common.eng_on_min_speed_mps:
        return False
    if P_wheel_req_W > common.eng_on_wheel_power_W:
        return True
    if soc < soc_low:
        return True
    return False


def mode_label(eng_on: bool, P_wheel_req_W: float, P_brake_fric_W: float) -> str:
    if P_wheel_req_W < 0:
        if P_brake_fric_W > 1.0:
            return "FrictionBrake"
        return "Regen"
    if eng_on:
        return "HybridDrive"
    return "EV"


def simulate_ths(
    wltc: pd.DataFrame,
    common: CommonConfig,
    veh: VehicleConfig,
    batt: BatteryConfig,
    init: InitialState,
    env: EnvironmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      timeseries_df: main signals
      constraints_df: residuals/constraints for audit
    """

    # PSD constants
    alpha = veh.Zs / (veh.Zs + veh.Zr)
    beta  = veh.Zr / (veh.Zs + veh.Zr)

    # battery state
    soc = float(init.soc0)
    E_batt_Wh = soc * batt.E_usable_Wh

    rows = []
    cres = []

    t = wltc["t_s"].to_numpy(dtype=float)
    dt = wltc["dt_s"].to_numpy(dtype=float)
    v = wltc["veh_spd_mps"].to_numpy(dtype=float)
    a = wltc["veh_acc_mps2"].to_numpy(dtype=float)
    phase = wltc["phase"].astype(str).to_numpy()

    for k in range(len(t)):
        Tamb_C = float(env.Tamb_C)  # Level 1: constant
        rho = air_density_kgpm3(Tamb_C, env.p_amb_Pa, common)

        P_hvac = hvac_power_W(Tamb_C, env)
        P_aux = batt.P_aux_base_W + P_hvac

        Tbatt_C = battery_temp_C(Tamb_C, env)
        P_dis_max, P_chg_max = batt_power_limits_W(Tbatt_C, batt)  # charge max is magnitude

        # --- vehicle longitudinal forces
        F_roll = veh.mass_kg * common.g * veh.Crr
        F_aero = 0.5 * rho * veh.CdA * (v[k] ** 2)
        F_iner = veh.mass_kg * a[k]
        F_total = F_roll + F_aero + F_iner

        T_wheel_req = F_total * veh.tire_radius_m
        P_wheel_req = F_total * v[k]

        # --- driveline kinematics
        wheel_omega = v[k] / max(veh.tire_radius_m, 1e-9)  # rad/s
        ring_omega = wheel_omega * veh.final_drive
        ring_rpm = ring_omega * 60.0 / (2.0 * np.pi)

        # ring demand torque (simple mapping)
        T_ring_demand = T_wheel_req / max(veh.final_drive, 1e-9)

        # --- initialize state for this step
        eng_on = decide_engine_on(P_wheel_req, v[k], soc, common) if P_wheel_req > 0 else False
        cmd_P_eng = 0.0

        # Contributions
        eng_rpm = 0.0
        eng_tq = 0.0
        mg1_rpm = 0.0
        mg2_rpm = ring_rpm

        # PSD part to ring (simplified but power-auditable)
        T_psd_to_ring = 0.0

        # MG torques
        mg1_tq = 0.0
        mg2_tq = 0.0

        # friction brake (positive dissipated)
        P_brake_fric = 0.0

        # --- traction / braking split
        if P_wheel_req < 0:
            # braking: try regen with MG2, limited by MG2 torque and battery acceptance
            eng_on = False
            eng_rpm = 0.0
            eng_tq = 0.0
            T_psd_to_ring = 0.0
            mg1_rpm = 0.0
            mg1_tq = 0.0

            mg2_tq = float(np.clip(T_ring_demand, -veh.mg2_tq_max_Nm, veh.mg2_tq_max_Nm))
            P_mg2_mech = mg2_tq * ring_omega
            P_mg2_elec = mech_to_elec_power(P_mg2_mech, veh.eta_mg2)

            # net battery request with regen
            P_batt_req = P_mg2_elec + P_aux

            # if too much charging (too negative), scale down regen to meet charge acceptance
            P_batt_min = -P_chg_max  # most negative allowed
            if P_batt_req < P_batt_min:
                # scale factor on mg2 torque (regen magnitude)
                # want: s*P_mg2_elec + P_aux = P_batt_min
                # note: P_mg2_elec is negative here
                denom = P_mg2_elec
                if abs(denom) > 1e-6:
                    s = (P_batt_min - P_aux) / denom
                    s = float(np.clip(s, 0.0, 1.0))
                    mg2_tq *= s
                    P_mg2_mech = mg2_tq * ring_omega
                    P_mg2_elec = mech_to_elec_power(P_mg2_mech, veh.eta_mg2)
                    P_batt_req = P_mg2_elec + P_aux

            # wheel delivered from ring (simple)
            T_ring_deliv = mg2_tq  # PSD part zero
            P_ring_deliv = T_ring_deliv * ring_omega
            P_wheel_deliv = P_ring_deliv * veh.driveline_eff

            # friction needed to match WLTC braking demand
            P_brake_fric = max(0.0, P_wheel_deliv - P_wheel_req)
            # clamp battery actual power (should already respect charge limit, but keep robust)
            P_batt_act = float(np.clip(P_batt_req, -P_chg_max, P_dis_max))

        else:
            # traction or coast
            # refinement loop to reduce ring torque residual by increasing engine command
            P_batt_act = 0.0
            P_batt_req = 0.0
            T_ring_deliv = 0.0
            P_wheel_deliv = 0.0
            P_ring_deliv = 0.0
            P_mg1_mech = 0.0
            P_mg2_mech = 0.0
            P_mg1_elec = 0.0
            P_mg2_elec = 0.0
            P_eng_mech = 0.0
            P_loop = 0.0

            for _ in range(max(common.traction_refine_iters, 1)):
                # engine command
                if eng_on:
                    soc_err = max(common.soc_target - soc, 0.0)  # only charge when SOC low
                    P_charge = common.soc_charge_gain_W_per_soc * soc_err
                    # engine covers wheel + (optional) charging
                    cmd_P_eng = max(P_wheel_req / max(veh.driveline_eff, 1e-9), 0.0) + P_charge
                else:
                    cmd_P_eng = 0.0

                # pick engine operating point
                if eng_on and cmd_P_eng > 1.0:
                    eng_rpm, eng_tq = pick_engine_point_min_bsfc(
                        P_eng_cmd_W=cmd_P_eng,
                        ring_spd_rpm=ring_rpm,
                        alpha=alpha, beta=beta,
                        eng_rpm_min=veh.eng_rpm_min,
                        eng_rpm_max=veh.eng_rpm_max,
                        eng_tq_max_Nm=veh.eng_tq_max_Nm,
                        mg1_rpm_max=veh.mg1_rpm_max,
                    )
                else:
                    eng_rpm, eng_tq = 0.0, 0.0

                # kinematics
                omega_eng = eng_rpm * 2.0 * np.pi / 60.0
                omega_ring = ring_omega
                omega_mg1 = 0.0
                if eng_rpm > 0:
                    omega_mg1 = (omega_eng - beta * omega_ring) / max(alpha, 1e-9)
                mg1_rpm = omega_mg1 * 60.0 / (2.0 * np.pi)

                # engine mech power
                P_eng_mech = eng_tq * omega_eng

                # PSD ring torque contribution (simplified)
                # NOTE: This is a simplification; audit residuals will tell feasibility.
                T_psd_to_ring = beta * eng_tq

                # MG2 torque needed to meet ring demand
                mg2_tq_cmd = T_ring_demand - T_psd_to_ring
                mg2_tq = float(np.clip(mg2_tq_cmd, -veh.mg2_tq_max_Nm, veh.mg2_tq_max_Nm))

                # delivered ring torque
                T_ring_deliv = T_psd_to_ring + mg2_tq

                # ring power delivered
                P_ring_deliv = T_ring_deliv * omega_ring
                P_wheel_deliv = P_ring_deliv * veh.driveline_eff

                # PSD power split: ring_psd power is T_psd_to_ring*omega_ring
                P_ring_psd = T_psd_to_ring * omega_ring

                # MG1 gets remainder of engine power in PSD
                P_mg1_mech = P_eng_mech - P_ring_psd
                # MG1 torque from power
                if abs(omega_mg1) > 1e-6:
                    mg1_tq_cmd = P_mg1_mech / omega_mg1
                else:
                    mg1_tq_cmd = 0.0
                mg1_tq = float(np.clip(mg1_tq_cmd, -veh.mg1_tq_max_Nm, veh.mg1_tq_max_Nm))
                P_mg1_mech = mg1_tq * omega_mg1  # actual after saturation

                # MG2 mech power
                P_mg2_mech = mg2_tq * omega_ring

                # electrical conversions
                P_mg1_elec = mech_to_elec_power(P_mg1_mech, veh.eta_mg1)
                P_mg2_elec = mech_to_elec_power(P_mg2_mech, veh.eta_mg2)

                # battery required/actual (actual clipped)
                P_batt_req = P_mg1_elec + P_mg2_elec + P_aux
                P_batt_act = float(np.clip(P_batt_req, -P_chg_max, P_dis_max))

                # loop power proxy: overlap of MG1 generation and MG2 motoring
                P_loop = 0.0
                if (P_mg1_elec < 0) and (P_mg2_elec > 0):
                    P_loop = min(-P_mg1_elec, P_mg2_elec)

                # if ring torque residual still positive (cannot meet demand), try enabling engine (or increase command next iter)
                resid_ring_tq = T_ring_demand - T_ring_deliv
                if (resid_ring_tq > 1.0) and (not eng_on) and (P_wheel_req > common.eng_on_wheel_power_W):
                    eng_on = True
                    continue
                # otherwise stop refinement
                break

            # in traction/coast, friction brake is zero by definition
            P_brake_fric = 0.0

        # --- SOC update using ACTUAL battery power
        # Positive discharge reduces energy
        E_batt_Wh_unclipped = E_batt_Wh - (P_batt_act * dt[k]) / 3600.0
        E_batt_Wh = float(np.clip(
            E_batt_Wh_unclipped,
            batt.soc_min * batt.E_usable_Wh,
            batt.soc_max * batt.E_usable_Wh,
        ))
        soc = E_batt_Wh / batt.E_usable_Wh

        # --- flags
        flag_eng_sat = int(abs(eng_tq) >= veh.eng_tq_max_Nm - 1e-6) if eng_on else 0
        flag_mg1_sat = int(abs(mg1_tq) >= veh.mg1_tq_max_Nm - 1e-6)
        flag_mg2_sat = int(abs(mg2_tq) >= veh.mg2_tq_max_Nm - 1e-6)
        flag_mg1_overspeed = int(abs(mg1_rpm) > veh.mg1_rpm_max + 1e-6)
        flag_mg2_overspeed = int(abs(mg2_rpm) > veh.mg2_rpm_max + 1e-6)
        flag_batt_sat = int(abs(P_batt_req - P_batt_act) > 1e-6)

        # --- derive remaining powers for reporting (ensure defined in both branches)
        omega_eng = eng_rpm * 2.0 * np.pi / 60.0
        omega_mg1 = mg1_rpm * 2.0 * np.pi / 60.0
        omega_ring = ring_omega

        P_eng_mech = eng_tq * omega_eng
        P_mg1_mech = mg1_tq * omega_mg1
        P_mg2_mech = mg2_tq * omega_ring
        P_mg1_elec = mech_to_elec_power(P_mg1_mech, veh.eta_mg1)
        P_mg2_elec = mech_to_elec_power(P_mg2_mech, veh.eta_mg2)

        P_loop = 0.0
        if (P_mg1_elec < 0) and (P_mg2_elec > 0):
            P_loop = min(-P_mg1_elec, P_mg2_elec)
        loop_ratio = float(P_loop / max(P_eng_mech, 1.0)) if eng_on else 0.0

        # wheel delivered computed from ring
        T_ring_deliv = T_psd_to_ring + mg2_tq
        P_ring_deliv = T_ring_deliv * omega_ring
        P_wheel_deliv = P_ring_deliv * veh.driveline_eff

        # --- constraints / residuals (AUDIT)
        resid_psd_speed_rpm = float(eng_rpm - (alpha * mg1_rpm + beta * ring_rpm))
        resid_ring_torque_Nm = float(T_ring_demand - T_ring_deliv)
        resid_wheel_power_W = float(P_wheel_req - P_wheel_deliv - (-P_brake_fric))  # since P_wheel_req = P_wheel_deliv - P_fric  -> resid = P_req - P_deliv + P_fric
        resid_wheel_power_W = float(P_wheel_req - P_wheel_deliv + P_brake_fric)
        resid_elec_power_W = float(P_batt_act - (P_mg1_elec + P_mg2_elec + P_aux))

        rows.append({
            # time & wltc
            "phase": phase[k],
            "t_s": float(t[k]),
            "dt_s": float(dt[k]),
            "veh_spd_mps": float(v[k]),
            "veh_spd_kmh": float(wltc["veh_spd_kmh"].iloc[k]),
            "veh_acc_mps2": float(a[k]),

            # environment
            "Tamb_C": Tamb_C,
            "rho_air_kgpm3": float(rho),
            "Tbatt_C": float(Tbatt_C),

            # longitudinal
            "F_roll_N": float(F_roll),
            "F_aero_N": float(F_aero),
            "F_iner_N": float(F_iner),
            "F_total_N": float(F_total),
            "T_wheel_req_Nm": float(T_wheel_req),
            "P_wheel_req_W": float(P_wheel_req),
            "P_brake_fric_W": float(P_brake_fric),

            # driveline / ring
            "wheel_omega_radps": float(wheel_omega),
            "ring_omega_radps": float(ring_omega),
            "ring_spd_rpm": float(ring_rpm),
            "T_ring_demand_Nm": float(T_ring_demand),

            # THS states
            "eng_on": int(eng_on),
            "mode": mode_label(eng_on, P_wheel_req, P_brake_fric),

            "alpha": float(alpha),
            "beta": float(beta),

            "eng_spd_rpm": float(eng_rpm),
            "eng_tq_Nm": float(eng_tq),

            "mg1_spd_rpm": float(mg1_rpm),
            "mg1_tq_Nm": float(mg1_tq),

            "mg2_spd_rpm": float(mg2_rpm),
            "mg2_tq_Nm": float(mg2_tq),

            # torque split
            "T_psd_to_ring_Nm": float(T_psd_to_ring),
            "T_ring_deliv_Nm": float(T_ring_deliv),

            # mech powers
            "P_wheel_deliv_W": float(P_wheel_deliv),
            "P_eng_mech_W": float(P_eng_mech),
            "P_mg1_mech_W": float(P_mg1_mech),
            "P_mg2_mech_W": float(P_mg2_mech),
            "P_loop_W": float(P_loop),
            "loop_ratio": float(loop_ratio),

            # elec powers
            "P_mg1_elec_W": float(P_mg1_elec),
            "P_mg2_elec_W": float(P_mg2_elec),
            "P_hvac_W": float(P_hvac),
            "P_aux_W": float(P_aux),
            "P_batt_req_W": float(P_batt_req),
            "P_batt_act_W": float(P_batt_act),

            # batt limits
            "lim_batt_discharge_W": float(P_dis_max),
            "lim_batt_charge_W": float(P_chg_max),

            # battery state
            "soc_pct": float(soc * 100.0),
            "E_batt_Wh": float(E_batt_Wh),
            "E_batt_Wh_unclipped": float(E_batt_Wh_unclipped),

            # limits (pointwise)
            "lim_eng_tq_Nm": float(veh.eng_tq_max_Nm),
            "lim_mg1_tq_Nm": float(veh.mg1_tq_max_Nm),
            "lim_mg2_tq_Nm": float(veh.mg2_tq_max_Nm),
            "lim_mg1_spd_rpm": float(veh.mg1_rpm_max),
            "lim_mg2_spd_rpm": float(veh.mg2_rpm_max),

            # flags
            "flag_eng_sat": int(flag_eng_sat),
            "flag_mg1_sat": int(flag_mg1_sat),
            "flag_mg2_sat": int(flag_mg2_sat),
            "flag_batt_sat": int(flag_batt_sat),
            "flag_mg1_overspeed": int(flag_mg1_overspeed),
            "flag_mg2_overspeed": int(flag_mg2_overspeed),
        })

        cres.append({
            "t_s": float(t[k]),
            "phase": phase[k],
            "resid_psd_speed_rpm": resid_psd_speed_rpm,
            "resid_ring_torque_Nm": resid_ring_torque_Nm,
            "resid_wheel_power_W": resid_wheel_power_W,
            "resid_elec_power_W": resid_elec_power_W,
            # SOC reconstruction residual computed later in audit module (integral check)
        })

    ts = pd.DataFrame(rows)
    cons = pd.DataFrame(cres)
    return ts, cons