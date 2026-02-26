# ths_zero/environment.py
from __future__ import annotations
import numpy as np

R_AIR = 287.05  # J/(kg*K)

def air_density_kgpm3(Tamb_C: float, p_amb_Pa: float = 101325.0, common=None) -> float:
    """Ideal gas air density. common は互換のため未使用でもOK."""
    T_K = Tamb_C + 273.15
    return float(p_amb_Pa / (R_AIR * T_K))

def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))

def hvac_power_W(Tamb_C: float, env) -> float:
    """
    Cabin HVAC electrical power model (simple but realistic order of magnitude).

    - Cooling at hot ambient, heating at cold ambient.
    - Uses COP that degrades at very hot/cold.
    - Adds blower/base power.
    """
    T_set = getattr(env, "cabin_setpoint_C", getattr(env, "T_cabin_set_C", 23.0))
    blower_W = getattr(env, "hvac_blower_W", 150.0)
    max_W = getattr(env, "hvac_Pmax_W", getattr(env, "hvac_max_W", 3500.0))

    # Thermal load (W_th) per K: tuned to yield ~1–2kW electrical at 10C/35C
    base_th_W = getattr(env, "hvac_base_thermal_W", 400.0)
    # configs.py では *_W_per_C を使っているので優先
    k_cool_th = getattr(env, "hvac_k_cool_W_per_C",
                 getattr(env, "hvac_cool_thermal_W_perK", 250.0))
    k_heat_th = getattr(env, "hvac_k_heat_W_per_C",
                 getattr(env, "hvac_heat_thermal_W_perK", 200.0))

    dT = Tamb_C - T_set

    # COP models (very simplified)
    # Cooling COP: ~3.0 at 25C, drops as ambient gets hotter
    cop_cool_25 = getattr(env, "hvac_cop_cool_25C", 3.0)
    cop_cool = cop_cool_25 - 0.05 * max(0.0, Tamb_C - 25.0)
    cop_cool = _clip(cop_cool, 1.8, 3.5)

    # Heating COP: ~3.2 at 10C, drops quickly below 10C
    cop_heat_10 = getattr(env, "hvac_cop_heat_10C", 3.2)
    cop_heat = cop_heat_10 - 0.08 * max(0.0, 10.0 - Tamb_C)
    cop_heat = _clip(cop_heat, 1.3, 3.8)

    # Optional PTC assist for very cold (defrost etc.)
    ptc_enable_C = getattr(env, "ptc_enable_C", 5.0)
    ptc_W = getattr(env, "ptc_W", 1200.0)

    if dT >= 0.0:
        # Cooling
        Q_th = base_th_W + k_cool_th * dT
        P = blower_W + Q_th / max(cop_cool, 1e-6)
    else:
        # Heating
        Q_th = base_th_W + k_heat_th * (-dT)
        P = blower_W + Q_th / max(cop_heat, 1e-6)
        if Tamb_C < ptc_enable_C:
            P += ptc_W

    return float(np.clip(P, 0.0, max_W))

def battery_temp_C(Tamb_C: float, env) -> float:
    """
    Simple battery temperature proxy.
    - No thermal state yet (no lag).
    - Adds heat-soak offset at hot ambient.
    """
    model = getattr(env, "battery_temp_model", "ambient")
    bias = getattr(env, "Tbatt_bias_C", 0.0)
    if model == "fixed":
        return float(getattr(env, "Tbatt_fixed_C", 25.0))
    # default: ambient
    return float(Tamb_C + bias)

def batt_power_limits_W(Tbatt_C: float, batt) -> tuple[float, float]:
    """
    Temperature-dependent battery power limits (simple piecewise-linear).

    Returns:
      P_dis_max_W (positive)
      P_chg_max_W (positive magnitude)
    """
    # Base ratings (fallbacks)
    # configs.py の命名を優先（互換で旧名も拾う）
    P_dis_ref = float(getattr(batt, "P_discharge_nom_W",
                      getattr(batt, "P_dis_ref_W",
                      getattr(batt, "P_dis_max_W", 30000.0))))
    P_chg_ref = float(getattr(batt, "P_charge_nom_W",
                      getattr(batt, "P_chg_ref_W",
                      getattr(batt, "P_chg_max_W", 20000.0))))

    # Charge acceptance is more temperature-sensitive than discharge.
    # x-axis: temperature [C], y-axis: scale factor
    T_pts_chg = np.array([-10, 0, 10, 20, 25, 35, 45, 55], dtype=float)
    S_pts_chg = np.array([0.15, 0.35, 0.60, 0.85, 1.00, 1.00, 0.80, 0.30], dtype=float)

    T_pts_dis = np.array([-10, 0, 10, 25, 45, 55], dtype=float)
    # Cold discharge derate was too strict for WLTC feasibility at -10C.
    # Raise low-temp scale factors to reduce traction shortfall.
    S_pts_dis = np.array([0.85, 0.90, 0.95, 1.00, 0.85, 0.55], dtype=float)

    s_chg = float(np.interp(Tbatt_C, T_pts_chg, S_pts_chg))
    s_dis = float(np.interp(Tbatt_C, T_pts_dis, S_pts_dis))

    P_chg_max = P_chg_ref * s_chg
    P_dis_max = P_dis_ref * s_dis

    # Safety clamps
    P_chg_max = float(np.clip(P_chg_max, 1000.0, P_chg_ref))
    P_dis_max = float(np.clip(P_dis_max, 2000.0, P_dis_ref))

    return P_dis_max, P_chg_max