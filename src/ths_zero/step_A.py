# ths_zero/step_A.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import numpy as np

# ============================================================
# Sign conventions (GLOBAL)
# ------------------------------------------------------------
# Battery power:   P_batt_W > 0 => discharge (SOC down)
#                 P_batt_W < 0 => charge    (SOC up)
# Electrical power: P_elec_W > 0 => consumes electrical
#                   P_elec_W < 0 => generates electrical
# Ring speed: omega_ring > 0 is forward
# Braking friction: P_brake_fric_W >= 0 (dissipated heat)
# ============================================================


# --- Map interfaces
EtaMap = Callable[[float, float], float]     # (speed_rpm, torque_Nm) -> eta in (0,1]
BsfcMap = Callable[[float, float], float]    # (eng_rpm, eng_tq_Nm) -> g/kWh
EngTqMaxMap = Callable[[float], float]       # eng_rpm -> Tmax Nm (>=0)
EngDragMinMap = Callable[[float], float]     # eng_rpm -> Tmin Nm (<=0) during fuel-cut (optional)


# ============================================================
# Fixed I/Fs (previously agreed)
# ============================================================

@dataclass(frozen=True)
class ElecConvResult:
    P_elec_W: float   # + consumes electrical, - generates electrical
    loss_W: float     # >=0


def mech_to_elec(P_mech_W: float, eta: float) -> ElecConvResult:
    """
    Mechanical -> electrical conversion with sign-consistent loss.

    P_mech_W > 0 : motoring  -> P_elec = P_mech/eta, loss = P_elec - P_mech
    P_mech_W < 0 : generating -> P_elec = P_mech*eta, loss = P_mech - P_elec  (>=0)
    """
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0,1], got {eta}")

    if P_mech_W >= 0:
        P_elec = P_mech_W / eta
        loss = P_elec - P_mech_W
    else:
        P_elec = P_mech_W * eta
        loss = P_mech_W - P_elec  # both negative -> loss >=0

    loss = float(max(loss, 0.0))
    return ElecConvResult(float(P_elec), loss)


@dataclass(frozen=True)
class FuelResult:
    mdot_fuel_gps: float     # [g/s]
    bsfc_g_per_kwh: float    # [g/kWh]


def fuel_model_bsfc(eng_rpm: float, eng_tq_Nm: float, bsfc_g_per_kwh: float) -> FuelResult:
    """
    Fuel rate from BSFC.

    mdot[g/s] = bsfc[g/kWh] * P_eng[kW] / 3600
    For eng_rpm<=0 or eng_tq<=0 -> mdot = 0 (fuel-cut / no positive work)
    """
    if eng_rpm <= 0 or eng_tq_Nm <= 0:
        return FuelResult(0.0, float(bsfc_g_per_kwh))

    omega = eng_rpm * 2.0 * np.pi / 60.0
    P_kW = (eng_tq_Nm * omega) / 1000.0
    mdot = float(bsfc_g_per_kwh * P_kW / 3600.0)
    return FuelResult(mdot, float(bsfc_g_per_kwh))


@dataclass(frozen=True)
class RegenFrictionResult:
    T_mg2_Nm: float
    P_batt_req_W: float
    P_batt_act_W: float
    P_brake_fric_W: float
    flag_batt_charge_limited: int


def enforce_charge_limit_with_friction(
    *,
    P_wheel_req_W: float,          # negative in braking
    T_ring_req_Nm: float,          # negative in braking (for omega_ring>0)
    omega_ring_radps: float,       # ring speed (rad/s)
    T_psd_to_ring_Nm: float,       # ring torque contribution from PSD (can be 0)
    T_mg2_max_Nm: float,
    eta_mg2: EtaMap,               # (rpm, tq)->eta
    P_aux_W: float,
    P_chg_max_W: float,            # magnitude, positive
) -> RegenFrictionResult:
    """
    Braking: enforce battery charge acceptance by reducing regen and adding friction.

    Ensures:
      P_batt_act_W >= -P_chg_max_W
      Remaining braking -> P_brake_fric_W >= 0
    """
    if P_wheel_req_W > 1e-6:
        raise ValueError("This function is for braking (P_wheel_req_W <= 0).")
    if P_chg_max_W < 0:
        raise ValueError("P_chg_max_W must be >=0.")
    if abs(omega_ring_radps) < 1e-12:
        # near zero speed: no meaningful regen power
        return RegenFrictionResult(
            T_mg2_Nm=0.0,
            P_batt_req_W=P_aux_W,
            P_batt_act_W=max(P_aux_W, -P_chg_max_W),
            P_brake_fric_W=float(max(0.0, -P_wheel_req_W)),
            flag_batt_charge_limited=1,
        )

    # (1) torque command + limit
    T_mg2_cmd = T_ring_req_Nm - T_psd_to_ring_Nm
    T_mg2 = float(np.clip(T_mg2_cmd, -T_mg2_max_Nm, +T_mg2_max_Nm))

    # (2) mg2 electrical power
    ring_rpm = omega_ring_radps * 60.0 / (2.0 * np.pi)
    P_mg2_mech = T_mg2 * omega_ring_radps
    eta = float(eta_mg2(ring_rpm, T_mg2))
    P_mg2_elec = mech_to_elec(P_mg2_mech, eta).P_elec_W

    P_batt_req = P_mg2_elec + P_aux_W
    P_batt_min = -float(P_chg_max_W)

    flag = 0
    P_batt_act = P_batt_req

    # (3) if too much charge (too negative), scale regen magnitude toward zero
    if P_batt_req < P_batt_min - 1e-9:
        flag = 1
        denom = P_mg2_elec  # negative
        if abs(denom) > 1e-9:
            s = (P_batt_min - P_aux_W) / denom
            s = float(np.clip(s, 0.0, 1.0))
        else:
            s = 0.0

        T_mg2 *= s
        P_mg2_mech = T_mg2 * omega_ring_radps
        eta = float(eta_mg2(ring_rpm, T_mg2))
        P_mg2_elec = mech_to_elec(P_mg2_mech, eta).P_elec_W
        P_batt_act = max(P_mg2_elec + P_aux_W, P_batt_min)

    # (4) friction = remaining braking not achieved by regen
    T_ring_deliv = T_psd_to_ring_Nm + T_mg2
    P_wheel_deliv = T_ring_deliv * omega_ring_radps  # negative in braking

    # braking demand is negative: we want P_wheel_deliv ~= P_wheel_req (both negative)
    # remaining magnitude -> friction positive
    P_brake_fric = float(max(0.0, -(P_wheel_req_W - P_wheel_deliv)))

    return RegenFrictionResult(
        T_mg2_Nm=float(T_mg2),
        P_batt_req_W=float(P_batt_req),
        P_batt_act_W=float(P_batt_act),
        P_brake_fric_W=float(P_brake_fric),
        flag_batt_charge_limited=int(flag),
    )


@dataclass(frozen=True)
class SocUpdateResult:
    E_batt_Wh: float
    soc: float  # 0..1


def soc_update(
    *,
    E_batt_Wh: float,
    P_batt_W: float,        # + discharge, - charge
    dt_s: float,
    E_usable_Wh: float,
    Emin_Wh: float,
    Emax_Wh: float,
) -> SocUpdateResult:
    """
    Energy integration with clipping.

    E_next = clip(E - P_batt*dt/3600, Emin, Emax)
    soc_next = E_next / E_usable
    """
    dE_Wh = (P_batt_W * dt_s) / 3600.0
    E_next = float(np.clip(E_batt_Wh - dE_Wh, Emin_Wh, Emax_Wh))
    soc_next = float(E_next / max(E_usable_Wh, 1e-9))
    soc_next = float(np.clip(soc_next, 0.0, 1.0))
    return SocUpdateResult(E_next, soc_next)


# ============================================================
# Step solver (A: grid search)
# ============================================================

@dataclass(frozen=True)
class StepInputs:
    # timebase
    t_s: float
    dt_s: float
    phase: str

    # ring / demand (already mapped from vehicle dynamics)
    ring_rpm: float
    ring_omega_radps: float
    T_ring_req_Nm: float          # + traction, - braking
    P_wheel_req_W: float          # + traction, - braking

    # state
    soc: float                    # 0..1
    E_batt_Wh: float

    # battery constants (IMPORTANT: never infer!)
    E_usable_Wh: float
    Emin_Wh: float
    Emax_Wh: float

    # environment / aux
    P_aux_W: float                # >=0
    Tamb_C: float
    Tbatt_C: float

    # battery limits (per-step after temp derate)
    P_dis_max_W: float            # + discharge limit
    P_chg_max_W: float            # charging acceptance magnitude (+)

    # component limits
    mg1_rpm_max: float
    mg2_rpm_max: float
    mg1_tq_max_Nm: float
    mg2_tq_max_Nm: float
    eng_rpm_min: float
    eng_rpm_max: float

    # PSD constants
    alpha: float
    beta: float

    # optional smoothing info
    prev_eng_rpm: Optional[float] = None
    prev_eng_tq_Nm: Optional[float] = None


@dataclass(frozen=True)
class StepWeights:
    w_fuel: float = 1.0
    w_soc: float = 2.0e4
    w_fric: float = 5.0e-4
    w_short_tq: float = 5.0e3
    w_spin: float = 5.0e-3
    w_smooth: float = 1.0e-4
    w_charge_track: float = 2.0e-4
    w_over_tq: float = 5.0e3


@dataclass(frozen=True)
class FeasibilityStats:
    n_total: int
    n_kept: int
    n_fail_mg1_speed: int
    n_fail_mg2_speed: int
    n_fail_mg1_tq: int
    n_fail_eng_tq: int


@dataclass(frozen=True)
class StepResult:
    mode: str
    fuel_cut: int

    eng_rpm: float
    eng_tq_Nm: float
    mg1_rpm: float
    mg1_tq_Nm: float
    mg2_rpm: float
    mg2_tq_Nm: float

    # powers
    P_eng_mech_W: float
    P_mg1_mech_W: float
    P_mg2_mech_W: float
    P_mg1_elec_W: float
    P_mg2_elec_W: float
    P_aux_W: float
    P_batt_req_W: float
    P_batt_act_W: float

    # explicit slack
    P_brake_fric_W: float
    shortfall_tq_Nm: float
    shortfall_power_W: float

    # ★追加（over-delivery）
    excess_tq_Nm: float
    J_over: float

    # state update
    E_batt_next_Wh: float
    soc_next: float

    # objective breakdown
    J_total: float
    J_fuel: float
    J_soc: float
    J_fric: float
    J_short: float
    J_spin: float
    J_smooth: float
    J_charge: float

    # audits (should be ~0 except slack-defined)
    resid_psd_speed_rpm: float
    resid_elec_power_W: float
    resid_ring_torque_Nm: float
    resid_soc_recon_pct: float   # set 0 here; whole-run audit computes by re-integration

    stats: FeasibilityStats


def omega_from_rpm(rpm: float) -> float:
    return float(rpm * 2.0 * np.pi / 60.0)


def choose_fuelcut_eng_rpm(
    ring_rpm: float,
    alpha: float,
    beta: float,
    mg1_rpm_max: float,
    eng_rpm_max: float,
) -> float:
    """
    Fuel-cut (no fuel), but PSD still constrains speeds.
    Choose eng_rpm that tries to keep mg1_rpm ~ 0:
      mg1_rpm=0 => eng_rpm = beta*ring_rpm
    If anything violates bounds, clamp via mg1 limit and recompute eng_rpm.
    """
    eng_rpm = beta * ring_rpm
    mg1_rpm = (eng_rpm - beta * ring_rpm) / max(alpha, 1e-9)  # 0 ideally
    if abs(mg1_rpm) <= mg1_rpm_max and 0.0 <= eng_rpm <= eng_rpm_max:
        return float(np.clip(eng_rpm, 0.0, eng_rpm_max))

    mg1_rpm = float(np.clip(mg1_rpm, -mg1_rpm_max, +mg1_rpm_max))
    eng_rpm = alpha * mg1_rpm + beta * ring_rpm
    return float(np.clip(eng_rpm, 0.0, eng_rpm_max))


def spin_penalty(eng_rpm: float, mg1_rpm: float, eng_rpm_max: float, mg1_rpm_max: float) -> float:
    return float((eng_rpm / max(eng_rpm_max, 1e-9)) ** 2 + (mg1_rpm / max(mg1_rpm_max, 1e-9)) ** 2)


def solve_step_A(
    x: StepInputs,
    *,
    weights: StepWeights,
    bsfc_map: BsfcMap,
    eng_tq_max_map: EngTqMaxMap,
    eta_mg1_map: EtaMap,
    eta_mg2_map: EtaMap,
    eng_drag_min_map: Optional[EngDragMinMap] = None,
    eng_rpm_step: float = 100.0,
    eng_tq_step: float = 5.0,
    soc_target: float = 0.55,
    soc_band: float = 0.05,
) -> StepResult:
    """
    A案（グリッド探索）1ステップ解法：最速版
    - 牽引(Hybrid/Charge)は「前ステップ近傍の局所探索」を先に実施
    - 局所で可行解が見つからない場合のみ「全探索」にフォールバック
    - EV(牽引)と減速(Regen/Friction)は原則1点評価（グリッドしない）
    """

    alpha, beta = x.alpha, x.beta
    ring_rpm = x.ring_rpm
    omega_ring = x.ring_omega_radps

    # -----------------------------
    # mode candidates
    # -----------------------------
    if x.P_wheel_req_W < 0:
        modes = ["Regen", "FrictionBrake"]
    else:
        modes = ["EV", "HybridDrive"]
        if x.soc < (soc_target - soc_band):
            modes.append("Charge")

    best: Optional[Dict[str, Any]] = None

    # stats
    n_total = n_kept = 0
    n_fail_mg1_speed = n_fail_mg2_speed = 0
    n_fail_mg1_tq = n_fail_eng_tq = 0

    def eval_candidate(mode: str, eng_rpm: float, eng_tq: float) -> Optional[Dict[str, Any]]:
        nonlocal n_total, n_kept, n_fail_mg1_speed, n_fail_mg2_speed, n_fail_mg1_tq, n_fail_eng_tq
        n_total += 1

        # PSD kinematics
        mg1_rpm = (eng_rpm - beta * ring_rpm) / max(alpha, 1e-9)
        if abs(mg1_rpm) > x.mg1_rpm_max + 1e-9:
            n_fail_mg1_speed += 1
            return None

        mg2_rpm = ring_rpm
        if abs(mg2_rpm) > x.mg2_rpm_max + 1e-9:
            n_fail_mg2_speed += 1
            return None

        # engine torque bounds
        if mode in ["HybridDrive", "Charge"]:
            Tmax = float(eng_tq_max_map(eng_rpm))
            if eng_tq < -1e-9 or eng_tq > Tmax + 1e-9:
                n_fail_eng_tq += 1
                return None
        else:
            drag_min = float(eng_drag_min_map(eng_rpm)) if eng_drag_min_map else 0.0
            if eng_tq < drag_min - 1e-9 or eng_tq > 1e-9:
                n_fail_eng_tq += 1
                return None

        # PSD torque relation (quasi-static)
        mg1_tq = -alpha * eng_tq
        if abs(mg1_tq) > x.mg1_tq_max_Nm + 1e-9:
            n_fail_mg1_tq += 1
            return None

        # ring torque contribution from PSD
        T_psd_to_ring = beta * eng_tq

        # MG2 torque command
        mg2_tq_cmd = x.T_ring_req_Nm - T_psd_to_ring
        mg2_tq_act = float(np.clip(mg2_tq_cmd, -x.mg2_tq_max_Nm, +x.mg2_tq_max_Nm))

        # mechanical powers
        omega_eng = omega_from_rpm(eng_rpm)
        omega_mg1 = omega_from_rpm(mg1_rpm)

        P_eng_mech = eng_tq * omega_eng
        P_mg1_mech = mg1_tq * omega_mg1

        # MG1 electrical
        eta1 = float(eta_mg1_map(mg1_rpm, mg1_tq))
        P_mg1_elec = mech_to_elec(P_mg1_mech, eta1).P_elec_W

        # traction/battery-aware MG2 clamp (simple but fast)
        traction = (x.P_wheel_req_W >= 0)

        if traction:
            # Battery bounds translated to MG2 electrical power bounds
            # P_batt = P_mg1_elec + P_mg2_elec + P_aux
            P_mg2_elec_min = -x.P_chg_max_W - x.P_aux_W - P_mg1_elec  # (not too negative)
            P_mg2_elec_max = +x.P_dis_max_W - x.P_aux_W - P_mg1_elec  # (not too positive)

            if abs(omega_ring) > 1e-6:
                # Use a conservative eta for bounds (safe side)
                # You can also use eta_mg2_map(mg2_rpm, mg2_tq_cmd) if you want.
                eta_assume = float(np.clip(eta_mg2_map(mg2_rpm, mg2_tq_act), 0.80, 0.98))

                # Upper bound (motoring, tq>=0): P_elec = tq*omega/eta  -> tq <= P_elec_max*eta/omega
                tq_upper_batt = (P_mg2_elec_max * eta_assume) / omega_ring
                tq_upper_batt = float(np.clip(tq_upper_batt, -x.mg2_tq_max_Nm, +x.mg2_tq_max_Nm))

                # Lower bound (generating, tq<=0): P_elec = tq*omega*eta -> tq >= P_elec_min/(omega*eta)
                tq_lower_batt = P_mg2_elec_min / max(omega_ring * eta_assume, 1e-9)
                tq_lower_batt = float(np.clip(tq_lower_batt, -x.mg2_tq_max_Nm, +x.mg2_tq_max_Nm))

                # Apply BOTH bounds (this allows mg2_tq_act < 0 in traction when needed)
                mg2_tq_act = float(np.clip(mg2_tq_cmd, tq_lower_batt, tq_upper_batt))
            else:
                # near zero speed: only torque limit
                mg2_tq_act = float(np.clip(mg2_tq_cmd, -x.mg2_tq_max_Nm, +x.mg2_tq_max_Nm))

        # MG2 powers
        P_mg2_mech = mg2_tq_act * omega_ring
        eta2 = float(eta_mg2_map(mg2_rpm, mg2_tq_act))
        P_mg2_elec = mech_to_elec(P_mg2_mech, eta2).P_elec_W

        # battery request/actual
        P_batt_req = P_mg1_elec + P_mg2_elec + x.P_aux_W
        P_batt_act = float(np.clip(P_batt_req, -x.P_chg_max_W, x.P_dis_max_W))

        if traction:
            T_ring_deliv = T_psd_to_ring + mg2_tq_act
            shortfall_tq = float(max(0.0, x.T_ring_req_Nm - T_ring_deliv))
            excess_tq   = float(max(0.0, T_ring_deliv - x.T_ring_req_Nm))
            P_brake_fric = 0.0

        else:
            rf = enforce_charge_limit_with_friction(
                P_wheel_req_W=x.P_wheel_req_W,
                T_ring_req_Nm=x.T_ring_req_Nm,
                omega_ring_radps=omega_ring,
                T_psd_to_ring_Nm=T_psd_to_ring,
                T_mg2_max_Nm=x.mg2_tq_max_Nm,
                eta_mg2=eta_mg2_map,
                P_aux_W=x.P_aux_W,
                P_chg_max_W=x.P_chg_max_W,
            )
            mg2_tq_act = rf.T_mg2_Nm
            P_batt_req = rf.P_batt_req_W
            P_batt_act = rf.P_batt_act_W
            P_brake_fric = rf.P_brake_fric_W
            shortfall_tq = 0.0
            excess_tq = 0.0  # ← ブレーキでは over-penalty 無効
            #（必要なら T_ring_deliv はここで再計算してログ用に使う）

            # recompute MG2 power consistent with rf torque
            P_mg2_mech = mg2_tq_act * omega_ring
            eta2 = float(eta_mg2_map(mg2_rpm, mg2_tq_act))
            P_mg2_elec = mech_to_elec(P_mg2_mech, eta2).P_elec_W

        # SOC update (NO inference)
        su = soc_update(
            E_batt_Wh=x.E_batt_Wh,
            P_batt_W=P_batt_act,
            dt_s=x.dt_s,
            E_usable_Wh=x.E_usable_Wh,
            Emin_Wh=x.Emin_Wh,
            Emax_Wh=x.Emax_Wh,
        )
        soc_next = su.soc

        # objective
        fuel_cut = int(mode in ["EV", "Regen", "FrictionBrake"])
        if fuel_cut:
            mdot_fuel = 0.0
            J_fuel = 0.0
        else:
            bsfc_val = float(bsfc_map(eng_rpm, eng_tq))
            mdot_fuel = fuel_model_bsfc(eng_rpm, eng_tq, bsfc_val).mdot_fuel_gps
            J_fuel = weights.w_fuel * mdot_fuel

        J_soc = weights.w_soc * (soc_target - soc_next) ** 2
        J_fric = weights.w_fric * P_brake_fric
        J_short = weights.w_short_tq * (shortfall_tq ** 2)
        J_over = weights.w_over_tq * (excess_tq ** 2)
        J_spin = weights.w_spin * spin_penalty(eng_rpm, mg1_rpm, x.eng_rpm_max, x.mg1_rpm_max)
        
        J_smooth = 0.0
        if x.prev_eng_rpm is not None and x.prev_eng_tq_Nm is not None:
            J_smooth = weights.w_smooth * ((eng_rpm - x.prev_eng_rpm) ** 2 + (eng_tq - x.prev_eng_tq_Nm) ** 2)

        J_charge = 0.0
        if mode == "Charge":
            soc_err = max(0.0, soc_target - x.soc)
            P_charge_target = -min(x.P_chg_max_W, 20000.0 * soc_err)
            J_charge = weights.w_charge_track * (P_batt_act - P_charge_target) ** 2

        J_total = J_fuel + J_soc + J_fric + J_short + J_over + J_spin + J_smooth + J_charge

        # audits
        resid_psd_speed = float(eng_rpm - (alpha * mg1_rpm + beta * ring_rpm))
        resid_elec = float(P_batt_act - (P_mg1_elec + P_mg2_elec + x.P_aux_W))
        resid_ring = float(x.T_ring_req_Nm - (beta * eng_tq + mg2_tq_act) - shortfall_tq)
        shortfall_power = float(max(0.0, shortfall_tq * max(omega_ring, 0.0)))

        n_kept += 1
        return dict(
            mode=mode,
            fuel_cut=fuel_cut,
            eng_rpm=float(eng_rpm), eng_tq=float(eng_tq),
            mg1_rpm=float(mg1_rpm), mg1_tq=float(mg1_tq),
            mg2_rpm=float(mg2_rpm), mg2_tq=float(mg2_tq_act),
            P_eng_mech=float(P_eng_mech),
            P_mg1_mech=float(P_mg1_mech),
            P_mg2_mech=float(P_mg2_mech),
            P_mg1_elec=float(P_mg1_elec),
            P_mg2_elec=float(P_mg2_elec),
            P_aux=float(x.P_aux_W),
            P_batt_req=float(P_batt_req),
            P_batt_act=float(P_batt_act),
            P_brake_fric=float(P_brake_fric),
            shortfall_tq=float(shortfall_tq),
            shortfall_power=float(shortfall_power),
            E_batt_next=float(su.E_batt_Wh),
            soc_next=float(soc_next),
            J_total=float(J_total),
            J_fuel=float(J_fuel), J_soc=float(J_soc), J_fric=float(J_fric),
            J_short=float(J_short), J_spin=float(J_spin), J_smooth=float(J_smooth),
            J_charge=float(J_charge),J_over=float(J_over),excess_tq=float(excess_tq),
            resid_psd_speed=float(resid_psd_speed),
            resid_elec=float(resid_elec),
            resid_ring=float(resid_ring),
        )

    # -----------------------------
    # helpers
    # -----------------------------
    def traction_rpm_bounds(mode: str) -> tuple[float, float]:
        lo = beta * ring_rpm - alpha * x.mg1_rpm_max
        hi = beta * ring_rpm + alpha * x.mg1_rpm_max
        if mode in ["HybridDrive", "Charge"]:
            lo = max(lo, x.eng_rpm_min)
        else:
            lo = max(lo, 0.0)
        hi = min(hi, x.eng_rpm_max)
        return float(lo), float(hi)

    # local search config (most effective)
    LOCAL_RPM_WINDOW = 200.0
    LOCAL_TQ_WINDOW = 20.0
    LOCAL_RPM_STEP = eng_rpm_step
    LOCAL_TQ_STEP = eng_tq_step

    # =============================
    # mode enumeration
    # =============================
    for mode in modes:
        # ---- braking: 1-point eval
        if mode in ["Regen", "FrictionBrake"]:
            eng_rpm = choose_fuelcut_eng_rpm(ring_rpm, alpha, beta, x.mg1_rpm_max, x.eng_rpm_max)
            cand = eval_candidate(mode, eng_rpm, 0.0)
            if cand is not None and (best is None or cand["J_total"] < best["J_total"]):
                best = cand
            continue

        # ---- EV (traction): 1-point eval (fast)
        if mode == "EV":
            eng_rpm = choose_fuelcut_eng_rpm(ring_rpm, alpha, beta, x.mg1_rpm_max, x.eng_rpm_max)
            cand = eval_candidate(mode, eng_rpm, 0.0)
            if cand is not None and (best is None or cand["J_total"] < best["J_total"]):
                best = cand
            continue

        # ---- Hybrid/Charge: local search first, fallback to global
        lo, hi = traction_rpm_bounds(mode)
        if hi < lo:
            continue

        local_found = 0

        if x.prev_eng_rpm is not None and x.prev_eng_tq_Nm is not None:
            rpm0 = float(np.clip(x.prev_eng_rpm, lo, hi))
            rpm_lo = max(lo, rpm0 - LOCAL_RPM_WINDOW)
            rpm_hi = min(hi, rpm0 + LOCAL_RPM_WINDOW)
            rpm_grid = np.arange(rpm_lo, rpm_hi + 1e-9, LOCAL_RPM_STEP)

            for eng_rpm in rpm_grid:
                Tmax = float(eng_tq_max_map(float(eng_rpm)))
                tq0 = float(np.clip(x.prev_eng_tq_Nm, 0.0, Tmax))
                tq_lo = max(0.0, tq0 - LOCAL_TQ_WINDOW)
                tq_hi = min(Tmax, tq0 + LOCAL_TQ_WINDOW)
                tq_grid = np.arange(tq_lo, tq_hi + 1e-9, LOCAL_TQ_STEP)

                for eng_tq in tq_grid:
                    cand = eval_candidate(mode, float(eng_rpm), float(eng_tq))
                    if cand is None:
                        continue
                    local_found += 1
                    if best is None or cand["J_total"] < best["J_total"]:
                        best = cand

        # fallback to global only if local found nothing
        if local_found == 0:
            rpm_grid = np.arange(lo, hi + 1e-9, eng_rpm_step)
            for eng_rpm in rpm_grid:
                Tmax = float(eng_tq_max_map(float(eng_rpm)))
                tq_grid = np.arange(0.0, Tmax + 1e-9, eng_tq_step)
                for eng_tq in tq_grid:
                    cand = eval_candidate(mode, float(eng_rpm), float(eng_tq))
                    if cand is None:
                        continue
                    if best is None or cand["J_total"] < best["J_total"]:
                        best = cand

    # fallback (rare)
    if best is None:
        eng_rpm = choose_fuelcut_eng_rpm(ring_rpm, alpha, beta, x.mg1_rpm_max, x.eng_rpm_max)
        best = eval_candidate("EV" if x.P_wheel_req_W >= 0 else "FrictionBrake", eng_rpm, 0.0)

    if best is None:
        best = dict(
            mode="EV",
            fuel_cut=1,
            eng_rpm=float(eng_rpm), eng_tq=0.0,
            mg1_rpm=0.0, mg1_tq=0.0,
            mg2_rpm=float(ring_rpm), mg2_tq=0.0,
            P_eng_mech=0.0, P_mg1_mech=0.0, P_mg2_mech=0.0,
            P_mg1_elec=0.0, P_mg2_elec=0.0,
            P_aux=float(x.P_aux_W),
            P_batt_req=float(x.P_aux_W),
            P_batt_act=float(np.clip(x.P_aux_W, -x.P_chg_max_W, x.P_dis_max_W)),
            P_brake_fric=float(max(0.0, -x.P_wheel_req_W)),
            shortfall_tq=float(max(0.0, x.T_ring_req_Nm)),
            shortfall_power=float(max(0.0, x.T_ring_req_Nm * max(omega_ring, 0.0))),
            excess_tq=0.0,
            J_over=0.0,
            E_batt_next=float(x.E_batt_Wh),
            soc_next=float(x.soc),
            J_total=1e18, J_fuel=0.0, J_soc=0.0, J_fric=0.0, J_short=0.0, J_spin=0.0, J_smooth=0.0, J_charge=0.0,
            resid_psd_speed=0.0, resid_elec=0.0, resid_ring=0.0,
        )

    stats = FeasibilityStats(
        n_total=n_total,
        n_kept=n_kept,
        n_fail_mg1_speed=n_fail_mg1_speed,
        n_fail_mg2_speed=n_fail_mg2_speed,
        n_fail_mg1_tq=n_fail_mg1_tq,
        n_fail_eng_tq=n_fail_eng_tq,
    )

    return StepResult(
        mode=str(best["mode"]),
        fuel_cut=int(best["fuel_cut"]),
        eng_rpm=float(best["eng_rpm"]),
        eng_tq_Nm=float(best["eng_tq"]),
        mg1_rpm=float(best["mg1_rpm"]),
        mg1_tq_Nm=float(best["mg1_tq"]),
        mg2_rpm=float(best["mg2_rpm"]),
        mg2_tq_Nm=float(best["mg2_tq"]),
        P_eng_mech_W=float(best["P_eng_mech"]),
        P_mg1_mech_W=float(best["P_mg1_mech"]),
        P_mg2_mech_W=float(best["P_mg2_mech"]),
        P_mg1_elec_W=float(best["P_mg1_elec"]),
        P_mg2_elec_W=float(best["P_mg2_elec"]),
        P_aux_W=float(best["P_aux"]),
        P_batt_req_W=float(best["P_batt_req"]),
        P_batt_act_W=float(best["P_batt_act"]),
        P_brake_fric_W=float(best["P_brake_fric"]),
        shortfall_tq_Nm=float(best["shortfall_tq"]),
        shortfall_power_W=float(best["shortfall_power"]),
        E_batt_next_Wh=float(best["E_batt_next"]),
        soc_next=float(best["soc_next"]),
        J_total=float(best["J_total"]),
        J_fuel=float(best["J_fuel"]),
        J_soc=float(best["J_soc"]),
        J_fric=float(best["J_fric"]),
        J_short=float(best["J_short"]),
        excess_tq_Nm=float(best["excess_tq"]),
        J_over=float(best["J_over"]),
        J_spin=float(best["J_spin"]),
        J_smooth=float(best["J_smooth"]),
        J_charge=float(best["J_charge"]),
        resid_psd_speed_rpm=float(best["resid_psd_speed"]),
        resid_elec_power_W=float(best["resid_elec"]),
        resid_ring_torque_Nm=float(best["resid_ring"]),
        resid_soc_recon_pct=0.0,
        stats=stats,
    )
