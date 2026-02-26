from __future__ import annotations

"""Phase B one-step solver (stub).

FROZEN GOALS (Phase B):
- Add fuel model (BSFC) and explicit loss accounting.
- Preserve Phase A contracts and dashboard compatibility.
- Return a composed StepResultB: base (Phase A-like) + batt/fuel/loss/split.

IMPORT FIREWALL (IMMUTABLE):
- MUST NOT import Phase A solver modules at runtime (no `import step_A`).
- May import Phase A *configs/environment/wltc/run_io* modules read-only.

NOTE:
- Type hints may refer to Phase A StepResult under TYPE_CHECKING only.
"""

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from .battery_B import BatteryConfigB, BattChemResult, battery_B0, battery_B1
from .fuel_B import FuelAccount, BsfcMap, calc_fuel_account
from .losses_B import LossAccount, SimConfigB, mg_loss_from_mech_and_bus, inverter_losses
from .accounting_B import split_charging_origin


# --- Map interfaces (copied, do not import step_A)
EtaMap = Callable[[float, float], float]     # (speed_rpm, torque_Nm) -> eta in (0,1]
EngTqMaxMap = Callable[[float], float]       # eng_rpm -> Tmax Nm (>=0)
EngDragMinMap = Callable[[float], float]     # eng_rpm -> Tmin Nm (<=0) during fuel-cut (optional)


if TYPE_CHECKING:
    # for typing only (does not execute at runtime)
    from .step_A import StepResult  # noqa: F401


@dataclass(frozen=True)
class ChargeSplit:
    """Battery charging source attribution (both >=0)."""
    P_batt_chg_from_regen_W: float
    P_batt_chg_from_engine_W: float


@dataclass(frozen=True)
class StepResultB:
    """Phase B outputs - wraps Phase A-like base + energy accounting."""

    base: "StepResult"  # Phase A result object (frozen type)
    batt: BattChemResult
    fuel: FuelAccount
    losses: LossAccount
    split: ChargeSplit


@dataclass(frozen=True)
class StepInputsB:
    """Phase B step inputs.

    Mirror of Phase A StepInputs, duplicated to avoid importing step_A.
    """

    # timebase
    t_s: float
    dt_s: float
    phase: str

    # ring / demand
    ring_rpm: float
    ring_omega_radps: float
    T_ring_req_Nm: float
    P_wheel_req_W: float

    # state
    soc: float
    E_batt_Wh: float

    # battery constants
    E_usable_Wh: float
    Emin_Wh: float
    Emax_Wh: float

    # environment / aux
    P_aux_W: float
    Tamb_C: float
    Tbatt_C: float

    # battery limits
    P_dis_max_W: float
    P_chg_max_W: float

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


def enrich_with_phase_b_extensions(
    *,
    base: "StepResult",
    bsfc_map: BsfcMap,
    batt_cfg: BatteryConfigB,
    sim_cfg: SimConfigB,
    lhv_J_per_g: float | None = None,
) -> StepResultB:
    """Enrich a Phase A step solution with Phase B fuel/loss/split accounting.

    This function is deliberately **pure** and respects the import firewall:
      - It does NOT import or call the Phase A solver.
      - It only consumes an already-solved Phase A `StepResult`-like object.

    Frozen sign conventions used:
      - base.P_batt_act_W > 0 : discharge (battery -> bus)
      - base.P_batt_act_W < 0 : charge    (bus -> battery)
      - base.P_mg*_elec_W < 0 : generating to the bus
      - All loss terms are non-negative.

    Args:
        base:
            Phase A step result (frozen type). Must expose at least:
              - eng_rpm, eng_tq_Nm
              - P_batt_act_W
              - P_mg1_mech_W, P_mg1_elec_W
              - P_mg2_mech_W, P_mg2_elec_W
        bsfc_map:
            BSFC map callable: (eng_rpm, eng_tq_Nm) -> g/kWh.
        batt_cfg:
            Battery model config (B0 implemented, B1 stub).
        sim_cfg:
            Sim flags (inverter loss mode, etc.).
        lhv_J_per_g:
            Optional LHV override [J/g]. If None, uses fuel_B default.

    Returns:
        StepResultB composed of:
          - base (Phase A)
          - batt (BattChemResult)
          - fuel (FuelAccount)
          - losses (LossAccount)
          - split (ChargeSplit)
    """

    # --- Fuel
    if lhv_J_per_g is None:
        fuel = calc_fuel_account(
            eng_rpm=float(base.eng_rpm),
            eng_tq_Nm=float(base.eng_tq_Nm),
            bsfc_map=bsfc_map,
        )
    else:
        fuel = calc_fuel_account(
            eng_rpm=float(base.eng_rpm),
            eng_tq_Nm=float(base.eng_tq_Nm),
            bsfc_map=bsfc_map,
            lhv_J_per_g=float(lhv_J_per_g),
        )

    # --- Battery (terminal power is Phase A P_batt_act_W)
    P_term_W = float(base.P_batt_act_W)
    if batt_cfg.model == "B0":
        batt: BattChemResult = battery_B0(P_term_W=P_term_W, cfg=batt_cfg)
    else:
        # B1 requires SOC; use Phase A soc_next if present, else raise.
        soc = getattr(base, "soc_next", None)
        if soc is None:
            raise ValueError("B1 battery model requires base.soc_next (or equivalent SOC field)")
        batt = battery_B1(P_term_W=P_term_W, soc=float(soc), cfg=batt_cfg)

    # --- MG & inverter losses (bus power is Phase A P_mg*_elec_W)
    loss_mg1_W = mg_loss_from_mech_and_bus(P_mech_W=float(base.P_mg1_mech_W), P_bus_elec_W=float(base.P_mg1_elec_W))
    loss_mg2_W = mg_loss_from_mech_and_bus(P_mech_W=float(base.P_mg2_mech_W), P_bus_elec_W=float(base.P_mg2_elec_W))

    loss_inv1_W, loss_inv2_W = inverter_losses(
        cfg=sim_cfg,
        P_mg1_bus_W=float(base.P_mg1_elec_W),
        P_mg2_bus_W=float(base.P_mg2_elec_W),
    )

    losses = LossAccount(
        loss_mg1_W=loss_mg1_W,
        loss_mg2_W=loss_mg2_W,
        loss_inv1_W=float(loss_inv1_W),
        loss_inv2_W=float(loss_inv2_W),
        loss_engine_W=float(fuel.loss_engine_W),
    )

    # --- Charging origin split
    split = postprocess_split(
        P_batt_term_W=P_term_W,
        P_mg1_bus_W=float(base.P_mg1_elec_W),
        P_mg2_bus_W=float(base.P_mg2_elec_W),
        mode=getattr(base, 'mode', None),
    )

    return StepResultB(base=base, batt=batt, fuel=fuel, losses=losses, split=split)


def solve_step_B(
    x: StepInputsB,
    *,
    # maps
    bsfc_map: BsfcMap,
    eta_mg1_map: EtaMap,
    eta_mg2_map: EtaMap,
    eng_tq_max_map: EngTqMaxMap,
    eng_drag_min_map: EngDragMinMap | None,
    # phase B configs
    batt_cfg: BatteryConfigB,
    sim_cfg: SimConfigB,
    # grid knobs (same style as A)
    eng_rpm_step: float = 100.0,
    eng_tq_step: float = 5.0,
    soc_target: float = 0.55,
    soc_band: float = 0.05,
) -> StepResultB:
    """Solve one step with Phase B accounting (stub).

    Required behavior (frozen):
    - Produce a physically-consistent base solution (Phase A-like).
    - Compute:
        - batt: P_batt_chem_W, loss_batt_W, I_batt_A
        - fuel: mdot_fuel_gps, P_fuel_W, loss_engine_W
        - losses: loss_mg*, loss_inv*, loss_engine
        - split: P_batt_chg_from_regen_W / P_batt_chg_from_engine_W

    Returns:
        StepResultB (composed result).

    Raises:
        NotImplementedError: Until Phase B.1 solver implementation.
    """
    raise NotImplementedError("Phase B.1 - step solver pending")


def postprocess_split(
    *,
    P_batt_term_W: float,
    P_mg1_bus_W: float,
    P_mg2_bus_W: float,
    mode: str | None = None,
    P_wheel_deliv_W_dbg: float | None = None,
) -> ChargeSplit:
    """Compute charging origin split.

    Notes:
      - This function remains firewall-compliant and pure.
      - If braking context is provided (mode or wheel power), regen attribution is only
        allowed during braking to avoid regen_utilization > 1.
    """
    P_regen, P_eng = split_charging_origin(
        P_batt_term_W=P_batt_term_W,
        P_mg1_bus_W=P_mg1_bus_W,
        P_mg2_bus_W=P_mg2_bus_W,
        mode=mode,
        P_wheel_deliv_W_dbg=P_wheel_deliv_W_dbg,
    )
    return ChargeSplit(P_batt_chg_from_regen_W=P_regen, P_batt_chg_from_engine_W=P_eng)
