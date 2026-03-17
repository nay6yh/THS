from __future__ import annotations

"""THS colinear-diagram-based kinematic + powerflow simulator.

This module keeps kinematic constraints (speed relations) separate from torque strategy.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ModeRequest(str, Enum):
    """Supported high-level operating requests for this simplified simulator.

    This enum is intentionally scoped to the currently implemented strategy set
    and is not an exhaustive taxonomy of all real-world/transitional THS modes.
    """

    EV = "EV"
    HV = "HV"
    CHARGE = "CHARGE"
    REGEN = "REGEN"
    ENGINE_ONLY = "ENGINE_ONLY"
    ENGINE_GEN = "ENGINE_GEN"


@dataclass(frozen=True)
class ColinearParams:
    """Planetary colinear diagram parameters.

    rho: front planetary x-distance ratio (>0)
    grm: rear planetary x-distance ratio (>0)
    """

    rho: float
    grm: float

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ValueError("rho must be > 0")
        if self.grm <= 0:
            raise ValueError("grm must be > 0")


@dataclass(frozen=True)
class SimulatorConfig:
    """Static model constants."""

    colinear: ColinearParams
    prop_speed_gain_rad_per_mps: float
    idle_engine_speed_radps: float = 120.0


@dataclass(frozen=True)
class SimulationInput:
    vehicle_speed_mps: float
    wheel_torque_request_Nm: float
    mode_request: ModeRequest
    engine_on_flag: bool
    battery_soc: float
    battery_power_limit_W: float
    engine_speed_target_radps: float | None = None
    loss_front_W: float = 0.0
    loss_rear_W: float = 0.0


@dataclass(frozen=True)
class SpeedState:
    Ng: float
    Ne: float
    Np: float
    Nm: float


@dataclass(frozen=True)
class TorqueState:
    Tg: float
    Te: float
    Tp: float
    Tm: float


@dataclass(frozen=True)
class PowerState:
    Pg: float
    Pe: float
    Pp: float
    Pm: float
    battery_power_W: float


@dataclass(frozen=True)
class BalanceState:
    front_balance_W: float
    rear_balance_W: float


@dataclass(frozen=True)
class SimulationResult:
    mode_state: ModeRequest
    speeds: SpeedState
    torques: TorqueState
    powers: PowerState
    balance: BalanceState
    energy_flows: List[str] = field(default_factory=list)
    sign_checks: List[str] = field(default_factory=list)


class THSColinearSimulator:
    """Minimal THS model with explicit speed constraints and separate torque allocation."""

    def __init__(self, cfg: SimulatorConfig) -> None:
        self.cfg = cfg

    def compute_speeds(self, x: SimulationInput) -> SpeedState:
        """Step 1-4: compute speeds only from constraints + mode policy."""

        Np = x.vehicle_speed_mps * self.cfg.prop_speed_gain_rad_per_mps
        Nm = -self.cfg.colinear.grm * Np

        if not x.engine_on_flag or x.mode_request in (ModeRequest.EV, ModeRequest.REGEN):
            Ne = 0.0
        else:
            Ne = (
                self.cfg.idle_engine_speed_radps
                if x.engine_speed_target_radps is None
                else x.engine_speed_target_radps
            )

        rho = self.cfg.colinear.rho
        Ng = ((1.0 + rho) * Ne - Np) / rho

        return SpeedState(Ng=Ng, Ne=Ne, Np=Np, Nm=Nm)

    def simulate(self, x: SimulationInput) -> SimulationResult:
        speeds = self.compute_speeds(x)
        torques = self._initial_torques(x, speeds)
        torques = self._enforce_power_balance(torques, speeds, x.loss_front_W, x.loss_rear_W)
        powers = self._compute_powers(torques, speeds)
        checks = self._sign_checks(x.mode_request, speeds, torques)
        flows = self._energy_flows(powers)
        balance = BalanceState(
            front_balance_W=torques.Tg * speeds.Ng + torques.Te * speeds.Ne + torques.Tp * speeds.Np + x.loss_front_W,
            rear_balance_W=torques.Tp * speeds.Np + torques.Tm * speeds.Nm + x.loss_rear_W,
        )

        return SimulationResult(
            mode_state=x.mode_request,
            speeds=speeds,
            torques=torques,
            powers=powers,
            balance=balance,
            energy_flows=flows,
            sign_checks=checks,
        )

    def _initial_torques(self, x: SimulationInput, s: SpeedState) -> TorqueState:
        """Step 5: representative torque strategy by mode.

        This is intentionally not unique; it is a tunable policy layer.
        """

        Tp = x.wheel_torque_request_Nm

        if x.mode_request == ModeRequest.EV:
            return TorqueState(Tg=5.0, Te=0.0, Tp=Tp, Tm=-abs(Tp))
        if x.mode_request == ModeRequest.REGEN:
            return TorqueState(Tg=0.0, Te=0.0, Tp=Tp, Tm=abs(Tp))
        if x.mode_request == ModeRequest.CHARGE:
            return TorqueState(Tg=-20.0, Te=80.0, Tp=Tp, Tm=0.0)
        if x.mode_request == ModeRequest.ENGINE_GEN:
            return TorqueState(Tg=-30.0, Te=max(60.0, Tp), Tp=Tp, Tm=0.0)
        if x.mode_request == ModeRequest.ENGINE_ONLY:
            return TorqueState(Tg=0.0, Te=max(40.0, Tp), Tp=Tp, Tm=0.0)
        # HV general
        return TorqueState(Tg=-10.0, Te=max(20.0, 0.6 * Tp), Tp=Tp, Tm=-0.4 * Tp)

    def _enforce_power_balance(self, t: TorqueState, s: SpeedState, loss_front_W: float, loss_rear_W: float) -> TorqueState:
        """Step 6: enforce per-planetary power equations (ideal/loss-included)."""

        # Rear: Tp*Np + Tm*Nm + loss_rear = 0
        Tm = t.Tm
        if abs(s.Nm) > 1e-9:
            Tm = -(t.Tp * s.Np + loss_rear_W) / s.Nm

        # Front: Tg*Ng + Te*Ne + Tp*Np + loss_front = 0
        Tg = t.Tg
        if abs(s.Ng) > 1e-9:
            Tg = -(t.Te * s.Ne + t.Tp * s.Np + loss_front_W) / s.Ng

        return TorqueState(Tg=Tg, Te=t.Te, Tp=t.Tp, Tm=Tm)

    @staticmethod
    def _compute_powers(t: TorqueState, s: SpeedState) -> PowerState:
        Pg = t.Tg * s.Ng
        Pe = t.Te * s.Ne
        Pp = t.Tp * s.Np
        Pm = t.Tm * s.Nm
        # ideal DC bus sign: positive = battery charge, negative = battery discharge
        battery_power_W = max(Pg, 0.0) + max(Pm, 0.0) - max(-Pg, 0.0) - max(-Pm, 0.0)
        return PowerState(Pg=Pg, Pe=Pe, Pp=Pp, Pm=Pm, battery_power_W=battery_power_W)

    @staticmethod
    def _sign_checks(mode: ModeRequest, s: SpeedState, t: TorqueState) -> List[str]:
        checks: List[str] = []
        if s.Np > 0 and s.Nm >= 0:
            checks.append("Expected Nm < 0 while Np > 0 for rear fixed-carrier relation")
        if mode in (ModeRequest.EV, ModeRequest.REGEN) and abs(s.Ne) > 1e-9:
            checks.append("Engine-off mode requires Ne = 0")
        if mode == ModeRequest.EV and s.Np > 0 and s.Ng >= 0:
            checks.append("EV forward expects Ng < 0 when Ne = 0")
        if mode == ModeRequest.CHARGE and abs(s.Np) < 1e-9 and s.Ng <= 0:
            checks.append("Standstill charging with Ne > 0 expects Ng > 0")
        if mode == ModeRequest.REGEN and t.Tm >= 0:
            checks.append("Regen expects Tm < 0 while Nm < 0")
        return checks

    @staticmethod
    def _energy_flows(p: PowerState) -> List[str]:
        flows: List[str] = []
        if p.Pe > 0 and p.Pp > 0:
            flows.append("engine -> wheel")
        if p.Pe > 0 and p.Pg < 0:
            flows.append("engine -> MG1(generator)")
        if p.Pm > 0:
            flows.append("wheel -> MG2 -> battery")
        if p.Pm < 0:
            flows.append("battery -> MG2 -> wheel")
        return flows
