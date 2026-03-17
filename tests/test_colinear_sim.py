import math
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root / 'src') not in sys.path:
    sys.path.insert(0, str(_root / 'src'))


from ths_zero.colinear_sim import (
    ColinearParams,
    ModeRequest,
    SimulationInput,
    SimulatorConfig,
    THSColinearSimulator,
)


def _sim() -> THSColinearSimulator:
    cfg = SimulatorConfig(
        colinear=ColinearParams(rho=0.5, grm=2.6),
        prop_speed_gain_rad_per_mps=20.0,
        idle_engine_speed_radps=110.0,
    )
    return THSColinearSimulator(cfg)


def test_ev_forward_speed_constraints_and_signs() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=10.0,
            wheel_torque_request_Nm=100.0,
            mode_request=ModeRequest.EV,
            engine_on_flag=False,
            battery_soc=0.6,
            battery_power_limit_W=40_000.0,
        )
    )
    assert out.speeds.Ne == 0.0
    assert out.speeds.Np > 0
    assert out.speeds.Nm < 0
    assert out.speeds.Ng < 0
    assert out.sign_checks == []


def test_rear_constraint_equation() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=8.0,
            wheel_torque_request_Nm=80.0,
            mode_request=ModeRequest.HV,
            engine_on_flag=True,
            battery_soc=0.6,
            battery_power_limit_W=40_000.0,
            engine_speed_target_radps=140.0,
        )
    )
    assert math.isclose(out.speeds.Nm, -2.6 * out.speeds.Np, rel_tol=1e-9)


def test_front_constraint_equation() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=5.0,
            wheel_torque_request_Nm=40.0,
            mode_request=ModeRequest.HV,
            engine_on_flag=True,
            battery_soc=0.5,
            battery_power_limit_W=30_000.0,
            engine_speed_target_radps=120.0,
        )
    )
    rho = 0.5
    expected = (1 + rho) * out.speeds.Ne - rho * out.speeds.Ng
    assert math.isclose(out.speeds.Np, expected, rel_tol=1e-9)


def test_standstill_charge_keeps_np_nm_zero_and_ng_positive() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=0.0,
            wheel_torque_request_Nm=0.0,
            mode_request=ModeRequest.CHARGE,
            engine_on_flag=True,
            battery_soc=0.4,
            battery_power_limit_W=20_000.0,
            engine_speed_target_radps=100.0,
        )
    )
    assert out.speeds.Np == 0.0
    assert out.speeds.Nm == 0.0
    assert out.speeds.Ne > 0
    assert out.speeds.Ng > 0


def test_power_balances_hold_with_losses() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=12.0,
            wheel_torque_request_Nm=120.0,
            mode_request=ModeRequest.ENGINE_GEN,
            engine_on_flag=True,
            battery_soc=0.6,
            battery_power_limit_W=50_000.0,
            engine_speed_target_radps=150.0,
            loss_front_W=500.0,
            loss_rear_W=300.0,
        )
    )
    assert abs(out.balance.front_balance_W) < 1e-6
    assert abs(out.balance.rear_balance_W) < 1e-6


def test_regen_sign_pattern() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=9.0,
            wheel_torque_request_Nm=-60.0,
            mode_request=ModeRequest.REGEN,
            engine_on_flag=False,
            battery_soc=0.7,
            battery_power_limit_W=25_000.0,
        )
    )
    assert out.speeds.Nm < 0
    assert out.torques.Tm < 0
    assert abs(out.balance.rear_balance_W) < 1e-6
    assert not any("Regen expects" in msg for msg in out.sign_checks)
    assert "wheel -> MG2 -> battery" in out.energy_flows
    assert out.powers.battery_power_W > 0


def test_engine_speed_target_zero_is_not_replaced_with_idle() -> None:
    sim = _sim()
    out = sim.simulate(
        SimulationInput(
            vehicle_speed_mps=6.0,
            wheel_torque_request_Nm=50.0,
            mode_request=ModeRequest.HV,
            engine_on_flag=True,
            battery_soc=0.5,
            battery_power_limit_W=30_000.0,
            engine_speed_target_radps=0.0,
        )
    )
    assert out.speeds.Ne == 0.0
