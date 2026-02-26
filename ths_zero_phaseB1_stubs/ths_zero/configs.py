from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class CommonConfig:
    # physics constants
    g: float = 9.80665
    R_air: float = 287.05  # J/kg/K (dry air)

    # control / energy management (simple baseline)
    soc_target: float = 0.55
    soc_band: float = 0.05
    eng_on_wheel_power_W: float = 8000.0
    eng_on_min_speed_mps: float = 2.0
    soc_charge_gain_W_per_soc: float = 25000.0  # W per SOC error (0..1)

    # solver tweaks
    traction_refine_iters: int = 2  # small refinement loop for traction


@dataclass
class VehicleConfig:
    name: str = "Yaris_HEV_like"
    mass_kg: float = 1250.0
    Crr: float = 0.010
    CdA: float = 0.62
    tire_radius_m: float = 0.31
    final_drive: float = 3.7
    driveline_eff: float = 0.97

    # PSD teeth
    Zs: int = 30
    Zr: int = 78

    # limits
    eng_rpm_min: float = 1200.0
    eng_rpm_max: float = 5200.0
    mg1_rpm_max: float = 10000.0
    mg2_rpm_max: float = 12000.0

    eng_tq_max_Nm: float = 120.0
    mg1_tq_max_Nm: float = 80.0
    mg2_tq_max_Nm: float = 220.0

    # simple efficiencies (placeholders; replace with maps later)
    eta_mg1: float = 0.92
    eta_mg2: float = 0.92


@dataclass
class BatteryConfig:
    E_usable_Wh: float = 800.0
    soc_min: float = 0.30
    soc_max: float = 0.80

    # base power limits at nominal temp (used as upper envelope)
    P_discharge_nom_W: float = 30000.0  # + (discharge)
    P_charge_nom_W: float = 25000.0     # magnitude for - (charge)

    eta_discharge: float = 0.96
    eta_charge: float = 0.96

    P_aux_base_W: float = 300.0  # base auxiliaries not HVAC


@dataclass
class InitialState:
    soc0: float = 0.55  # 0..1


@dataclass
class EnvironmentConfig:
    # ambient conditions
    Tamb_C: float = 20.0
    p_amb_Pa: float = 101325.0

    # HVAC simple model
    cabin_setpoint_C: float = 23.0
    hvac_k_cool_W_per_C: float = 250.0   # extra W per °C above setpoint
    hvac_k_heat_W_per_C: float = 350.0   # extra W per °C below setpoint
    hvac_Pmax_W: float = 2500.0          # cap for HVAC power (simple)

    # battery temp model (Level 1): Tbatt = Tamb
    battery_temp_model: str = "ambient"  # "ambient" or "fixed"
    Tbatt_fixed_C: float = 25.0


def load_or_default(path: str | None, default_obj):
    if path is None:
        return default_obj
    d = _read_json(path)
    cls = type(default_obj)
    return cls(**d)


def dump_config(path: str, cfg_obj) -> None:
    _write_json(path, asdict(cfg_obj))