from __future__ import annotations

import json
import hashlib
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any
import shutil
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_df_parquet_or_csv(df: pd.DataFrame, path_no_ext: Path) -> Path:
    """
    Try parquet; fallback to csv if pyarrow/fastparquet not available.
    Returns actual file path written.
    """
    ensure_dir(path_no_ext.parent)
    try:
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = path_no_ext.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def copy_input(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def build_manifest(run_id: str, inputs: dict[str, Any], configs: dict[str, Any], outputs: dict[str, str]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
        "sign_conventions": {
            "P_batt_W": "+ discharge, - charge",
            "P_wheel_req_W": "+ traction, - braking",
            "P_brake_fric_W": ">=0 dissipated heat",
        },
        "inputs": inputs,
        "configs": configs,
        "outputs": outputs,
        "schema_version": "0.1.0",
    }


def default_schema() -> dict[str, Any]:
    """
    Minimal schema doc. Extend freely.
    """
    return {
        "notes": [
            "This schema describes key columns only. Add more fields as needed.",
            "P_batt_act_W is used for SOC integration; P_batt_req_W shows infeasible demand when battery is saturated.",
        ],
        "columns": {
            "Tamb_C": {"unit": "degC", "domain": "environment"},
            "rho_air_kgpm3": {"unit": "kg/m3", "domain": "environment"},
            "P_hvac_W": {"unit": "W", "domain": "aux", "sign": ">=0"},
            "P_aux_W": {"unit": "W", "domain": "aux", "sign": ">=0"},
            "P_batt_req_W": {"unit": "W", "domain": "battery", "sign": "+ discharge, - charge"},
            "P_batt_act_W": {"unit": "W", "domain": "battery", "sign": "+ discharge, - charge"},
            "soc_pct": {"unit": "%", "domain": "battery"},
            "resid_psd_speed_rpm": {"unit": "rpm", "domain": "constraint"},
            "resid_elec_power_W": {"unit": "W", "domain": "constraint"},
            "resid_wheel_power_W": {"unit": "W", "domain": "constraint"},
            "resid_soc_recon_pct": {"unit": "%", "domain": "constraint"},
        }
    }