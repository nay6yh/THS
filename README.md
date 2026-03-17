# THS Phase B Runner Quickstart

This repository contains scripts to execute a Phase B simulation run and save results in a per-run folder under `artifacts/out_phaseB/`.

## 1) Create and activate a Conda environment

```bash
conda create -n ths python=3.11 -y
conda activate ths
```

## 2) Install the project in editable mode

From the repository root:

```bash
pip install -e .
```

If your environment has import-path drift, run scripts with explicit `PYTHONPATH=src` for reproducibility:

```bash
PYTHONPATH=src python scripts/run_phaseB_suite.py --wltc_csv data/cycles/WLTC3b.csv --out_dir artifacts/out_phaseB_suite
```

## 3) Generate a Phase B run folder

Run the Phase B script:

```bash
python scripts/run_phaseB.py --out_root artifacts/out_phaseB --copy_inputs
```

This creates a new timestamped run folder under `artifacts/out_phaseB/`.

## 4) Run visualization into that run folder

You have two options:

- Run simulation + visualization together by adding `--run_viz` to `run_phaseB.py`, or
- Run visualization separately for an existing run folder.

Separate visualization example:

```bash
python scripts/viz_ths_dashboards_phaseB.py \
  --timeseries artifacts/out_phaseB/<run_id>/signals/timeseries_phaseB_<stamp>.csv \
  --outdir artifacts/out_phaseB/<run_id>/Viz \
  --dpi 160
```

## Expected run-folder structure

Each run folder under `artifacts/out_phaseB/<run_id>/` is expected to contain:

```text
artifacts/out_phaseB/<run_id>/
├── inputs/
├── configs/
├── signals/
├── audit/
├── Viz/
└── manifest.json
```

## One command example

If you want simulation and visualization in one shot:

```bash
python scripts/run_phaseB.py --out_root artifacts/out_phaseB --copy_inputs --run_viz --viz_dpi 160
```

## Phase B.2 acceptance-oriented comparison flow

1. Run baseline + supervisor variants and determinism check:

```bash
PYTHONPATH=src python scripts/run_phaseB_suite.py --wltc_csv data/cycles/WLTC3b.csv --out_dir artifacts/out_phaseB_suite
```

2. Print gate and baseline comparison summaries:

```bash
PYTHONPATH=src python scripts/analyze_phaseB_suite.py --root artifacts/out_phaseB_suite
```

The suite now auto-generates baseline comparison JSONs for KPI deltas and gate debug counters:

- `B00b_compare_vs_baseline.json` (start penalty effectiveness / residual preservation)
- `B00c_compare_vs_baseline.json` (min_on/min_off effectiveness / residual preservation)

These include:

- KPI deltas vs baseline (`count_eng_start`, `fuel_g_per_km`, residual metrics)
- gate debug counters (`min_off_hits`, `overridden`, `by_power`, `by_soc`, `free_relight_req`, `best_relight`)

Suite summary now also includes structured triage fields per variant in `suite_summary.csv` (and mirrored in `suite_summary.json`):

- `variant`
- `PASS`
- `gating_mode` (`blocking`, `informational`, `expected_fail_probe`, or `blocking_expected_fail`)
- `primary_failure_reason` (`pass`, `traction_shortfall`, `residual_limit_exceeded`, `audit_A_failed`, `audit_B_failed`, `determinism_failed`, `expected_fail_not_detected`, etc.)
- `note` (short per-row detail)
- `blocking` (whether that row contributes to `OVERALL_PASS`)

`F01_expected_fail_low_mg2_tq_max` and `F01_expected_fail_check` are now easier to read together via `expected_fail_check.json`, which records both the expected-fail verdict and the trigger gate key.

## Ignored output directories

The following directories are output/archive locations and should be treated as ignored (not committed source):

- `artifacts/`
- `_local_archive/`
