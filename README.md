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

## Ignored output directories

The following directories are output/archive locations and should be treated as ignored (not committed source):

- `artifacts/`
- `_local_archive/`
