# AGENTS.md

## Purpose
This repository contains THS simulation, Phase B suite evaluation, and related analysis utilities.

When working in this repo:
- prefer small, reviewable changes,
- preserve current acceptance semantics unless the task explicitly asks to change them,
- keep `main` green,
- and avoid broad behavioral changes unless clearly justified.

## Environment and execution
### Python / imports
Some scripts require `src` on the import path.

On PowerShell, set:
```powershell
$env:PYTHONPATH="src"
```

Then run commands normally, for example:
```powershell
python -m pytest -q
python scripts/run_phaseB_suite.py --wltc_csv data/cycles/WLTC3b.csv --out_dir artifacts/out_phaseB_suite
```

Do not assume Linux-style inline env assignment works on PowerShell.

## Key commands
### Test suite
```powershell
python -m pytest -q
```

### Phase B suite
```powershell
python scripts/run_phaseB_suite.py --wltc_csv data/cycles/WLTC3b.csv --out_dir artifacts/out_phaseB_suite
```

### Analyze Phase B suite
```powershell
python scripts/analyze_phaseB_suite.py --root artifacts/out_phaseB_suite
```

## Repository expectations
### Keep main green
Do not introduce new blocking failures on `main`.

If the task adds comparison or evaluation outputs that are useful but not yet acceptance-ready:
- keep them informational / non-gating unless explicitly requested otherwise,
- label them clearly,
- and do not silently convert informational checks into blocking gates.

### Do not weaken checks silently
Do not make acceptance criteria looser just to get green results.
Do not change thresholds, audit rules, or gating behavior unless the task explicitly asks for that and the PR summary explains it clearly.

### Preserve semantics unless requested
Do not change physical model behavior, sign conventions, or suite scope unless that is the purpose of the task.
If a change affects semantics, explain it clearly in the PR summary and update tests accordingly.

## Phase B suite guidance
### Failure triage
When modifying the Phase B suite, prefer adding structured diagnosis rather than only console prints.

Useful output fields include:
- `variant`
- `PASS`
- `gating`
- `failure_reason`
- `note`

Prefer machine-readable artifacts such as CSV or JSON for downstream review.

### Expected-fail cases
For expected-fail variants, make it obvious whether:
- the failure happened,
- it happened for the intended reason,
- and whether the follow-up interpretation passed.

Do not make expected-fail rows confusing to interpret.

### Informational supervisor comparisons
B00b/B00c-style supervisor comparison rows should remain informational / non-gating unless the task explicitly asks to make them blocking.

## Tests
Whenever behavior changes, update or add tests.
Prefer focused regression tests for:
- bug fixes,
- sign convention handling,
- acceptance summary logic,
- expected-fail interpretation,
- and CLI/config persistence.

Do not lock inconsistent behavior into tests unless the task explicitly documents that inconsistency as intentional.

## PR guidance
### Before opening a PR
Try to run:
```powershell
python -m pytest -q
```

And, when relevant:
```powershell
python scripts/run_phaseB_suite.py --wltc_csv data/cycles/WLTC3b.csv --out_dir artifacts/out_phaseB_suite_check
```

### PR summary should include
1. What changed
2. Why it changed
3. Whether semantics changed
4. What tests were run
5. Whether the set of failing variants differs from current `main`

### Keep PRs scoped
Prefer one purpose per PR.
Avoid mixing:
- evaluation framework changes,
- supervisor tuning,
- physical model behavior changes,
- and unrelated README cleanup

in one PR unless explicitly requested.

## Documentation
If a task adds a new artifact, output field, or workflow step, update README or the relevant docs briefly.
Keep documentation practical and consistent with actual commands.

## Style
- Prefer explicit naming over clever shortcuts.
- Prefer stable output field names for downstream analysis.
- Reuse existing artifacts and audit outputs where possible.
- Minimize noisy debug prints unless they are task-relevant.

