# AGENTS.md

## Core rules
- Keep `main` green.
- Do not silently weaken acceptance checks.
- Informational rows must not become blocking unless explicitly requested.
- Preserve semantics unless the task explicitly asks to change them.
- Prefer small, reviewable PRs.

## Environment
On PowerShell:
```powershell
$env:PYTHONPATH="src"
```

Common commands:
```powershell
python -m pytest -q
python scripts/run_phaseB_suite.py --wltc_csv data/cycles/WLTC3b.csv --out_dir artifacts/out_phaseB_suite
python scripts/analyze_phaseB_suite.py --root artifacts/out_phaseB_suite
```

## Phase B suite
- Prefer structured failure diagnostics over console-only output.
- Keep B00b/B00c-style supervisor comparison rows informational / non-gating unless explicitly requested otherwise.
- Expected-fail rows must clearly show whether failure happened and whether it happened for the intended reason.

## PR expectations
Include in the PR summary:
1. What changed
2. Why it changed
3. Whether semantics changed
4. What tests were run
5. Whether the failing-variant set differs from current `main`

## Tests
- Add or update focused regression tests for behavior changes.
- Do not lock inconsistent behavior into tests unless explicitly documented as intentional.

## Notes
- Keep output field names stable for downstream analysis.
- Reuse existing artifacts and audit outputs where possible.
- If needed, place a more specific `AGENTS.md` in a subdirectory; Codex will prefer the closest one to the changed files.

