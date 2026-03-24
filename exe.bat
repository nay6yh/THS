@echo off
setlocal

set "PYTHONPATH=src"

python -m pytest -q tests/test_audit_B.py
if errorlevel 1 goto :error

python -m pytest -q
if errorlevel 1 goto :error

python scripts\run_phaseB_suite.py --wltc_csv data\cycles\WLTC3b.csv --out_dir artifacts\out_phaseB_suite_after_index_fix
if errorlevel 1 goto :error

python scripts\analyze_phaseB_suite.py --root artifacts\out_phaseB_suite_after_index_fix
if errorlevel 1 goto :error

echo.
echo All steps completed successfully.
exit /b 0

:error
echo.
echo Failed with errorlevel %errorlevel%.
exit /b %errorlevel%