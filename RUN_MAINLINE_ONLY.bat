@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>nul
if errorlevel 1 (
  echo Conda was not found on PATH. Start from an Anaconda Prompt or add conda to PATH.
  exit /b 1
)
set CONDA_ENV=pytorch
call conda activate %CONDA_ENV%
if errorlevel 1 exit /b %errorlevel%

python scripts\00_audit_data.py
if errorlevel 1 exit /b %errorlevel%
python scripts\01_preprocess_cips.py
if errorlevel 1 exit /b %errorlevel%
python scripts\02_preprocess_cdmet.py
if errorlevel 1 exit /b %errorlevel%
python scripts\03_preprocess_water_quality.py
if errorlevel 1 exit /b %errorlevel%
python scripts\04_build_crsm_index.py
if errorlevel 1 exit /b %errorlevel%
python scripts\05_build_scenario_stress_tests.py
if errorlevel 1 exit /b %errorlevel%
python scripts\06_train_models.py
if errorlevel 1 exit /b %errorlevel%
python scripts\07_evaluate_models.py
if errorlevel 1 exit /b %errorlevel%
python scripts\08_interpretability.py
if errorlevel 1 exit /b %errorlevel%
python scripts\12_freeze_and_qc.py
if errorlevel 1 exit /b %errorlevel%

echo Mainline C-RSM analysis pipeline completed.
endlocal
