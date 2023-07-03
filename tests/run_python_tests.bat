@echo off

if "%CONDA_PYTHON_EXE%"=="" (
    echo Python tests require conda environment to run.
    exit /b 1
)

python %~dp0testing/run_python_tests.py %*
