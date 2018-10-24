@echo off
REM Check In Test D3D12 - Runs the Release version of the D3D12 Tests, locally.
call python.exe RunTestsSet.py -rb -ts TestConfigs\TS_D3D12_Release.json -br "dev-3.0.9"
