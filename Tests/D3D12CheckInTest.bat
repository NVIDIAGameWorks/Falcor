@echo off
REM Check In Test D3D12 - Runs the Release version of the D3D12 Tests, locally.
call python.exe RunTestsSet.py -rb -md ../ -ts TestConfigs\TS_ReleaseD3D12.json
