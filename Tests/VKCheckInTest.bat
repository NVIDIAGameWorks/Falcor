@echo off
REM Check In Test VK - Runs the Release version of the D3D12 Tests, locally.
call python.exe RunTestsSet.py -rb -md ../ -ts TestConfigs\TS_ReleaseVK.json
