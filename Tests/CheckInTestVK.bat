@echo off
REM Check In Test VK - Runs the Release version of the VK Tests, locally.
call python.exe RunTestsSet.py -rb -ts TestConfigs\TS_VK_Release.json -br "dev-3.0.9"
