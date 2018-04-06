@ECHO off
REM Generates reference screenshots for the D3D12 tests specified in TC_D3D12_Release
call python.exe RunGenerateReferences.py -tc TestConfigs\TC_D3D12_Release.json