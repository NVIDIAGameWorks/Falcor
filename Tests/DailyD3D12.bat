@ECHO off
REM Daily Test - Clones and Runs the D3D12 tests specified in TC_D3D12_Release
call python.exe RunTestsCollection.py -tc TestConfigs\TC_D3D12_Release.json -ne