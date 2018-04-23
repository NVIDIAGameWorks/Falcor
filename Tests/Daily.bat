@ECHO off
REM Daily Test - Clones and Runs the tests specified in TC_Daily (D3D12 and Vulkan)
call python.exe RunTestsCollection.py -tc TestConfigs\TC_Daily.json -ne