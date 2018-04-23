@ECHO off
REM Generates reference screenshots for the tests specified in TC_Daily (D3D12 and Vulkan)
call python.exe RunGenerateReferences.py -tc TestConfigs\TC_Daily.json