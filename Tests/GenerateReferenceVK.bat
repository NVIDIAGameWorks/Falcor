@ECHO off
REM Generates reference screenshots for the Vulkan tests specified in TC_VK_Release
call python.exe RunGenerateReferences.py -tc TestConfigs\TC_VK_Release.json