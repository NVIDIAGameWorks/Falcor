@ECHO off
REM Daily Test - Clones and Runs the Vulkan tests specified in TC_VK_Release
call python.exe RunTestsCollection.py -tc TestConfigs\TC_VK_Release.json -ne