@ECHO off
REM Daily Test - Clones and Runs the tests on the specified branch.
call python.exe RunGenerateReferences.py -tc TestConfigs\TC_Daily.json