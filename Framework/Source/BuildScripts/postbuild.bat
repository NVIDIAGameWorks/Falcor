@echo on

rem %1 -> Solution Directory Path.
rem %2 -> Project Directory Path.
rem %3 -> Platform Name.
rem %4 -> Platform Short Name.
rem %5 -> Configuration.

rem Echo Directory Paths and Backend for Falcor.
echo %1
echo %2
echo %3
echo %4
echo %5

set outdirtype=Debug

if /I "%5"=="Debug" set outdirtype=Debug
if /I "%5"=="DebugD3D12" set outdirtype=Debug
if /I "%5"=="DebugVK" set outdirtype=Debug

if /I "%5"=="Release" set outdirtype=Release
if /I "%5"=="ReleaseD3D12" set outdirtype=Release
if /I "%5"=="ReleaseVK" set outdirtype=Release


call %2\BuildScripts\movedata.bat %1 %2 %3 %4 %5

