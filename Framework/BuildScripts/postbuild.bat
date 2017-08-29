@echo off

rem %1 -> Falcor Core Directory Path
rem %2 -> Solution Directory Path
rem %3 -> Project Directory Path
rem %4 -> Platform Name.
rem %5 -> Platform Short Name.
rem %6 -> Configuration.
rem %7 -> Output Directory


rem echo "Falcor Core Directory Path:"
rem echo %1

rem echo "Solution Directory Path:"
rem echo %2

rem echo "Project Directory Path:"
rem echo %3

rem echo "Platform Name:"
rem echo %4

rem echo "Platform Short Name:"
rem echo %5

rem echo "Configuration:"
rem echo %6

rem echo "Output Directory:"
rem echo %7

rem Default to Debug.
set outdirtype=Debug

rem Set the Output Directory Type to Debug
if /I "%6"=="Debug" set outdirtype=Debug
if /I "%6"=="DebugD3D12" set outdirtype=Debug
if /I "%6"=="DebugVK" set outdirtype=Debug

rem Set the Output Directory Type to Release
if /I "%6"=="Release" set outdirtype=Release
if /I "%6"=="ReleaseD3D12" set outdirtype=Release
if /I "%6"=="ReleaseVK" set outdirtype=Release


rem Call the Build Scripts to move the data.
call %1BuildScripts\movedata.bat %1 %2 %3 %4 %5 %6 %7 %6

