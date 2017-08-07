@echo off

rem %1 -> Solution Directory Path.
rem %2 -> Project Directory Path.
rem %3 -> Platform Name.
rem %4 -> Platform Short Name.
rem %5 -> Configuration.

rem Echo Directory Paths and Backend for Falcor.
rem echo "Solution Directory Path:"
rem echo %1

rem echo "Project Directory Path:"
rem echo %2

rem echo "Platform Name:"
rem echo %3

rem echo "Platform Short Name:"
rem echo %4

rem echo "Configuration:"
rem echo %5

rem Call Update Dependencies - Runs packman.
call %1\update_dependencies.bat 

rem Set the Falcor Backend to D3D12 by default.
set falcor_backend=FALCOR_D3D12

rem Set the Falcor Backend according to the Build Configuration.
if /I "%5"=="DebugD3D12" set falcor_backend=FALCOR_D3D12
if /I "%5"=="ReleaseD3D12" set falcor_backend=FALCOR_D3D12

if /I "%5"=="DebugVK" set falcor_backend=FALCOR_VK
if /I "%5"=="ReleaseVK" set falcor_backend=FALCOR_VK

rem Change the Props File to Align with the Backend.
rem This also changes the FALCOR_PROJECT_DIR
start /wait /b %2BuildScripts\PatchFalcorProps\PatchFalcorPropertySheet.exe %1\Falcor.sln %2\Falcor.props %falcor_backend%


rem Commented out.
rem Use Python - Adds dependency.
rem Change the Props file to Align with the Backend.
rem start /wait /b python.exe %1/Tools/PatchFalcorProps.py %1 %2 %3 %4 %5


