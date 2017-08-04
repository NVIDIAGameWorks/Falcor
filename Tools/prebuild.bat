@echo off

rem %1 -> Solution Directory Path.
rem %2 -> Project Directory Path.
rem %3 -> Platform Name.
rem %4 -> Platform Short Name.
rem %5 -> Configuration.

rem Echo Directory Paths and Backend for Falcor.
echo "Solution Directory Path:"
echo %1

echo "Project Directory Path:"
echo %2

echo "Platform Name:"
echo %3

echo "Platform Short Name:"
echo %4

echo "Configuration:"
echo %5

rem Call Update Dependencies - Runs packman.
call %1\update_dependencies.bat 

rem Set the Falcor Backend to D3D12 by default.
set falcor_backend=FALCOR_D3D12

rem Set the Falcor Backend according to the Build Configuration.
if /I "%5"=="DebugD3D12" set falcor_backend=FALCOR_D3D12
if /I "%5"=="ReleaseD3D12" set falcor_backend=FALCOR_D3D12

if /I "%5"=="DebugVK" set falcor_backend=FALCOR_VK
if /I "%5"=="ReleaseVK" set falcor_backend=FALCOR_VK

echo %2
echo %1\.\framework\source\\..\

rem Change the Props File to Align with the Backend.
rem This also changes the FALCOR_PROJECT_DIR
start /wait /b %1\Tools\PatchFalcorProps\PatchFalcorPropertySheet.exe %1\Falcor.sln %2\Falcor.props %falcor_backend%


rem Commented out.
rem Use Python - Adds dependency.
rem Change the Props file to Align with the Backend.
rem start /wait /b python.exe %1/Tools/PatchFalcorProps.py %1 %2 %3 %4 %5


