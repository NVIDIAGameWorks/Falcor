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


rem Set the Falcor Backend to D3D12 by default.
set falcor_backend=FALCOR_D3D12

rem Set the Falcor Backend according to the Build Configuration.
if /I "%6"=="DebugD3D12" set falcor_backend=FALCOR_D3D12
if /I "%6"=="ReleaseD3D12" set falcor_backend=FALCOR_D3D12

if /I "%6"=="DebugVK" set falcor_backend=FALCOR_VK
if /I "%6"=="ReleaseVK" set falcor_backend=FALCOR_VK

rem Call Update Dependencies - Runs packman.
call %~dp0\update_dependencies.bat %1\Falcor\dependencies.xml
if errorlevel 1 exit /b 1

%1\Externals\.packman\Python\python.exe %~dp0\patchpropssheet.py %1 %2 %falcor_backend%
if errorlevel 1 exit /b 1
if exist %1\Internal\internal_dependencies.xml (call %~dp0\update_dependencies.bat %1\Internal\internal_dependencies.xml)
if errorlevel 1 exit /b 1
