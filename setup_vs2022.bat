: This script sets up a Visual Studio 2022 solution.

@echo off
setlocal

set PRESET_SUFFIX=""

if "%~1"=="ci" (
    set PRESET_SUFFIX="-ci"
)

: Fetch dependencies.
call %~dp0\setup.bat

: Configuration.
set PRESET=windows-vs2022%PRESET_SUFFIX%
set TOOLSET=host=x86
set CMAKE_EXE=%~dp0\tools\.packman\cmake\bin\cmake.exe
set CUSTOM_CUDA_DIR=%~dp0\external\packman\cuda

: Check if custom CUDA directory contains a valid CUDA SDK.
: Adjust toolset string to use the custom CUDA toolkit.
if exist %CUSTOM_CUDA_DIR%\bin\nvcc.exe (
    set TOOLSET=%TOOLSET%,cuda="%CUSTOM_CUDA_DIR%"
)

: Configure solution by running cmake.
echo Configuring Visual Studio solution ...
%CMAKE_EXE% --preset %PRESET% -T %TOOLSET%
if errorlevel 1 (
    echo Failed to configure solution!
    exit /b 1
)

: Success.
exit /b 0
