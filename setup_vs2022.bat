: This script sets up a Visual Studio 2022 solution.
: It takes an optional argument to specify the render backend:
: - d3d12       Native D3D12 backend (default)
: - gfx-d3d12   Slang/GFX D3D12 backend
: - gfx-vk      Slang/GFX Vulkan backend

@echo off
setlocal

if "%~1"=="" (
    set BACKEND=d3d12
) else if "%~1"=="d3d12" (
    set BACKEND=d3d12
) else if "%~1"=="gfx-d3d12" (
    set BACKEND=gfx-d3d12
) else if "%~1"=="gfx-vk" (
    set BACKEND=gfx-vk
) else (
    echo Error: Unknown rendering backend, use d3d12, gfx-d3d12 or gfx-vk.
    exit /b 1
)

: Fetch dependencies.
call %~dp0\setup.bat

: Configuration.
set PRESET=windows-vs2022-%BACKEND%
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
