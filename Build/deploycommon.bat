@echo off
setlocal


rem %1 -> Falcor Core Directory Path
rem %2 -> Platform Short Name
rem %3 -> Output Directory
rem %4 -> WINDSDK Directory

setlocal

set ExtDir=%1\Externals\.packman\
set OutDir=%3
set FalcorDir=%1\Falcor\
if not exist "%OutDir%" mkdir "%OutDir%"

set IsDebug=0
if "%OutDir:~-6%" == "Debug\" set IsDebug=1
if "%OutDir:~-5%" == "Debug" set IsDebug=1

rem Copy Falcor's files
if not exist %OutDir%\Data\ mkdir %OutDir%\Data >nul
call %~dp0\deployproject.bat %FalcorDir% %OutDir%

rem Copy externals
if %IsDebug% EQU 0 (
    robocopy %ExtDir%\deps\bin\ %OutDir% /E /r:0 >nul
) else (
    robocopy %ExtDir%\deps\debug\bin\ %OutDir% /E /r:0 >nul
    robocopy %ExtDir%\deps\bin\ %OutDir% assimp-vc142-mt.* /r:0 >nul
    rem Needed for OpenVDB (debug version links to release version of Half_2.5)
    robocopy %ExtDir%\deps\bin\ %OutDir% Half-2_5.* /r:0 >nul
)
robocopy %ExtDir%\python\ %OutDir% Python37*.dll /r:0 >nul
robocopy %ExtDir%\python %OutDir%\Python /E /r:0 >nul
robocopy %ExtDir%\slang\bin\windows-x64\release %OutDir% *.dll /r:0 >nul
robocopy %ExtDir%\WinPixEventRuntime\bin\x64 %OutDir% WinPixEventRuntime.dll /r:0 >nul
robocopy %ExtDir%\dxcompiler\bin\x64 %OutDir% dxil.dll /r:0 >nul
robocopy %ExtDir%\dxcompiler\bin\x64 %OutDir% dxcompiler.dll /r:0 >nul
robocopy %ExtDir%\nvtt\ %OutDir% cudart64_110.dll /r:0 >nul
robocopy %ExtDir%\nvtt\ %OutDir% nvtt30106.dll /r:0 >nul
robocopy %ExtDir%\Cuda\bin\ %OutDir% cudart*.dll /r:0 >nul
robocopy %ExtDir%\Cuda\bin\ %OutDir% nvrtc*.dll /r:0 >nul
robocopy %ExtDir%\Cuda\bin\ %OutDir% cublas*.dll /r:0 >nul

rem Copy NVAPI
set NvApiDir=%ExtDir%\nvapi
set NvApiTargetDir=%OutDir%\Shaders\NVAPI
if exist %NvApiDir% (
    if not exist %NvApiTargetDir% mkdir %NvApiTargetDir% >nul
    copy /y %NvApiDir%\nvHLSLExtns.h %NvApiTargetDir%
    copy /y %NvApiDir%\nvHLSLExtnsInternal.h %NvApiTargetDir%
    copy /y %NvApiDir%\nvShaderExtnEnums.h %NvApiTargetDir%
)

rem Copy Agility SDK Runtime
set AgilitySDKDir=%ExtDir%\agility-sdk
set AgilitySDKTargetDir=%OutDir%\D3D12
if exist %AgilitySDKDir% (
    if not exist %AgilitySDKTargetDir% mkdir %AgilitySDKTargetDir% >nul
    copy /y %AgilitySDKDir%\build\native\bin\x64\D3D12Core.dll %AgilitySDKTargetDir%
    copy /y %AgilitySDKDir%\build\native\bin\x64\d3d12SDKLayers.dll %AgilitySDKTargetDir%
)

rem Copy NanoVDB
set NanoVDBApiDir=%ExtDir%\nanovdb
set NanoVDBTargetDir=%OutDir%\Shaders\NanoVDB
if exist %NanoVDBApiDir% (
    if not exist %NanoVDBTargetDir% mkdir %NanoVDBTargetDir% >nul
    copy /y %NanoVDBApiDir%\include\nanovdb\PNanoVDB.h %NanoVDBTargetDir%
)

rem Copy USD files, making sure not to overwrite dlls provided by other components, or dlls that we don't need.
if %IsDebug% EQU 0 (
    robocopy %ExtDir%\nv-usd-release\lib %OutDir% *.dll /r:0 /XF Alembic.dll dds.dll nv_freeimage.dll python*.dll hdf5*.dll tbb*.dll >nul
    robocopy %ExtDir%\nv-usd-release\lib\usd %OutDir%\usd /E /r:0 >nul
    robocopy %ExtDir%\nv-usd-release\lib\python\pxr %OutDir%\Python\Lib\pxr /E /r:0 >nul
) else (
    robocopy %ExtDir%\nv-usd-debug\lib %OutDir% *.dll /r:0 /XF Alembic.dll dds.dll nv_freeimage.dll python*.dll hdf5*.dll tbb*.dll >nul
    robocopy %ExtDir%\nv-usd-debug\lib\usd %OutDir%\usd /E /r:0 >nul
    robocopy %ExtDir%\nv-usd-debug\lib\python\pxr %OutDir%\Python\Lib\pxr /E /r:0 >nul
)

rem Copy MDL libs after USD to overwrite older versions included in USD distribution
set MDLDir=%ExtDir%\mdl_sdk
if exist %MDLDir% (
    robocopy %MDLDir%\nt-x86-64\lib %OutDir% *.dll /r:0 >nul
)

rem Copy NVTT
if %IsDebug% EQU 0 (
    robocopy %ExtDir%\nvtt\lib\x64-v141\Release %OutDir% nvtt.dll /r:0 >nul
) else (
    robocopy %ExtDir%\nvtt\lib\x64-v141\Debug %OutDir% nvtt.dll /r:0 >nul
)

rem robocopy sets the error level to something that is not zero even if the copy operation was successful. Set the error level to zero
exit /b 0
