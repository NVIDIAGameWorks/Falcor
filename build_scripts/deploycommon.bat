@echo off
setlocal

rem %1 -> Project directory
rem %2 -> Binary output directory
rem %3 -> Build configuration

set ExtDir=%1\external\packman\
set OutDir=%2

set IsDebug=0
if "%3" == "Debug" set IsDebug=1

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
robocopy %ExtDir%\pix\bin\x64 %OutDir% WinPixEventRuntime.dll /r:0 >nul
robocopy %ExtDir%\dxcompiler\bin\x64 %OutDir% dxil.dll /r:0 >nul
robocopy %ExtDir%\dxcompiler\bin\x64 %OutDir% dxcompiler.dll /r:0 >nul
robocopy %ExtDir%\nvtt\ %OutDir% cudart64_110.dll /r:0 >nul
robocopy %ExtDir%\nvtt\ %OutDir% nvtt30106.dll /r:0 >nul
robocopy %ExtDir%\cuda\bin\ %OutDir% cudart*.dll /r:0 >nul
robocopy %ExtDir%\cuda\bin\ %OutDir% nvrtc*.dll /r:0 >nul
robocopy %ExtDir%\cuda\bin\ %OutDir% cublas*.dll /r:0 >nul
robocopy %ExtDir%\cuda\bin\ %OutDir% curand*.dll /r:0 >nul

rem Copy NVAPI
set NvApiDir=%ExtDir%\nvapi
set NvApiTargetDir=%OutDir%\Shaders\NVAPI
if exist %NvApiDir% (
    if not exist %NvApiTargetDir% mkdir %NvApiTargetDir% >nul
    copy /y %NvApiDir%\nvHLSLExtns.h %NvApiTargetDir% >nul
    copy /y %NvApiDir%\nvHLSLExtnsInternal.h %NvApiTargetDir% >nul
    copy /y %NvApiDir%\nvShaderExtnEnums.h %NvApiTargetDir% >nul
)

rem Copy NRD
set NrdDir=%ExtDir%\nrd
set NrdTargetDir=%OutDir%\Shaders\nrd\Shaders
if exist %NrdDir% (
    if not exist %NrdTargetDir% mkdir %NrdTargetDir% >nul
    robocopy %NrdDir%\Shaders %NrdTargetDir% /s /r:0 >nul
    if %IsDebug% EQU 0 (
        robocopy %NrdDir%\Lib\Release %OutDir% *.dll /r:0 >nul
    ) else (
        robocopy %NrdDir%\Lib\Debug %OutDir% *.dll /r:0 >nul
    )
)

rem Copy RTXDI SDK shaders
set RtxdiSDKDir=%ExtDir%\rtxdi\rtxdi-sdk\include\rtxdi
set RtxdiSDKTargetDir=%OutDir%\Shaders\rtxdi
if exist %RtxdiSDKDir% (
    if not exist %RtxdiSDKTargetDir% mkdir %RtxdiSDKTargetDir% >nul
    copy /y %RtxdiSDKDir%\ResamplingFunctions.hlsli %RtxdiSDKTargetDir% >nul
    copy /y %RtxdiSDKDir%\Reservoir.hlsli %RtxdiSDKTargetDir% >nul
    copy /y %RtxdiSDKDir%\RtxdiHelpers.hlsli %RtxdiSDKTargetDir% >nul
    copy /y %RtxdiSDKDir%\RtxdiMath.hlsli %RtxdiSDKTargetDir% >nul
    copy /y %RtxdiSDKDir%\RtxdiParameters.h %RtxdiSDKTargetDir% >nul
    copy /y %RtxdiSDKDir%\RtxdiTypes.h %RtxdiSDKTargetDir% >nul
)

rem Copy RTXGI SDK shaders
set RtxgiApiDir=%ExtDir%\rtxgi\rtxgi-sdk
set RtxgiTargetDir=%OutDir%\Shaders\rtxgi
if exist %RtxgiApiDir% (
    if not exist %RtxgiTargetDir% mkdir %RtxgiTargetDir% >nul
    robocopy %RtxgiApiDir%\include\ %RtxgiTargetDir%\include\ /s /r:0 >nul
    robocopy %RtxgiApiDir%\shaders\ %RtxgiTargetDir%\shaders\ /s /r:0 >nul
)

rem Copy Agility SDK Runtime
set AgilitySDKDir=%ExtDir%\agility-sdk
set AgilitySDKTargetDir=%OutDir%\D3D12
if exist %AgilitySDKDir% (
    if not exist %AgilitySDKTargetDir% mkdir %AgilitySDKTargetDir% >nul
    copy /y %AgilitySDKDir%\build\native\bin\x64\D3D12Core.dll %AgilitySDKTargetDir% >nul
    copy /y %AgilitySDKDir%\build\native\bin\x64\d3d12SDKLayers.dll %AgilitySDKTargetDir% >nul
)

rem Copy NanoVDB
set NanoVDBDir=%ExtDir%\nanovdb
set NanoVDBTargetDir=%OutDir%\Shaders\NanoVDB
if exist %NanoVDBDir% (
    if not exist %NanoVDBTargetDir% mkdir %NanoVDBTargetDir% >nul
    copy /y %NanoVDBDir%\include\nanovdb\PNanoVDB.h %NanoVDBTargetDir% >nul
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
set MDLDir=%ExtDir%\mdl-sdk
if exist %MDLDir% (
    robocopy %MDLDir%\nt-x86-64\lib %OutDir% *.dll /r:0 >nul
)

rem Copy NVTT
if %IsDebug% EQU 0 (
    robocopy %ExtDir%\nvtt\lib\x64-v141\Release %OutDir% nvtt.dll /r:0 >nul
) else (
    robocopy %ExtDir%\nvtt\lib\x64-v141\Debug %OutDir% nvtt.dll /r:0 >nul
)

rem Copy DLSS
set NGXDir=%ExtDir%\ngx
if exist %NGXDir% (
    robocopy %NGXDir%\lib\Windows_x86_64\rel %OutDir% nvngx_dlss.dll /r:0 >nul
)

rem robocopy sets the error level to something that is not zero even if the copy operation was successful. Set the error level to zero
exit /b 0
