@echo off
setlocal

rem %1 -> Falcor Core Directory Path
rem %2 -> Platform Short Name
rem %3 -> Output Directory
rem %4 -> WINDSDK Directory

setlocal

SET ExtDir=%1\Externals\.packman\
SET OutDir=%3
SET FalcorDir=%1\Falcor\
if not exist "%OutDir%" mkdir "%OutDir%"

rem Copy Falcor's files
IF not exist %OutDir%\Data\ mkdir %OutDir%\Data >nul
IF exist %FalcorDir%\ShadingUtils\ (xcopy %FalcorDir%\ShadingUtils\*.* %OutDir%\Data /s /y /d)
IF exist %FalcorDir%\Raytracing\Data\ (xcopy %FalcorDir%\Raytracing\Data\*.* %OutDir%\Data /s /y /d /q >nul)
call %~dp0\deployproject.bat %FalcorDir% %OutDir%

rem Copy externals
robocopy %ExtDir%\Python\ %OutDir% Python37*.dll /r:0 >nul
robocopy %ExtDir%\Python %OutDir%\Python /E /r:0 >nul
robocopy %ExtDir%\AntTweakBar\lib %OutDir% AntTweakBar64.dll /r:0 >nul
robocopy %ExtDir%\FreeImage %OutDir%  freeimage.dll /r:0 >nul
robocopy %ExtDir%\assimp\bin\%2 %OutDir%  *.dll /r:0 >nul
robocopy %ExtDir%\FFMpeg\bin\%2 %OutDir%  *.dll /r:0 >nul
robocopy %ExtDir%\dxcompiler\%2 %OutDir%  dxcompiler.dll /r:0 >nul
rem robocopy %ExtDir%\dxcompiler\windows-x86_64\release\%2 %OutDir%  *.dll /r:0 >nul
robocopy %ExtDir%\OptiX\bin64 %OutDir%  *.dll /r:0 >nul
robocopy %ExtDir%\openvr\bin\win64 %OutDir%  openvr_api.dll /r:0 >nul
robocopy %ExtDir%\Slang\bin\windows-x64\release %OutDir%  *.dll /r:0 >nul
robocopy %ExtDir%\GLFW\lib %OutDir%  *.dll /r:0 >nul
robocopy %ExtDir%\WinPixEventRuntime\bin\x64 %OutDir% WinPixEventRuntime.dll /r:0 >nul
robocopy "%~4\Redist\D3D\%2" %OutDir% dxil.dll /r:0 >nul

rem Copy NVAPI
set NvApiDir=%ExtDir%\NVAPI
IF exist %NvApiDir% (
    IF not exist %OutDir%\Data\NVAPI mkdir %OutDir%\Data\NVAPI >nul
    copy /y %NvApiDir%\nvHLSLExtns.h %OutDir%\Data\NVAPI
    copy /y %NvApiDir%\nvHLSLExtnsInternal.h %OutDir%\Data\NVAPI
    copy /y %NvApiDir%\nvShaderExtnEnums.h %OutDir%\Data\NVAPI
)

rem robocopy sets the error level to something that is not zero even if the copy operation was successful. Set the error level to zero
exit /b 0
