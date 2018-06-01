@echo off

rem %1 -> Falcor Core Directory Path
rem %2 -> Solution Directory Path
rem %3 -> Project Directory Path
rem %4 -> Platform Name.
rem %5 -> Platform Short Name.
rem %6 -> Configuration.
rem %7 -> Output Directory
rem %8 -> FALCOR_BACKEND

setlocal

SET ExternalsSourceDirectory=%1\Externals\
SET DestinationDirectory=%7

echo "%ExternalsSourceDirectory%"
echo "%DestinationDirectory%"

if not exist "%DestinationDirectory%" mkdir "%DestinationDirectory%"

robocopy %ExternalsSourceDirectory%\AntTweakBar\lib %DestinationDirectory% AntTweakBar64.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\FreeImage %DestinationDirectory%  freeimage.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\assimp\bin\%5 %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\FFMpeg\bin\%5 %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\D3DCompiler\%5 %DestinationDirectory%  D3Dcompiler_47.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\OptiX\bin64 %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\openvr\bin\win64 %DestinationDirectory%  openvr_api.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\Slang\bin\windows-x64\release %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\GLFW\lib %DestinationDirectory%  *.dll /r:0 >nul
call %1\BuildScripts\moveprojectdata.bat %1\Source\ %DestinationDirectory%
call %1\BuildScripts\moveprojectdata.bat %3 %DestinationDirectory% /r:0 >nul

rem copy or clear the DXR DLLs
if "%8%"=="FALCOR_DXR" (
	robocopy %ExternalsSourceDirectory%\DXR\bin\x64 %DestinationDirectory%  *.dll /r:0 >nul
	) ELSE (
	del %DestinationDirectory%\d3d12.dll >nul
	del %DestinationDirectory%\d3d12SDKLayers.dll >nul
	del %DestinationDirectory%\dxcompiler.dll >nul
	del %DestinationDirectory%\dxgidebug.dll >nul
)

rem robocopy sets the error level to something that is not zero even if the copy operation was successful. Set the error level to zero
exit /b 0