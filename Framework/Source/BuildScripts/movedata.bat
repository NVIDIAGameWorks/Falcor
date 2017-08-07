rem %1 -> Solution Directory Path.
rem %2 -> Project Directory Path.
rem %3 -> Platform Name.
rem %4 -> Platform Short Name.
rem %5 -> Configuration.

SETLOCAL ENABLEDELAYEDEXPANSION

rem Echo Directory Paths and Backend for Falcor.
echo MOVEDATA
echo %1
echo %2
echo %3
echo %4
echo %5


SET ExternalsSourceDirectory=%2\..\
SET DestinationDirectory=%2\..\..\Bin\%4\Release\

echo "%ExternalsSourceDirectory%"
echo "%DestinationDirectory%"

if not exist "%DestinationDirectory%" mkdir "%DestinationDirectory%"


robocopy %ExternalsSourceDirectory%\Externals\AntTweakBar\lib %DestinationDirectory% "AntTweakBar64.dll" /r:0 >nul
robocopy %ExternalsSourceDirectory%\Externals\FreeImage %DestinationDirectory%  freeimage.dll /r:0 >nul 
robocopy %ExternalsSourceDirectory%\Externals\assimp\bin\%3 %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\Externals\FFMpeg\bin\%3 %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\Externals\D3DCompiler\%3 %DestinationDirectory%  D3Dcompiler_47.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\Externals\OptiX\bin64 %DestinationDirectory%  *.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\Externals\openvr\bin\win64 %DestinationDirectory%  openvr_api.dll /r:0 >nul
robocopy %ExternalsSourceDirectory%\Externals\Slang\bin\windows-x64\release %DestinationDirectory%  *.dll /r:0 >nul


rem Copy and overwrite internal files - no longer used?
rem for /r "%FALCOR_PROJECT_DIR%\..\..\Internals\" %%f in (*.dll) do @robocopy "%%~df%%"~pf %3 %%~nf%%~xf /r:0 >nul

echo %1
robocopy %2\BuildScripts\ %DestinationDirectory% moveprojectdata.bat /r:0 >nul
call %DestinationDirectory%\moveprojectdata.bat %1 %DestinationDirectory%