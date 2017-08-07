@echo on
rem %1==config %2==platformname %3==outputdir
set local

rem set some local variables
if "%2" == "Win32" SET ATBDLL="AntTweakBar.dll"
if "%2" == "x64" SET ATBDLL="AntTweakBar64.dll"

echo %~dp0

if not exist %3 mkdir %3
SET FALCOR_PROJECT_DIR=%~dp0
echo %FALCOR_PROJECT_DIR%
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\AntTweakBar\lib" %3  %ATBDLL% /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\FreeImage" %3  freeimage.dll /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\assimp\bin\%2" %3  *.dll /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\FFMpeg\bin\%2" %3  *.dll /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\D3DCompiler\%2" %3  D3Dcompiler_47.dll /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\OptiX\bin64" %3  *.dll /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\openvr\bin\win64" %3  openvr_api.dll /r:0 >nul
rem robocopy "%FALCOR_PROJECT_DIR%\Externals\Slang\bin\windows-x64\release" %3  *.dll /r:0 >nul

rem copy and overwrite internal files
rem for /r "%FALCOR_PROJECT_DIR%\..\..\Internals\" %%f in (*.dll) do @robocopy "%%~df%%"~pf %3 %%~nf%%~xf /r:0 >nul

rem robocopy "%FALCOR_PROJECT_DIR%\" %3 CopyData.bat /r:0 >nul
rem call %FALCOR_PROJECT_DIR%\CopyData.bat %FALCOR_PROJECT_DIR%\Source %3 >nul