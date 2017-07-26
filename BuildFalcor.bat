@echo on
if "%1"=="release" goto findVS
if "%1"=="debug" goto findVS
echo Usage: buildFalcor.bat ^<release^|debug^>

exit /B 1

:findVS
if defined VS120COMNTOOLS goto build
echo Can't find environment variable "VS120COMNTOOLS". Make sure Visual Studio 2013 is installed correctly.
:build
setlocal
echo Starting build
call "%VS120COMNTOOLS%\..\IDE\devenv.exe" Falcor.sln /rebuild %1 /project Framework/Source/Falcor.vcxproj
if not %errorlevel%==0 goto buildFailed
echo Build finished succesfully. Deploying...
exit /B 0

:buildFailed
echo Build failed...
exit /B 1