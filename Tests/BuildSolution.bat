@echo off
setlocal ENABLEDELAYEDEXPANSION 

set "action="
if "%1"=="clean" set action=clean
if "%1"=="build" set action=build
if "%1"=="rebuild" set action=rebuild
if not defined action goto usage

set "config="
if "%3"=="released3d12" set config=released3d12
if "%3"=="debugd3d12" set config=debugd3d12
if "%3"=="releasevk" set config=releasevk
if "%3"=="debugvk" set config=debugvk
if not defined config goto usage

goto findVS

:findVS
if defined VS140COMNTOOLS goto act
echo Can't find environment variable "VS140COMNTOOLS".  

:act
set solution=%2
set project=%4
set errFileSuffix=_BuildLog.txt
if defined project (
    set errFile="%project%%errFileSuffix%"
    echo Starting %action% of %config% config of project %project% in solution %2
    call "%VS140COMNTOOLS%\..\IDE\devenv.com" %solution% /%action% %config% /project %project% > !errFile!
) else (
    set errFile="Solution%errFileSuffix%"
    echo Starting %action% of %config% config of entire solution %2
    call "%VS140COMNTOOLS%\..\IDE\devenv.com" %solution% /%action% %config% > !errFile!
)
if not %errorlevel%==0 (
    goto buildFailed 
)else (
    del !errFile!
)

exit /B 0

:buildFailed
echo Build Failed
exit /B 1

:usage
echo Usage: BuildSolution.bat ^<clean^|build^|rebuild^> solution config [projectName(entire sln if omitted)]
exit /B 1
