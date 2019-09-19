@echo off
IF [%1] == [] GOTO helpMsg
set PM_DISABLE_VS_WARNING=true
call "%~dp0packman\packman.cmd " pull "%1" --platform win
if errorlevel 1 exit /b 1
exit /b 0

:helpMsg
echo Please specify a dependency file
exit /b 1