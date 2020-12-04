@echo off
if [%1] == [] goto helpMsg
set PM_DISABLE_VS_WARNING=true
call "%~dp0packman\packman.cmd " pull "%1" --platform windows-x86_64
if errorlevel 1 exit /b 1
exit /b 0

:helpMsg
echo Please specify a dependency file
exit /b 1
