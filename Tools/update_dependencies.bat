@echo off
set PM_DISABLE_VS_WARNING=true
call "%~dp0..\Build\packman\packman.cmd " pull "%~dp0dependencies.xml" --platform windows-x86_64
if errorlevel 1 exit /b 1
exit /b 0
