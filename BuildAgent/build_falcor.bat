@echo off

set pwd=%~dp0
set project_dir=%pwd%..\
set python=%project_dir%Tools\.packman\Python\python.exe

set PM_DISABLE_VS_WARNING=true

if not exist %python% call %project_dir%Tools\update_dependencies.bat

call "%~dp0..\Build\packman\packman.cmd " pull "%~dp0build_agent_dependencies.xml" --platform windows-x86_64

call %python% %pwd%\build_falcor.py %*
