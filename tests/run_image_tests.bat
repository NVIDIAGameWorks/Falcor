@echo off

set pwd=%~dp0
set project_dir=%pwd%..\
set python=%project_dir%tools\.packman\python\python.exe

if not exist %python% call %project_dir%setup.bat

call %python% %pwd%testing/run_image_tests.py %*
