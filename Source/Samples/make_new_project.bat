@echo off

set pwd=%~dp0
set project_dir=%pwd%..\..\
set python=%project_dir%Tools\.packman\Python\python.exe

if not exist %python% call %project_dir%Tools\update_dependencies.bat

call %python% %pwd%/make_new_project.py %*
