@echo off

set pwd=%~dp0
set project_dir=%pwd%..\
set python=%project_dir%tools\.packman\python\python.exe

if not exist %python% call %project_dir%setup.bat

call %python% %pwd%/make_new_render_pass.py %*
