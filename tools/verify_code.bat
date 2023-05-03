@echo off

set pwd=%~dp0
set project_dir=%pwd%..\
set python=%project_dir%tools\.packman\python\python.exe
set clang_format=%project_dir%tools\.packman\clang-format\clang-format.exe

if not exist %python% call %project_dir%setup.bat
if not exist %clang_format% call %project_dir%setup.bat

pushd %project_dir%
call %python% %pwd%run_clang_format.py --clang-format-executable=%clang_format% --color=never -r Source
popd
