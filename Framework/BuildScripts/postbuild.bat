@echo off

rem %1 -> Falcor Core Directory Path
rem %2 -> Solution Directory Path
rem %3 -> Project Directory Path
rem %4 -> Platform Name.
rem %5 -> Platform Short Name.
rem %6 -> Configuration.
rem %7 -> Output Directory
rem %8 -> FALCOR_BACKEND

rem echo "Falcor Core Directory Path:"
rem echo %1

rem echo "Solution Directory Path:"
rem echo %2

rem echo "Project Directory Path:"
rem echo %3

rem echo "Platform Name:"
rem echo %4

rem echo "Platform Short Name:"
rem echo %5

rem echo "Configuration:"
rem echo %6

rem echo "Output Directory:"
rem echo %7

rem echo "Falcor Backend:"
rem echo %8

rem Call the Build Scripts to move the data.
call %1BuildScripts\movedata.bat %1 %2 %3 %4 %5 %6 %7 %8
