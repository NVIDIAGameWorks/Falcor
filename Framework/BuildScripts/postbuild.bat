@echo off

rem %1 -> Falcor Core Directory Path
rem %2 -> Solution Directory Path
rem %3 -> Project Directory Path
rem %4 -> Platform Name.
rem %5 -> Platform Short Name.
rem %6 -> Configuration.
rem %7 -> Output Directory


echo "Falcor Core Directory Path:"
echo %1

echo "Solution Directory Path:"
echo %2

echo "Project Directory Path:"
echo %3

echo "Platform Name:"
echo %4

echo "Platform Short Name:"
echo %5

echo "Configuration:"
echo %6

echo "Output Directory:"
echo %7

rem Default to Debug.
set outdirtype=Debug

rem Set the Output Directory Type to Debug 
if /I "%6"=="Debug" set outdirtype=Debug
if /I "%6"=="DebugD3D12" set outdirtype=Debug
if /I "%6"=="DebugVK" set outdirtype=Debug

rem Set the Output Directory Type to Release  
if /I "%6"=="Release" set outdirtype=Release
if /I "%6"=="ReleaseD3D12" set outdirtype=Release
if /I "%6"=="ReleaseVK" set outdirtype=Release


rem Call the Build Scripts to move the data.
call %1BuildScripts\movedata.bat %1 %2 %3 %4 %5 %6 %7 %outdirtype%

