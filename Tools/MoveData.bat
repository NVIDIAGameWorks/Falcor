rem %1 -> Solution Directory Path.
rem %2 -> Project Directory Path.
rem %3 -> Platform Name.
rem %4 -> Platform Short Name.
rem %5 -> Configuration.

rem Echo Directory Paths and Backend for Falcor.
echo %1
echo %2
echo %3
echo %4
echo %5


if not exist %2\..\Bin\%4\Release mkdir %2\..\Bin\%4\Release