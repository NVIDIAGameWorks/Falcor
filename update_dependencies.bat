@set PM_DISABLE_VS_WARNING=true
@if not exist %~dp0Framework\Externals mkdir %~dp0Framework\Externals 
@call "%~dp0packman\packman.cmd " pull "%~dp0dependencies.xml"
@if errorlevel 1 exit 1
