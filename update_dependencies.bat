@set PM_DISABLE_VS_WARNING=true
@if not exist %~dp0\Framework\Externals mkdir %~dp0\Framework\Externals
@call "%~dp0packman\packman.cmd " pull "%~dp0dependencies.xml"
@if errorlevel 1 exit 1
