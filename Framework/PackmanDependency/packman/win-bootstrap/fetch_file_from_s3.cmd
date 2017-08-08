:: You need to specify <package-name> <target-name> as input to this command

@set PACKAGE_NAME=%1
@set TARGET=%2

@echo Fetching %PACKAGE_NAME% from s3 ...

@powershell -ExecutionPolicy ByPass -NoLogo -NoProfile -File "%~dp0fetch_file_from_s3.ps1" -sourceName %PACKAGE_NAME% -output %TARGET%
:: A bug in powershell prevents the errorlevel code from being set when using the -File execution option
:: We must therefore do our own failure analysis, basically make sure the file exists and is larger than 0 bytes:
@if not exist %TARGET% goto ERROR_DOWNLOAD_FAILED
@if %~z2==0 goto ERROR_DOWNLOAD_FAILED

@exit /b 0

:ERROR_DOWNLOAD_FAILED
@echo Failed to download file from %1
@echo Most likely because endpoint cannot be reached (VPN connection down?)
@exit /b 1