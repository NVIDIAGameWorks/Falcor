@set PM_PACKMAN_VERSION=5.14

:: Specify where packman command is rooted
@set PM_INSTALL_PATH=%~dp0..

:: The external root may already be configured and we should do minimal work in that case
@if defined PM_PACKAGES_ROOT goto ENSURE_DIR

:: If the folder isn't set we assume that the best place for it is on the drive that we are currently
:: running from
@set PM_DRIVE=%CD:~0,2%

@set PM_PACKAGES_ROOT=%PM_DRIVE%\packman-repo

:: We use *setx* here so that the variable is persisted in the user environment
@echo Setting user environment variable PM_PACKAGES_ROOT to %PM_PACKAGES_ROOT%
@setx PM_PACKAGES_ROOT %PM_PACKAGES_ROOT%
@if errorlevel 1 goto ERROR

:: The above doesn't work properly from a build step in VisualStudio because a separate process is
:: spawned for it so it will be lost for subsequent compilation steps - VisualStudio must
:: be launched from a new process. We catch this odd-ball case here:
@if defined PM_DISABLE_VS_WARNING goto ENSURE_DIR
@if not defined VSLANG goto ENSURE_DIR
@echo The above is a once-per-computer operation. Unfortunately VisualStudio cannot pick up environment change
@echo unless *VisualStudio is RELAUNCHED*.
@echo If you are launching VisualStudio from command line or command line utility make sure
@echo you have a fresh launch environment (relaunch the command line or utility).
@echo If you are using 'linkPath' and referring to packages via local folder links you can safely ignore this warning.
@echo You can disable this warning by setting the environment variable PM_DISABLE_VS_WARNING.
@echo.

:: Check for the directory that we need. Note that mkdir will create any directories
:: that may be needed in the path 
:ENSURE_DIR
@if not exist "%PM_PACKAGES_ROOT%" (
	@echo Creating directory %PM_PACKAGES_ROOT%
	@mkdir "%PM_PACKAGES_ROOT%"
	@if errorlevel 1 goto ERROR_MKDIR_PACKAGES_ROOT
)

:: The Python interpreter may already be externally configured
@if defined PM_PYTHON_EXT (
	@set PM_PYTHON=%PM_PYTHON_EXT%
	@goto PACKMAN
)

@set PM_PYTHON_VERSION=2.7.14-windows-x86_32
@set PM_PYTHON_BASE_DIR=%PM_PACKAGES_ROOT%\python
@set PM_PYTHON_DIR=%PM_PYTHON_BASE_DIR%\%PM_PYTHON_VERSION%
@set PM_PYTHON=%PM_PYTHON_DIR%\python.exe

@if exist "%PM_PYTHON%" goto PACKMAN
@if not exist "%PM_PYTHON_BASE_DIR%" call :CREATE_PYTHON_BASE_DIR

@set PM_PYTHON_PACKAGE=python@%PM_PYTHON_VERSION%.cab
@for /f "delims=" %%a in ('powershell -ExecutionPolicy ByPass -NoLogo -NoProfile -File "%~dp0\generate_temp_file_name.ps1"') do @set TEMP_FILE_NAME=%%a
@set TARGET=%TEMP_FILE_NAME%.zip
@call "%~dp0fetch_file_from_s3.cmd" %PM_PYTHON_PACKAGE% "%TARGET%"
@if errorlevel 1 goto ERROR

@for /f "delims=" %%a in ('powershell -ExecutionPolicy ByPass -NoLogo -NoProfile -File "%~dp0\generate_temp_folder.ps1" -parentPath "%PM_PYTHON_BASE_DIR%"') do @set TEMP_FOLDER_NAME=%%a
@echo Unpacking Python interpreter ...
@"%SystemRoot%\system32\expand.exe" -F:* "%TARGET%" "%TEMP_FOLDER_NAME%" 1> nul
@del "%TARGET%"
:: Failure during extraction to temp folder name, need to clean up and abort
@if errorlevel 1 (
    @call :CLEAN_UP_TEMP_FOLDER
    @goto ERROR
)

:: If python has now been installed by a concurrent process we need to clean up and then continue
@if exist "%PM_PYTHON%" (
    @call :CLEAN_UP_TEMP_FOLDER
    @goto PACKMAN
) else (
    @if exist "%PM_PYTHON_DIR%" ( @rd /s /q "%PM_PYTHON_DIR%" > nul )
)

:: Perform atomic rename
@rename "%TEMP_FOLDER_NAME%" "%PM_PYTHON_VERSION%" 1> nul
:: Failure during move, need to clean up and abort
@if errorlevel 1 (
    @call :CLEAN_UP_TEMP_FOLDER
    @goto ERROR
)

:PACKMAN
:: The packman module may already be externally configured
@if defined PM_MODULE_DIR_EXT (
	@set PM_MODULE_DIR=%PM_MODULE_DIR_EXT%
) else (
    @set PM_MODULE_DIR=%PM_PACKAGES_ROOT%\packman-common\%PM_PACKMAN_VERSION%
)

@set PM_MODULE=%PM_MODULE_DIR%\packman.py

@if exist "%PM_MODULE%" goto ENSURE_7ZA

@set PM_MODULE_PACKAGE=packman-common@%PM_PACKMAN_VERSION%.zip
@for /f "delims=" %%a in ('powershell -ExecutionPolicy ByPass -NoLogo -NoProfile -File "%~dp0\generate_temp_file_name.ps1"') do @set TEMP_FILE_NAME=%%a
@set TARGET=%TEMP_FILE_NAME%
@call "%~dp0fetch_file_from_s3.cmd" %PM_MODULE_PACKAGE% "%TARGET%"
@if errorlevel 1 goto ERROR

@echo Unpacking ...
@"%PM_PYTHON%" -S -s -u -E "%~dp0\install_package.py" "%TARGET%" "%PM_MODULE_DIR%"
@if errorlevel 1 goto ERROR

@del "%TARGET%"

:ENSURE_7ZA
@set PM_7Za_VERSION=16.02.4
@set PM_7Za_PATH=%PM_PACKAGES_ROOT%\7za\%PM_7ZA_VERSION%
@if exist "%PM_7Za_PATH%" goto END
@set PM_7Za_PATH=%PM_PACKAGES_ROOT%\chk\7za\%PM_7ZA_VERSION%
@if exist "%PM_7Za_PATH%" goto END

@"%PM_PYTHON%" -S -s -u -E "%PM_MODULE%" pull "%PM_MODULE_DIR%\deps.packman.xml"
@if errorlevel 1 goto ERROR

@goto END

:ERROR_MKDIR_PACKAGES_ROOT
@echo Failed to automatically create packman packages repo at %PM_PACKAGES_ROOT%.
@echo Please set a location explicitly that packman has permission to write to, by issuing:
@echo.
@echo    setx PM_PACKAGES_ROOT {path-you-choose-for-storing-packman-packages-locally}
@echo.
@echo Then launch a new command console for the changes to take effect and run packman command again.
@exit /B 1

:ERROR
@echo !!! Failure while configuring local machine :( !!!
@exit /B 1

:CLEAN_UP_TEMP_FOLDER
@rd /S /Q "%TEMP_FOLDER_NAME%"
@exit /B

:CREATE_PYTHON_BASE_DIR
:: We ignore errors and clean error state - if two processes create the directory one will fail which is fine
@md "%PM_PYTHON_BASE_DIR%" > nul 2>&1
@exit /B 0

:END
