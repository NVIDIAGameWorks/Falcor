
rem %1==projectDir %2==outputdir
setlocal

echo MOVEPROJECTDATA
echo "%1"
echo "%2"


IF not exist %2\Data\ ( mkdir %2\Data >nul )
IF exist %1\data\ ( xcopy %1\Data\*.* %2\Data /s /y /d /q >nul)

IF exist %1\ShadingUtils\ ( xcopy %1\ShadingUtils\*.* %2\Data /s /y /d)

rem deploy ray tracing data
IF exist %1\Raytracing\Data\ ( xcopy %1\Raytracing\Data\*.* %2\Data /s /y /d /q >nul)

rem deploy NVAPI
set NVAPI_DIR=%1\..\Externals\NVAPI
IF exist %NVAPI_DIR% (
    IF not exist %2\Data\NVAPI mkdir %2\Data\NVAPI >nul
    copy /y %NVAPI_DIR%\nvHLSLExtns.h %2\Data\NVAPI
    copy /y %NVAPI_DIR%\nvHLSLExtnsInternal.h %2\Data\NVAPI
    copy /y %NVAPI_DIR%\nvShaderExtnEnums.h %2\Data\NVAPI
)

rem deploy effects
