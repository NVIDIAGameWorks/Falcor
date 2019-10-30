
rem %1==projectDir %2==outputdir
setlocal

IF not exist %2\Data\ ( mkdir %2\Data >nul )
IF exist %1\data\ ( xcopy %1\Data\*.* %2\Data /s /y /d /q >nul)

