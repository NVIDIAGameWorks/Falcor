rem %1==dirToDelete
ECHO Cleaning %1
@echo off

IF EXIST %1 (
pushd %1
attrib /s /d -r > nul
rem set tlogs to read only so they wont be deleted
attrib /s /d *.tlog +r > nul
del. /s /q > nul
attrib /s /d -r > nul
popd
)