REM Check In Test D3D12 - Runs the Release version of the D3D12 Tests, locally.
call python.exe RunTestsSet.py -nb -d ../ -sln Falcor.sln -cfg ReleaseD3D12 -ts TestsCollectionsAndSets/TestsSetD3D12.json
