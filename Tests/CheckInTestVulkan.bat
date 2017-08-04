REM Check In Test Vulkan - Runs the Release version of the Vulkan Tests, locally.
call python.exe RunTestsSet.py -d ../ -sln Falcor.sln -cfg ReleaseVK -ts TestsCollectionsAndSets/TestsSetD3D12.json
