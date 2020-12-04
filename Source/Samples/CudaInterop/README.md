# Using CUDA with Falcor

Download and run the CUDA Toolkit installer from [here](https://developer.nvidia.com/cuda-10.1-download-archive-update1). Navigate to the location the CUDA Toolkit was installed to (`C:\Program Files\NVIDIA GPU Computing Tools\CUDA` by default). Copy the `v.10.1` folder into `Source/Externals/.packman` and rename it `cuda`.

The following will show how to create a CUDA project for use with Falcor:
1. Create a new CUDA Runtime project and add it to the Falcor solution.
2. In the Solution Explorer, right-click on `References` under the project and select `Add Reference`, then add `Falcor`.
3. Right-click on the project and go to `Build Dependencies -> Build Customizations`. Select `Find Existing`, and select `Source/Externals/.packman/cuda/extras/visual_studio_integration/MSBuildExtensions/CUDA 10.1.targets`.
4. Open the Property Manager and add the `Falcor` and `FalcorCUDA` property sheets to both Debug and Release. These are located in `Source/Falcor` and `Source/Samples/CudaInterop`, respectively.
5. Open the project's properties and go to `CUDA/C++` and set `CUDA Toolkit Custom Dir` to `$(SolutionDir)Source\Externals\.packman\cuda`, then go to `Linker -> System` and change the `SubSystem` to Windows.

The CudaInterop sample is tested on CUDA Toolkit v.10.1.168.
