# Falcor 5.0-preview

Falcor is a real-time rendering framework supporting DirectX 12. It aims to improve productivity of research and prototype projects.

Features include:
* Abstracting many common graphics operations, such as shader compilation, model loading, and scene rendering
* DirectX Raytracing abstraction
* Render Graph system
* Python scripting
* Common rendering effects such as shadows and post-processing effects
* Unbiased path tracer

The included path tracer requires NVAPI. Please make sure you have it set up properly, otherwise the path tracer won't work. You can find the instructions below.

## Prerequisites
- Windows 10 version 20H2 (October 2020 Update) or newer, OS build revision .789 or newer
- Visual Studio 2019
- [Windows 10 SDK (10.0.19041.0) for Windows 10, version 2004](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX
- NVIDIA driver 466.11 or newer

Optional:
- Windows 10 Graphics Tools. To run DirectX 12 applications with the debug layer enabled, you must install this. There are two ways to install it:
    - Click the Windows button and type `Optional Features`, in the window that opens click `Add a feature` and select `Graphics Tools`.
    - Download an offline package from [here](https://docs.microsoft.com/en-us/windows-hardware/test/hlk/windows-hardware-lab-kit#supplemental-content-for-graphics-media-and-mean-time-between-failures-mtbf-tests). Choose a ZIP file that matches the OS version you are using (not the SDK version used for building Falcor). The ZIP includes a document which explains how to install the graphics tools.
- NVAPI (see below)

## Microsoft DirectX 12 Agility SDK

Falcor uses the [Microsoft DirectX 12 Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/) to get access to the latest DirectX 12 features. Applications can enable the Agility SDK by putting `FALCOR_EXPORT_D3D12_AGILITY_SDK` in the main `.cpp` file. `Mogwai`, `FalcorTest` and `RenderGraphEditor` have the Agility SDK enabled by default.

## NVAPI
To enable NVAPI support, head over to https://developer.nvidia.com/nvapi and download the latest version of NVAPI (this build is tested against version R470).
Extract the content of the zip file into `Source/Externals/.packman/` and rename `R470-developer` to `nvapi`.

Finally, set `FALCOR_ENABLE_NVAPI` to `1` in `Source/Falcor/Core/FalcorConfig.h`

## CUDA
To enable CUDA support, download [CUDA 11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive). After running the installer, navigate to the CUDA installation (`C:\Program Files\NVIDIA GPU Computing Tools\CUDA` by default). Link or copy the `v.11.3` folder into `Source/Externals/.packman/cuda`.

Finally, set `FALCOR_ENABLE_CUDA` to `1` in `Source/Falcor/Core/FalcorConfig.h`

To run the `CudaInterop` sample application located in `Source/Samples/CudaInterop`, you first have to add it to the solution (it's not added by default to avoid errors when opening the solution without CUDA installed).

To create a new CUDA enabled Falcor application, follow these steps:
1. Create a new CUDA Runtime project and add it to the Falcor solution.
2. In the Solution Explorer, right-click on `References` under the project and select `Add Reference`, then add `Falcor`.
4. Open the Property Manager and add the `Falcor` and `FalcorCUDA` property sheets to both Debug and Release. These are located in `Source/Falcor`.
5. Open the project's properties and go to `CUDA/C++` and set `CUDA Toolkit Custom Dir` to `$(SolutionDir)Source\Externals\.packman\cuda`, then go to `Linker -> System` and change the `SubSystem` to Windows.

## OptiX
If you want to use Falcor's OptiX functionality (specifically the `OptixDenoiser` render pass) download the [OptiX SDK](https://developer.nvidia.com/designworks/optix/download) (Falcor is currently tested against OptiX version 7.3) After running the installer, link or copy the OptiX SDK folder into `Source\Externals\.packman\optix` (i.e., file `Source\Externals\.packman\optix\include\optix.h` should exist).

Finally, set `FALCOR_ENABLE_OPTIX` to `1` in `Source/Falcor/Core/FalcorConfig.h`

Note: You also need CUDA installed to compile the `OptixDenoiser` render pass, see above for details.

## Falcor Configuration
`FalcorConfig.h` contains some flags which control Falcor's behavior.
- `FALCOR_ENABLE_LOGGER` - Enable/disable the logger. By default, it is set to `1`.
- `FALCOR_ENABLE_PROFILER` - Enable/disable the internal CPU/GPU profiler. By default, it is set to `1`.

## Resources
- [Falcor](https://github.com/NVIDIAGameWorks/Falcor): Falcor's GitHub page.
- [Documentation](./Docs/index.md): Additional information and tutorials.
    - [Getting Started](./Docs/Getting-Started.md)
    - [Render Graph Tutorials](./Docs/Tutorials/index.md)
- [ORCA](https://developer.nvidia.com/orca): A collection of high quality scenes and assets optimized for Falcor.
- [Slang](https://github.com/shader-slang/slang): Falcor's shading language and compiler.

## Citation
If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

```bibtex
@Misc{Kallweit21,
   author =      {Simon Kallweit and Petrik Clarberg and Craig Kolb and Kai-Hwa Yao and Theresa Foley and Lifan Wu and Lucy Chen and Tomas Akenine-Moller and Chris Wyman and Cyril Crassin and Nir Benty},
   title =       {The {Falcor} Rendering Framework},
   year =        {2021},
   month =       {08},
   url =         {https://github.com/NVIDIAGameWorks/Falcor},
   note =        {\url{https://github.com/NVIDIAGameWorks/Falcor}}
}
```
