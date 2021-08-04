# Falcor 4.4

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
- Windows 10 version 20H2 (October 2020 Update) or newer
- Visual Studio 2019
- [Windows 10 SDK (10.0.19041.0) for Windows 10, version 2004](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX (make sure you have the latest driver)

Optional:
- Windows 10 Graphics Tools. To run DirectX 12 applications with the debug layer enabled, you must install this. There are two ways to install it:
    - Click the Windows button and type `Optional Features`, in the window that opens click `Add a feature` and select `Graphics Tools`.
    - Download an offline package from [here](https://docs.microsoft.com/en-us/windows-hardware/test/hlk/windows-hardware-lab-kit#supplemental-content-for-graphics-media-and-mean-time-between-failures-mtbf-tests). Choose a ZIP file that matches the OS version you are using (not the SDK version used for building Falcor). The ZIP includes a document which explains how to install the graphics tools.
- NVAPI (see below)

## NVAPI installation
After cloning the repository, head over to https://developer.nvidia.com/nvapi and download the latest version of NVAPI (this build is tested against version R440).
Extract the content of the zip file into `Source/Externals/.packman/` and rename `R470-developer` to `nvapi`.

Finally, set `_ENABLE_NVAPI` to `1` in `Source/Falcor/Core/FalcorConfig.h`

## CUDA Support
If you want to use CUDA C/C++ code as part of a Falcor project, then refer to the README located in the `Source/Samples/CudaInterop/` for instructions on how to set up your environment to use CUDA with Falcor.

If you want to execute Slang-based shader code through CUDA using `CUDAProgram`, then you will need to copy or link the root directory of the CUDA SDK under `Source/Externals/.packman/`, as a directory named `CUDA`.
Then, set `_ENABLE_CUDA` to `1` in `Source/Falcor/Core/FalcorConfig.h`

## OptiX Support
If you want to use Falcor's OptiX functionality (specifically the `OptiXDenoiser` render pass), then refer to the README location in `Source/Samples/OptixDenoiser` for instructions on setting up your environment to use OptiX with Falcor.

In particular, you will need to copy or link the root directory of the OptiX SDK under `Source/Externals/`, as a directory named `optix` (i.e., `Source/Externals/optix/include/optix.h` should exist).
Then, set `_ENABLE_OPTIX` to `1` in `Source/Falcor/Core/FalcorConfig.h`

## Falcor Configuration
`FalcorConfig.h` contains some flags which control Falcor's behavior.
- `_LOG_ENABLED` - Enable/disable log messages. By default, it is set to `1`.
- `_PROFILING_ENABLED` - Enable/Disable the internal CPU/GPU profiler. By default, it is set to `1`.

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
