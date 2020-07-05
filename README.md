# Falcor 4.1

Falcor is a real-time rendering framework supporting DirectX 12. It aims to improve the productivity of research and prototype projects.

Features include:

- Abstracting many common graphics operations, such as shader compilation, model loading, and scene rendering
- DirectX Raytracing abstraction
- Render Graph system
- Python scripting
- Common rendering effects such as shadows and post-processing effects
- Unbiased path tracer

The included path tracer requires NVAPI. Please make sure you have it set up properly, otherwise, the path tracer won't work. You can find the instructions below.

## Prerequisites

- Windows 10 version 1809 or newer
- Visual Studio 2019
- [Microsoft Windows SDK version 1903 (10.0.18362.1)](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive)

Optional:

- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX (make sure you have the latest driver)
- Windows 10 Graphics Tools. To run DirectX 12 applications with the debug layer enabled, you must install this. There are two ways to install it:
  - Click the Windows button and type `Optional Features`, in the window that opens click `Add a feature` and select `Graphics Tools`.
  - Download an offline package from [here](https://docs.microsoft.com/en-us/windows-hardware/test/hlk/windows-hardware-lab-kit#supplemental-content-for-graphics-media-and-mean-time-between-failures-mtbf-tests). Choose a ZIP file that matches the OS version you are using (not the SDK version used for building Falcor). The ZIP includes a document which explains how to install the graphics tools.
- NVAPI (see below)

## NVAPI installation

After cloning the repository, head over to https://developer.nvidia.com/nvapi and download the latest version of NVAPI (this build is tested against version R435).
Extract the content of the zip file into `Source/Externals/.packman/` and rename `R435-developer` to `NVAPI`.

Finally, set `_ENABLE_NVAPI` to `true` in `Source/Falcor/Core/FalcorConfig.h`

## CUDA Support

Refer to the README located in the `Source/Samples/CudaInterop/` for instructions on how to set up your environment to use CUDA with Falcor.

## Falcor Configuration

`FalcorConfig.h` contains some flags which control Falcor's behavior.

- `_LOG_ENABLED` - Enable/disable log messages. By default, it is set to `true`.
- `_PROFILING_ENABLED` - Enable/Disable the internal CPU/GPU profiler. By default, it is set to `true`.

## Resources

- [Falcor](https://github.com/NVIDIAGameWorks/Falcor): Falcor's GitHub page.
- [Documentation](./Docs/index.md): Additional information and tutorials.
  - [Getting Started](./Docs/Usage/Getting-Started.md)
  - [Render Graph Tutorials](./Docs/Tutorials/index.md)
- [ORCA](https://developer.nvidia.com/orca): A collection of high-quality scenes and assets optimized for Falcor.
- [Slang](https://github.com/shader-slang/slang): Falcor's shading language and compiler.

## Citation

If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

```bibtex
@Misc{Benty20,
   author =      {Nir Benty and Kai-Hwa Yao and Petrik Clarberg and Lucy Chen and Simon Kallweit and Tim Foley and Matthew Oakes and Conor Lavelle and Chris Wyman},
   title =       {The {Falcor} Rendering Framework},
   year =        {2020},
   month =       {03},
   url =         {https://github.com/NVIDIAGameWorks/Falcor},
   note=         {\url{https://github.com/NVIDIAGameWorks/Falcor}}
}
```
