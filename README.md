Falcor 4.0 Development Snapshot
===============================

Falcor is a real-time rendering framework supporting DirectX 12 and Vulkan. It aims to improve productivity of research and prototype projects.
Its features include:
* Abstracting many common graphics operations, such as shader compilation, model loading and scene rendering
* VR support using OpenVR
* Common rendering effects such as shadows and post-processing effects
* DirectX Raytracing abstraction 

Note that Falcor 4.0 is still under development. There will be more changes to the interfaces as well as new features. This is a snapshot of our development branch and doesn't represent an official version (not even alpha).
This release only supports DX12 on Windows.
The path tracer requires NVAPI. Please make sure you have it setup properly, otherwise the path-tracer won't work. You can find the instructions below.


Prerequisites
------------------------
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX (make sure you have the latest driver)
- Windows 10 RS5 (version 1809) or newer
- NVAPI

On Windows:
- Visual Studio 2017 15.9.10 (it might not compile with older VS versions)
- [Microsoft Windows SDK version 1809 (10.0.17763.0)](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive)
- To run DirectX 12 applications with the debug layer enabled, you need to install the Graphics Tools optional feature. The tools version must match the OS version you are using (not to be confused with the SDK version used for building Falcor). There are 2 ways to install it:
    - Click the Windows button and type `Optional Features`, in the window that opens click `Add a feature` and select `Graphics Tools`.
    - Download an offline pacakge from [here](https://docs.microsoft.com/en-us/windows-hardware/test/hlk/windows-hardware-lab-kit#supplemental-content-for-graphics-media-and-mean-time-between-failures-mtbf-tests). Choose a ZIP file that matches the OS version you are using. The ZIP includes a document which explains how to install the graphics tools.

NVAPI installation
------------------
After cloning the repository, head over to https://developer.nvidia.com/nvapi and download the latest version of NVAPI (this build is tested against version R435).
1. Extract the content of the zip file into `<FalcorRootDir>\Source\Externals\.packman`. If you have NVAPI version R435, you should have the `<FalcorRootDir>\Source\Externals\.packman\R435-developer` folder.
2. Rename `R435-developer` to `NVAPI`. You should end up with the `<FalcorRootDir>\Source\Externals\.packman\NVAPI` folder.

Building Falcor
---------------
Open `Falcor.sln` and it should build successfully in Visual Studio out of the box. If you wish to skip this step and add Falcor to your own Visual Studio solution directly,
follow the instructions below.

Creating a New Project
------------------------
- If you haven't done so already, create a Visual Studio solution and project for your code. Falcor only supports 64-bit builds, so make sure you have a 64-bit build configuration
- Add `Falcor.props` to your project (Property Manager -> Right click your project -> Add existing property sheet)
- Add `Falcor.vcxproj` to your solution (Located at `Framework/Source/Falcor.vcxproj`)
- Add a reference to Falcor in your project (Solution Explorer -> Your Project -> Right Click `References` -> Click `Add Reference...` -> Choose Falcor)

*Sample* Class
-------------------
This is the bootstrapper class of the application. Your class should inherit from it and override its protected methods which serve as the callback functions.
A good place to start looking for examples would be the ModelViewer sample.

Build Configurations
--------------------
Falcor has the following build configurations for DirectX 12, Vulkan and DXR, respectively:
- `DebugD3D12`
- `ReleaseD3D12`
- Currently, the `DebugVK` and `ReleaseVK` build is failing

Debug builds enable file logging and message boxes by default, and there is a lot of runtime error checking. If debug layers for the selected API are installed, they will be loaded as well.

Release builds disable logging and most runtime error checks. Use this configuration to measure performance.

Setting Up Debug Layers
------------------------
To use the DirectX 12 debug layer:
- Open the Start menu
- Type "Manage optional features" and press Enter
- Click "Add a feature"
- Install "Graphics Tools"

To use Vulkan debug layers:
- Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

Falcor Configuration
--------------------
`FalcorConfig.h` contains some flags which control Falcor's behavior.
- `_LOG_ENABLED` - Enable/disable log messages. By default, it is set to `false` for release build and `true` for debug builds
- `_PROFILING_ENABLED` - Enable/Disable the internal CPU/GPU profiler. By default, it is set to `true`

Data Files
--------------------
Data files include shader files, textures, and models.
By default, Falcor looks for data files in the following locations:
- The working directory. In some cases this is not the same as the executable directory. For example, if you launch the application from Visual Studio, by default the working directory is the directory containing the project file
- The executable directory
- An optional environment variable named `FALCOR_MEDIA_FOLDERS`. It is a semicolon-separated list of folders
- Any directory that was added to the data directories list by calling `addDataDirectory()`
- A directory called "Data/" under any of the above directories

To search for a data file, call `findFileInDataDirectories()`.

Shaders
-------

Falcor uses the [Slang](https://github.com/shader-slang/slang) shading language and compiler.
Users can write HLSL/Slang shader code in `.hlsl` or `.slang` files.
The framework handles cross-compilation to SPIR-V for you when targetting Vulkan; GLSL shaders are not supported.

Deployment
----------
The best practice is to create a directory called "Data/" next to your **project** file and place all your data files there (shaders/models).  If that directory exists, Falcor will copy it to the output directory, making the output directory self-contained (you can zip only the output directory and it should work).  If not, you will have to copy the data files yourself.

Citation
--------
If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

```bibtex
@Misc{Benty19,  
   author =      {Nir Benty and Kai-Hwa Yao and Lucy Chen and Tim Foley and Matthew Oakes and Conor Lavelle and Chris Wyman},  
   title =       {The {Falcor} Rendering Framework},  
   year =        {2019},  
   month =       {10},  
   url =         {https://github.com/NVIDIAGameWorks/Falcor},  
   note=         {\url{https://github.com/NVIDIAGameWorks/Falcor}}  
}
```

