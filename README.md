Falcor 2.0 beta
=================

Falcor is a real-time rendering framework supporting DirectX 12 and Vulkan. It aims to improve productivity of research and prototype projects.
Its features include:
* Abstract many common graphics operations, such as shader compilation, model loading and scene rendering.
* VR support using OpenVR.
* Common rendering effects such as shadows and post-processing effects.

This is a beta version. The interfaces are not final yet and there might be some performance/stability issues.

Prerequisites
------------------------
- Visual Studio 2015
- [Microsoft Windows SDK ver 10.0.14393.795](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive)
- Windows 10
- GPU that supports DirectX 12 or Vulkan

NVAPI Support
--------------
NVIDIA's NVAPI SDK exposes a set of GPU features that are not part of the DirectX spec.
Using it with Falcor is not mandatory. However, Falcor does abstract some of those features. For example, the SceneRenderer VR mode relies on Single Pass Stereo support.
If you want to use it:
- Please download the [NVAPI SDK](https://developer.nvidia.com/nvapi)
- Unzip the content of the package to Framework\Externals
- Rename the folder to 'NVAPI'

Building Falcor
---------------
Open `Falcor.sln` and it should build successfully in Visual Studio out of the box. If you wish to skip this step and add Falcor to your own Visual Studio solution directly,
run `update_dependencies.bat` to download the required dependencies then follow the instructions below. Otherwise, this will happen automatically when building the Falcor solution.

Creating a New Project
------------------------
- If you haven't done so already, create a Visual Studio solution and project for your code. Falcor only supports 64-bit builds, so make sure you have a 64-bit build configuration.
- Add Falcor.props to your project (Property Manager -> Right click your project -> Add existing property sheet).
- Add Falcor.vcxproj to your solution.
- Add a reference to Falcor in your project (Solution Explorer -> Right click your project -> Properties -> Common Properties -> References -> Add new reference -> Choose Falcor).

*Sample* Class
-------------------
This is the bootstrapper class of the application. Your class should inherit from it and override its protected methods which serve as the callback functions.
A good place to start looking for examples would be the ModelViewer sample.

Build Configurations
--------------------
Falcor has the following build configurations for DirectX 12 and Vulkan, respectively:
- `DebugD3D12`
- `ReleaseD3D12`
- `DebugVK`
- `ReleaseVK`

Debug builds enable file logging and message boxes by default, and there is a lot of runtime error checking. If debug layers for the selected API are installed, they will be loaded as well.

Release builds disable logging and most runtime error checks. Use this configuration to measure performance.

Setting Up Debug Layers
------------------------
To use DirectX 12 debug layers:
- Open the Start menu
- Type "Manage optional features" and press Enter
- Click "Add a feature"
- Install "Graphics Tools"

To use Vulkan debug layers:
- Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

Falcor Configuration
--------------------
`FalcorConfig.h` contains some flags which control Falcor's behavior.
- `_LOG_ENABLED` - Enable/disable log messages. By default, it is set to `false` for release build and `true` for debug builds.
- `_PROFILING_ENABLED` - Enable/Disable the internal CPU/GPU profiler. By default, it is set to `true`.

Data Files
--------------------
Data files include shader files, textures, and models.
By default, Falcor looks for data files in the following locations:
- The working directory. In some cases this is not the same as the executable directory. For example, if you launch the application from Visual Studio, by default the working directory is the directory containing the project file.
- The executable directory.
- An optional environment variable named `FALCOR_MEDIA_FOLDERS`. It is a semicolon-separated list of folders.
- Any directory that was added to the data directories list by calling `addDataDirectory()`.
- A directory called "Data/" under any of the above directories.

To search for a data file, call `findFileInDataDirectories()`.

Deployment
----------
The best practice is to create a directory called "Data/" next to your **project** file and place all your data files there (shaders/models).  If that directory exists, Falcor will copy it to the output directory, making the output directory self-contained (you can zip only the output directory and it should work).  If not, you will have to copy the data files yourself.

Citation
--------
If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

@Misc{Benty17,  
   author =      {Nir Benty and Kai-Hwa Yao and Tim Foley and Anton S. Kaplanyan and Conor Lavelle and Chris Wyman and Ashwin Vijay},  
   title =       {The {Falcor} Rendering Framework},  
   year =        {2017},  
   month =       {07},  
   url =         {https://github.com/NVIDIAGameWorks/Falcor},  
   note=         {\url{https://github.com/NVIDIAGameWorks/Falcor}}  
}
