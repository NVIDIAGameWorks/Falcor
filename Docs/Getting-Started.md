### [Index](./index.md) | Getting Started

--------

# Getting Started

## Solution and Project Layout

### Falcor
All core framework features and code are contained within the `Falcor` project in the solution. This is a DLL project and is not executable on its own. 

### Samples
Projects in the `Samples` folder (`HelloDXR`, `ModelViewer`, and `ShaderToy`) are applications using Falcor directly that demonstrate how to use some of the fundamental features and abstractions Falcor provides.

### Mogwai
The `Mogwai` project is an application created using Falcor that simplifies using render graphs and provides some useful utilities. Some sample render graphs are located under its project folder: `Source/Mogwai/Data/`.

### RenderPasses
The `RenderPasses` folder contains a number of components ("Render Passes") that can be assembled to create render graphs. These components are required for some of the included sample render graphs, but will not build unless you select them manually (Right-Click -> `Build`), or use `Build Solution`. 

-----------------------
## Workflows

There are two main workflows when using Falcor:

### Render Graphs
The recommended workflow when prototyping or implementing rendering techniques is to create render passes, render graphs, then render them with Mogwai. The [tutorials](./Tutorials/index.md) focus on this workflow.

#### To run a sample Render Graph:
1. Build the Falcor Solution
2. Run `Mogwai`
3. Press `Ctrl+O`, or from the top menu bar, select `File` -> `Load Script`
4. Select a Render Graph (.py file) in `Source/Mogwai/Data/`. Such as `ForwardRenderer.py`.
5. Press `Ctrl+Shift+O`, or from the top menu bar, select `File` -> `Load Scene`.
6. Select a scene or model, such as `Media/Arcade/Arcade.fscene`

Scenes and Render Graphs can also be loaded through drag and drop.

#### To create a Render Pass Project:
1. Navigate to `Source/RenderPasses/`.
2. Run `make_new_pass_project.bat <RenderPassName>` to create a new project in the folder.
3. Add the new project to the Visual Studio solution.

### Falcor Applications

In some cases, users may still prefer to create an application using Falcor directly. The `Renderer` class is the bootstrapper for Falcor applications. You should inherit from it and override its protected methods which serve as the callback functions. A good place to start looking for examples would be the `ModelViewer` sample.

#### To create a new Falcor project:
1. Navigate to `Source/Samples/`.
2. Run `make_new_project.bat <ProjectName>` to create a new project in the folder.
3. Add the new project to the Visual Studio solution.

-----------------------

## Using Shaders and Data Files
Falcor searches through multiple working directories for files specified by relative paths.

*Data* files are non-shader resources such as textures and models.

When running from Visual Studio:
- Falcor looks for data files in the following locations:
    - A subfolder named `Data` inside the project folder (the directory containing the Visual Studio project file).
    - A subfolder named `Data` inside the executable directory.
    - A optional environment variable named `FALCOR_MEDIA_FOLDERS` containing a semicolon-separated list of folders.
    - Any directory that was added to the data directories list by calling `addDataDirectory()`.
- Falcor looks for Shader files relative to your project folder.

Upon building, a project's `Data` folder and shader files will be automatically deployed to the `Data` and `Shaders` folders in the executable directory while preserving folder hierarchy. When running an application from its executable, Falcor will search in these folders instead. This allows the build output folder to be self-contained for easy sharing.

The best practice is to create a directory called `Data` next to your **project** file and place all your data files there. Your shader files should also have a `.slang`, `.slangh`, `.hlsl`, or `.hlsli` extension. Files with these extensions will be marked with the `Shader Source` item type in Visual Studio, and only these files will be deployed. Headers with a `.h` should be used for host-only files. Headers that will be shared between host and shader files should use the `.slang` or `.slangh` extension.

To search for a data or shader file, call `findFileInDataDirectories()` or `findFileInShaderDirectories()` respectively.

Falcor uses the [Slang](https://github.com/shader-slang/slang) shading language and compiler.
Users can write HLSL/Slang shader code in `.hlsl` or `.slang` files.
