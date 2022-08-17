### [Index](./index.md) | Getting Started

--------

# Getting Started

## Project Layout

### Falcor
The `Source/Falcor` folder contains the Falcor core framework. It is built as a shared library.

### Samples
The `Source/Samples` folder contains the Falcor sample applications. Each application is using Falcor directly and demonstrates how to use some of the fundamental features and abstractions Falcor provides.

### Mogwai
The `Source/Mogwai` folder contains the Mogwai application. It's the main application for using render graphs and provides some useful utilities. Some sample render graphs are located under its project folder: `Source/Mogwai/Data/`.

### RenderPasses
The `Source/RenderPasses` folder contains a number of components (shared libraries) that are used as the building blocks for creating render graphs. All render pass libraries are automatically built as dependencies of the `Mogwai` application.

-----------------------
## Workflows
There are two main workflows when using Falcor:

### Render Graphs
The recommended workflow when prototyping or implementing rendering techniques is to create render passes, render graphs, then render them with Mogwai. The [tutorials](./Tutorials/index.md) focus on this workflow.

#### To run a sample Render Graph:
1. Build Falcor
2. Run `Mogwai`
3. Press `Ctrl+O`, or from the top menu bar, select `File` -> `Load Script`
4. Select a Render Graph (.py file) in `Source/Mogwai/Data/`. Such as `ForwardRenderer.py`.
5. Press `Ctrl+Shift+O`, or from the top menu bar, select `File` -> `Load Scene`.
6. Select a scene or model, such as `media/Arcade/Arcade.pyscene`

Scenes and Render Graphs can also be loaded through drag and drop.

#### To create a Render Pass Library:
Run `tools/make_new_render_pass_library.bat <LibraryName>` to create a new render pass library.

### Sample Applications
In some cases, users may still prefer to create an application using Falcor directly. The `Renderer` class is the bootstrapper for Falcor applications. You should inherit from it and override its protected methods which serve as the callback functions. A good place to start looking for examples would be the `ModelViewer` sample.

#### To create a new Sample Application:
Run `tools/make_new_sample.bat <ProjectName>` to create a new sample application.

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

The best practice is to create a directory called `Data` next to your **project** file and place all your data files there. Your shader files should also have a `.slang`, `.slangh`, `.hlsl`, or `.hlsli` extension. Headers with a `.h` should be used for host-only files. Headers that will be shared between host and shader files should use the `.slang` or `.slangh` extension.

To search for a data or shader file, call `findFileInDataDirectories()` or `findFileInShaderDirectories()` respectively.

Falcor uses the [Slang](https://github.com/shader-slang/slang) shading language and compiler.
Users can write HLSL/Slang shader code in `.hlsl` or `.slang` files.
