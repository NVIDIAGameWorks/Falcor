### [Index](../index.md) | [Development](./index.md) | CMake

--------

# CMake

Falcor uses the [CMake](https://cmake.org) build system. A good introduction to using CMake can be found at [An Introduction to Modern CMake](https://cliutils.gitlab.io/modern-cmake).

## Adding source files to existing executables/libraries

To add new source or shader files to existing executable/library targets, simply list the files in the `target_sources` call in the respective `CMakeLists.txt` file. Try to keep the file list sorted alphabetically.

Shader files will automatically be recognized by their file extension and get copied to the output folder during build (assuming there is a call to `target_copy_shaders` in the respective `CMakeLists.txt`).

Note: After changing a `CMakeLists.txt` file, the next build will detect that change and re-run the CMake generator. When working with Visual Studio solutions generated form CMake, this will result in a dialog allowing the solution to be reloaded after a build to reflect the changes made to the solution file.

## Creating a new sample application

Run `tools/make_new_sample.bat <SampleName>` to create a new sample application and adjust the build files.

## Creating a new render pass

Run `tools/make_new_render_pass_library.bat <LibraryName>` to create a new render pass library and adjust the build files.

## Falcor specific CMake functions

There are a few Falcor specific functions available in the CMake scripts:

**`target_copy_shaders(<name> <output_dir>)`**

Setup build rules to copy all shaders of a `target` to the output directory.
The specified `output_dir` is relative to the global shader output directory (`FALCOR_SHADER_OUTPUT_DIRECTORY`).

**`target_copy_data_folder(<target>)`**

Setup a post-build rule to copy the data folder of a `target` to the output directory.

**`add_falcor_executable(<target>)`**

Create a Falcor application and link the main Falcor library.

**`add_renderpass(<target>)`**

Create a Falcor renderpass.

**`target_source_group(<target> <folder>)`**

Helper function to create a source group for Visual Studio.
This adds all the target's sources to a source group in the given folder.
