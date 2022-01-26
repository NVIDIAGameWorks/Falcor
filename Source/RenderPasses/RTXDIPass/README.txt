This directory contains an example of using the RTXDI integration built into Falcor.

Note:  RTXDI is *not* distributed with (public) Falcor releases, so users desiring RTXDI
need to register for access to the SDK code on Github on the NVIDIA Developer site:
    https://developer.nvidia.com/rtxdi

Once registered, you should be able to get the SDK from this Github repository:
    https://github.com/NVIDIAGameWorks/RTXDI

See the main README for installation details.

A Simple Example Using RTXDI:
-----------------------------

A sample render script that configures Falcor to use this render pass is in:
   `Source\Mogwai\Data\RTXDI.py`
or, after building the solution, it is also available as:
   `Bin\x64\Release\Data\RTXDI.py`

To test this simple example:

1) Make sure you open the `Falcor.sln` Visual Studio project and build the *entire* solution,
preferrably in release mode (aka `ReleaseD3D12` in the Visual Studio GUI).

2) Run Mogwai.exe, which allows you to load Falcor scenes and render scripts.

3) Load the RTXDI.py render script from one of the locations listed above (you can do this
via the Mogwai command line, from the menu after running Mogwai, or dragging & dropping the
.py file into a running Mogwai window).

4) Load an appropriate scene. After building, Visual Studio will have pulled a default scene
as part of the package dependencies it downloads. You can use this scene, located in:
    `Media\Arcade\Arcade.pyscene`
You can (also) load this scene via the Mogwai command line, from the menu after running Mogwai,
or by dragging and dropping the `.pyscene` into a running Mogwai window.
