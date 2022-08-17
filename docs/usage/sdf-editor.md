### [Index](../index.md) | [Usage](./index.md) | SDF Editor

--------

# SDF Editor

## Getting Started
1. Build Falcor.
2. Launch Mogwai.
3. Load the script `Source\RenderPasses\SDFEditor\Data\SDFEditor.py`. This uses the `PathTracer`, `AccumulatePass`, `ToneMapper`, and there is an `SDFEditor` pass at the end.
4. Load the scene `Source\RenderPasses\SDFEditor\Data\SDFEditorStartScene.pyscene`. You should see a single sphere in the center of the windows. Now you can start using the SDF editor. Pull down the GUI for the `SDFEditor` `RenderPass`. Hoover with the mouse over the tooltop, i.e., the "(?)" at the top to get instructions on how to use the editor.
    - `ALT` lets you add/subtract geometry while pressing the left mouse button.
    - `TAB` brings up the GUI for selecting which primitive and which primitive operation.

### File formats
There are two types of SDF file formats that Falcor currently supports:
- `.sdf`: That stores a list of 'edits' as a text file, and
    - Note that `SDFEditorStartScene.pyscene` (see Getting Started) loads the `single_sphere.sdf`, which contains just a single sphere.
    - You can change so that it loads `test_primitives.sdf` instead to see other primitives.
- `.sdfg`: That stores the signed distance field as a binary file.

However, the SDF editor only supports loading the `.sdf` format, but can save as a `.sdfg` file (this is likely changing).

## The SDF Editor RenderPass

The SDF Editor is implemented as a render pass, the inputs and outputs are as follows:
| Name | Type | Description |
| --- | --- | --- |
| `vbuffer` | Input | Encodes the primary hit points. |
| `linearZ` | Input | Linear depth. |
| `inputColor` | Input | Color texture from a previous render pass. |
| `output` | Output | Final color texture. |

A simple render script with the SDF Editor using the path tracer can be seen in `Source/RenderPasses/SDFEditor/Data/SDFEditor.py`.

To begin editing with the SDF editor, you need to have at least one SDF grid instance of the type SBS in the scene. The SDF grid is created by calling `SDFGrid.createSBS()`. It takes two optional arguments. One for the brick width (`brickWidth`), which is the number of voxels it should pack into one AABB to be ray traced, the other argument is if it should compress the data or not (`compressed`). The create function returns a SBS grid object that can be added to the scene by calling `addSDFGrid()` from the SceneBuilder.

If you do not want to start from scratch you can load a SDF file from memory by using the function `loadPrimitivesFromFile` from the sdf grid object. This takes in the path to the file (`path`) and the grid width (`gridWidth`). The grid width is the desired resolution of the grid (the internal implementation might increase the grid width if it does not align with the brick width).

## Data structure creation in pyscene-files
The typical setup is:
```
sdfGrid = SDFGrid.createSVS()
sdfGrid.loadValuesFromFile(path="test_primitives.sdf")
sceneBuilder.addSDFGridInstance(
    sceneBuilder.addNode('SDFGrid', Transform(translation=float3(0, 0.6, 0))),
    sceneBuilder.addSDFGrid(sdfGrid, sdfGridMaterial)
)
```
The supported grid data structures are:
- Normalized dense grid (NDG, which is a full mip hierarchy): `sdfGrid = SDFGrid.createNDGrid(narrowBandThickness=2.5)` (traversal is done on the SM)
- Sparse voxel octree (SVO): `sdfGrid = SDFGrid.createSVO()` (traversal is done on the SM)
- Sparse voxel set (SVS): `sdfGrid = SDFGrid.createSVS()` (uses the TTU, often the fastest method, but uses a lot of memory)
- Sparse brick set (SBS): `sdfGrid = SDFGrid.createSBS()` (This is the **best** tradeoff in terms of performance and memory usage. Combines the TTU with traversal on the SM.)

