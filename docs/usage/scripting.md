### [Index](../index.md) | [Usage](./index.md) | Scripting

--------

# Mogwai Scripting Reference

## Sample API

#### Global functions

| Function                         | Description                                                              |
|----------------------------------|--------------------------------------------------------------------------|
| `loadRenderPassLibrary(name)`    | Load a render pass library.                                              |
| `createPass(name, dict)`         | Create a new render pass with configuration options in `dict`.           |
| `renderFrame()`                  | Render a frame. If the clock is not paused, it will advance by one tick. |
| `setWindowPos(x, y)`             | Set the window's position in pixels.                                     |
| `resizeSwapChain(width, height)` | Resize the window/swapchain.                                             |
| `exit(errorCode=0)`              | Terminate the application with the given error code.                     |
| `cls()`                          | Clear the console.                                                       |

## Mogwai API

#### Global functions

| Function       | Description                      |
|----------------|----------------------------------|
| `help()`       | Print global help.               |
| `help(object)` | Print help of a specific object. |

#### Global variables

| Variable | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `m`      | Instance of `Renderer`.                                                     |
| `t`      | Instance of `Clock`. **DEPRECATED**: Use `m.clock` instead.                 |
| `fc`     | Instance of `FrameCapture`. **DEPRECATED**: Use `m.frameCapture` instead.   |
| `vc`     | Instance of `VideoCapture`. **DEPRECATED**: Use `m.videoCapture` instead.   |
| `tc`     | Instance of `TimingCapture`. **DEPRECATED**: Use `m.timingCapture` instead. |

#### Renderer

class falcor.**Renderer**

| Property        | Type            | Description                     |
|-----------------|-----------------|---------------------------------|
| `scene`         | `Scene`         | Active scene (readonly).        |
| `activeGraph`   | `RenderGraph`   | Active render graph (readonly). |
| `ui`            | `bool`          | Show/hide the UI.               |
| `clock`         | `Clock`         | Clock.                          |
| `profiler`      | `Profiler`      | Profiler.                       |
| `frameCapture`  | `FrameCapture`  | Frame capture.                  |
| `videoCapture`  | `VideoCapture`  | Video capture.                  |
| `timingCapture` | `TimingCapture` | Timing capture.                 |

| Method                                                  | Description                                                     |
|---------------------------------------------------------|-----------------------------------------------------------------|
| `script(path)`                                          | Run a script.                                                   |
| `loadScene(path, buildFlags=SceneBuilderFlags.Default)` | Load a scene. See available build flags below.                  |
| `unloadScene()`                                         | Explicitly unload the scene to free memory.                     |
| `saveConfig(path)`                                      | Save the current state to a config file.                        |
| `addGraph(graph)`                                       | Add a render graph.                                             |
| `removeGraph(graph)`                                    | Remove a render graph. `graph` can be a render graph or a name. |
| `getGraph(name)`                                        | Get a render graph by name.                                     |
| `resizeSwapChain(width, height)`                        | Resize the window/swapchain.                                    |

#### Clock

class falcor.**Clock**

| Property    | Type    | Description                                                                     |
|-------------|---------|---------------------------------------------------------------------------------|
| `time`      | `float` | Current time in _seconds_.                                                      |
| `frame`     | `int`   | Current frame number.                                                           |
| `framerate` | `int`   | Frame rate in _frames per second_.                                              |
| `timeScale` | `float` | Time scale factor.                                                              |
| `exitTime`  | `float` | Time in _seconds_ at which to terminate the application. Set to `0` to disable. |
| `exitFrame` | `int`   | Frame at which to terminate the application. Set to `0` to disable.             |

| Method           | Description                       |
|------------------|-----------------------------------|
| `pause()`        | Pause the clock.                  |
| `play()`         | Resume the clock.                 |
| `stop()`         | Stop the clock (pause and reset). |
| `step(frames=1)` | Step forward or backward in time. |

#### Profiler

class falcor.**Profiler**

| Property      | Type   | Description                               |
|---------------|--------|-------------------------------------------|
| `enabled`     | `bool` | Enable/disable profiler.                  |
| `paused`      | `bool` | Pause/resume profiler.                    |
| `isCapturing` | `bool` | True if profiler is capturing (readonly). |
| `events`      | `dict` | Profiler events (readonly).               |

| Method           | Description                              |
|------------------|------------------------------------------|
| `startCapture()` | Start capturing.                         |
| `endCapture()`   | End capturing. Returns the capture data. |

##### Profiler event names

Profiler event names are expressed as paths due to their hierarchical nature. For example, `/present` is a top level event corresponding to the time it takes to present the final image, whereas `/onFrameRender/RenderGraphExe::execute()` is a nested event corresponding to the time it takes to execute the render graph. Because profiler events measure both CPU and GPU time, the paths are extended with either `/gpuTime` or `/cpuTime` to uniquely identify the actual measured quantity. As an example, `/present/gpuTime` refers to the GPU time it takes to present the final image.

##### Accessing profiler data

Profiler event data can be accessed through `m.profiler.events`. This returns a dictionary containing all event data of the last frame using the event name as a key. Each item itself is a dictionary containing the following keys/values:

| Key       | Value                                                        |
|-----------|--------------------------------------------------------------|
| `name`    | The name of the event (same as the key to access this item). |
| `value`   | Last measured value in _ms_.                                 |
| `average` | Exponential moving average in _ms_.                          |
| `stats`   | Dictionary containing stats from up to the last 512 frames.  |

The `stats` dictionary contains the following keys/values:

| Key      | Value                           |
|----------|---------------------------------|
| `mean`   | The mean value in _ms_.         |
| `stdDev` | The standard deviation in _ms_. |
| `min`    | The minimum value in _ms_.      |
| `max`    | The maximum value in _ms_.      |

To get the current present GPU time you can use `m.profiler.events["/present/gpuTime"]["value"]`. To get the mean from the last 512 frames you can use `m.profiler.events["/present/"gpuTime"]["stats"]["mean"]`.

##### Capturing profiler data

To capture profiler data from multiple frames, a separate capturing API can be used. A profile capture is started using `m.profiler.startCapture()`. Profile data is internally captured until a call to `m.profiler.endCapture()`, which will return a dictionary with all the captured data.

The capture dictionary contains the following keys/values:

| Key          | Value                                          |
|--------------|------------------------------------------------|
| `frameCount` | Total number of frames captured.               |
| `events`     | Dictionary containing the captured event data. |

The `events` dictionary uses event names as keys. Each item itself is a dictionary containing the following keys/values:

| Key       | Value                                                        |
|-----------|--------------------------------------------------------------|
| `name`    | The name of the event (same as the key to access this item). |
| `records` | Array containing the raw per frame data in _ms_.             |
| `stats`   | Dictionary containing stats from the raw frame data.         |

The `stats` dictionary has the same structure as explained above but is computed over the captured data instead of the last 512 frames.

The following snippet shows how to capture profiling data over 256 frames and print the mean GPU frame render time:

```python
m.profiler.enabled = True
m.profiler.startCapture()
for frame in range(256):
    m.renderFrame()
capture = m.profiler.endCapture()
m.profiler.enabled = False

meanFrameTime = capture["events"]["/onFrameRender/gpuTime"]["stats"]["mean"]
print(f"Mean frame time: {}", meanFrameTime)
```

#### FrameCapture

The frame capture will always dump the marked graph output. You can use `graph.markOutput()` and `graph.unmarkOutput()` to control which outputs to dump.

The frame counter starts at zero in Falcor (it starts by default at one in Maya, so the frame numbers may be offset by one frame).

By default, the captures frames are stored to the executable directory. This can be changed by setting `outputDir`.

**Note:** The frame counter is not advanced when time is paused. If you capture with time paused, the captured frame will be overwritten for every rendered frame. The workaround is to change the base filename between captures with `fc.capture()`, see example below.

class falcor.**FrameCapture**

| Property       | Type   | Description                                                                  |
|----------------|--------|------------------------------------------------------------------------------|
| `outputDir`    | `str`  | Capture output directory.                                                    |
| `baseFilename` | `str`  | Capture base filename. The frameID and output name will be appended to this. |
| `ui`           | `bool` | Show/hide the UI.                                                            |

| Method                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `reset(graph)`             | Reset frame capturing for the given graph (or all graphs if set to `None`). |
| `capture()`                | Capture the current frame.                                                  |
| `addFrames(graph, frames)` | Add a list of frames to capture for the given graph.                        |
| `print()`                  | Print the requested frames to capture for all available graphs.             |
| `print(graph)`             | Print the requested frames to capture for the specified graph.              |

**Example:** *Capture list of frames with clock running and then exit*
```python
m.clock.exitFrame = 101
m.frameCapture.outputDir = "../../../Output"
m.frameCapture.baseFilename = "Mogwai"
m.frameCapture.addFrames(m.activeGraph, [20, 50, 100])
```

**Example:** *Capture frames with clock paused and then exit*
```python
m.clock.pause()
m.frameCapture.outputDir = "../../../Output"

frames = [20, 50, 100]
for i in range(101):
    renderFrame()
    if i in frames:
        m.frameCapture.baseFilename = f"Mogwai-{i:04d}"
        m.frameCapture.capture()
exit()
```


#### VideoCapture

The video capture will always capture the marked graph output. You can use `graph.markOutput()` and `graph.unmarkOutput()` to control which outputs to dump.

enum falcor.**Codec**

`Raw`, `H264`, `HVEC`, `MPEG2`, `MPEG4`

class falcor.**VideoCapture**

| Property       | Type    | Description                                                      |
|----------------|---------|------------------------------------------------------------------|
| `outputDir`    | `str`   | Capture output directory.                                        |
| `baseFilename` | `str`   | Capture base filename. The output name will be appended to this. |
| `ui`           | `bool`  | Show/hide the UI.                                                |
| `codec`        | `Codec` | Video codec (`Raw`, `H264`, `HVEC`, `MPEG2`, `MPEG4`).           |
| `fps`          | `int`   | Video frame rate.                                                |
| `bitrate`      | `float` | Video bitrate in Mpbs.                                           |
| `gopSize`      | `int`   | Video GOP size.                                                  |

| Method                     | Description                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------|
| `reset(graph)`             | Reset video capturing for the given graph (or all graphs if set to `None`).                           |
| `addRanges(graph, ranges)` | Add a list of frame ranges to capture for a given graph. `ranges` is a list of `(start, end)` tuples. |
| `print()`                  | Print the requested ranges to capture for all available graphs.                                       |
| `print(graph)`             | Print the requested ranges to capture for the specified graph.                                        |

Example:
```python
# Video Capture
m.videoCapture.outputDir = "."
m.videoCapture.baseFilename = "Mogwai"
m.videoCapture.codec = Codec.H264
m.videoCapture.fps = 60
m.videoCapture.bitrate = 4.0
m.videoCapture.gopSize = 10
m.videoCapture.addRanges(m.activeGraph, [[30, 300]])
```

#### TimingCapture

class falcor.**TimingCapture**

| Method                   | Description                                       |
|--------------------------|---------------------------------------------------|
| `captureFrameTime(path)` | Start writing frame times to the given file path. |

Example:
```python
# Timing Capture
m.timingCapture.captureFrameTime("timecapture.csv")
```

### Core API

module **falcor**

#### Global Functions

| Function                      | Description                 |
|-------------------------------|-----------------------------|
| `loadRenderPassLibrary(name)` | Load a render pass library. |
| `cls`                         | Clear the console.          |

#### ResourceFormat

enum falcor.**ResourceFormat**

`Unknown`, `R8Unorm`, `R8Snorm`, `R16Unorm`, `R16Snorm`, `RG8Unorm`, `RG8Snorm`, `RG16Unorm`, `RG16Snorm`, `RGB16Unorm`, `RGB16Snorm`, `R24UnormX8`, `RGB5A1Unorm`, `RGBA8Unorm`, `RGBA8Snorm`, `RGB10A2Unorm`, `RGB10A2Uint`, `RGBA16Unorm`, `RGBA8UnormSrgb`, `R16Float`, `RG16Float`, `RGB16Float`, `RGBA16Float`, `R32Float`, `R32FloatX32`, `RG32Float`, `RGB32Float`, `RGBA32Float`, `R11G11B10Float`, `RGB9E5Float`, `R8Int`, `R8Uint`, `R16Int`, `R16Uint`, `R32Int`, `R32Uint`, `RG8Int`, `RG8Uint`, `RG16Int`, `RG16Uint`, `RG32Int`, `RG32Uint`, `RGB16Int`, `RGB16Uint`, `RGB32Int`, `RGB32Uint`, `RGBA8Int`, `RGBA8Uint`, `RGBA16Int`, `RGBA16Uint`, `RGBA32Int`, `RGBA32Uint`, `BGRA8Unorm`, `BGRA8UnormSrgb`, `BGRX8Unorm`, `BGRX8UnormSrgb`, `Alpha8Unorm`, `Alpha32Float`, `R5G6B5Unorm`, `D32Float`, `D16Unorm`, `D32FloatS8X24`, `D24UnormS8`, `BC1Unorm`, `BC1UnormSrgb`, `BC2Unorm`, `BC2UnormSrgb`, `BC3Unorm`, `BC3UnormSrgb`, `BC4Unorm`, `BC4Snorm`, `BC5Unorm`, `BC5Snorm`, `BC6HS16`, `BC6HU16`, `BC7Unorm`, `BC7UnormSrgb`

#### RenderGraph

class falcor.**RenderGraph**

| Property | Type  | Description               |
|----------|-------|---------------------------|
| `name`   | `str` | Name of the render graph. |

| Method                         | Description                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------------|
| `RenderGraph(name)`            | Create a new render graph.                                                                   |
| `addPass(pass, name)`          | Add a render pass to the graph.                                                              |
| `removePass(name)`             | Remove a render pass from the graph.                                                         |
| `updatePass(name, dict)`       | Update a render pass with new configuration options in `dict`.                               |
| `getPass(name)`                | Get a pass by name.                                                                          |
| `addEdge(src, dst)`            | Add an edge to the render graph.                                                             |
| `removeEdge(src, dst)`         | Remove an edge from the render graph.                                                        |
| `markOutput(name, mask)`       | Mark a render pass output as graph output. `mask` is an optional `TextureChannelFlags` enum. |
| `unmarkOutput(name)`           | Unmark an output.                                                                            |
| `getOutput(index)`             | Get an output by index.                                                                      |
| `getOutput(name)`              | Get an output by name.                                                                       |

**Note:**
* `markOutput` marks an output to be selectable in Mogwai and for frame capture. The first marked output will be the default output in Mogwai.
* The output is identified by name on the form `renderPass.outputName`.
* The function takes an optional `mask` parameter to specify which color channels to capture for frame capture. If no mask is given, the default is RGB and ignore alpha.
* An output can be marked multiple times with different masks. Each version will be written to a separate file by frame capture.

#### RenderPass

class falcor.**RenderPass**

| Property | Type  | Description                                |
|----------|-------|--------------------------------------------|
| `name`   | `str` | Name of the render pass (readonly).        |
| `type`   | `str` | Type name of the render pass (readonly).   |
| `desc`   | `str` | Description of the render pass (readonly). |

| Method            | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| `getDictionary()` | Returns the configuration dictionary with the current settings. |

#### Texture

class falcor.**Texture**

| Property    | Type             | Description                        |
|-------------|------------------|------------------------------------|
| `width`     | `int`            | Width in pixels (readonly).        |
| `height`    | `int`            | Height in pixels (readonly).       |
| `depth`     | `int`            | Depth in pixels/layers (readonly). |
| `mipCount`  | `int`            | Number of mip levels (readonly).   |
| `arraySize` | `int`            | Size of array (readonly).          |
| `samples`   | `int`            | Number of samples (readonly).      |
| `format`    | `ResourceFormat` | Texture format (readonly).         |

| Method        | Description                                   |
|---------------|-----------------------------------------------|
| `data(index)` | Returns raw data of given sub resource index. |

#### AABB

class falcor.**AABB**

| Property   | Type     | Description                               |
|------------|----------|-------------------------------------------|
| `minPoint` | `float3` | Minimum point.                            |
| `maxPoint` | `float3` | Maximum point.                            |
| `valid`    | `bool`   | True if AABB is valid (readonly).         |
| `center`   | `float3` | Center point (readonly).                  |
| `extent`   | `float3` | Extents (readonly).                       |
| `area`     | `float`  | Total area (readonly).                    |
| `volume`   | `float`  | Total volume (readonly).                  |
| `radius`   | `float`  | Radius of an enclosing sphere (readonly). |

| Method            | Description                       |
|-------------------|-----------------------------------|
| `invalidate()`    | Invalidate the AABB.              |
| `include(p)`      | Include a point in the AABB.      |
| `include(b)`      | Include another AABB in the AABB. |
| `intersection(b)` | Intersect with another AABB.      |


### Scene API

#### SceneRenderSettings

class falcor.**SceneRenderSettings**

| Property            | Type   | Description                                   |
|---------------------|--------|-----------------------------------------------|
| `useEnvLight`       | `bool` | Enable/disable lighting from environment map. |
| `useAnalyticLights` | `bool` | Enable/disable lighting from analytic lights. |
| `useEmissiveLights` | `bool` | Enable/disable lighting from emissive lights. |
| `useGridVolumes`    | `bool` | Enable/disable rendering of grid volumes.     |

#### Scene

class falcor.**Scene**

| Property         | Type                    | Description                                                             |
|------------------|-------------------------|-------------------------------------------------------------------------|
| `stats`          | `dict`                  | Dictionary containing scene stats.                                      |
| `bounds`         | `AABB`                  | World space scene bounds (readonly).                                    |
| `animated`       | `bool`                  | Enable/disable scene animations.                                        |
| `loopAnimations` | `bool`                  | Enable/disable globally looping scene animations.                       |
| `renderSettings` | `SceneRenderSettings`   | Settings to determine how the scene is rendered.                        |
| `updateCallback` | `function(scene, time)` | Called at the beginning of each frame to update the scene procedurally. |
| `camera`         | `Camera`                | Camera.                                                                 |
| `cameraSpeed`    | `float`                 | Speed of the interactive camera.                                        |
| `envMap`         | `EnvMap`                | Environment map.                                                        |
| `animations`     | `list(Animation)`       | List of animations.                                                     |
| `cameras`        | `list(Camera)`          | List of cameras.                                                        |
| `lights`         | `list(Light)`           | List of lights.                                                         |
| `materials`      | `list(Material)`        | List of materials.                                                      |
| `volumes`        | `list(Volume)`          | **DEPRECATED**: Use `gridVolumes` instead.                              |
| `gridVolumes`    | `list(GridVolume)`      | List of grid volumes.                                                   |

| Method                               | Description                                            |
|--------------------------------------|--------------------------------------------------------|
| `setEnvMap(path)`                    | Load an environment map from an image.                 |
| `getLight(index)`                    | Return a light by index.                               |
| `getLight(name)`                     | Return a light by name.                                |
| `getMaterial(index)`                 | Return a material by index.                            |
| `getMaterial(name)`                  | Return a material by name.                             |
| `getVolume(index)`                   | **DEPRECATED**: Use `getGridVolume` instead.           |
| `getGridVolume(index)`               | Return a grid volume by index.                         |
| `getVolume(name)`                    | **DEPRECATED**: Use `getGridVolume` instead.           |
| `getGridVolume(name)`                | Return a grid volume by name.                          |
| `addViewpoint()`                     | Add current camera's viewpoint to the viewpoint list.  |
| `addViewpoint(position, target, up)` | Add a viewpoint to the viewpoint list.                 |
| `removeViewpoint()`                  | Remove selected viewpoint.                             |
| `selectViewpoint(index)`             | Select a specific viewpoint and move the camera to it. |

#### Camera

class falcor.**Camera**

| Property         | Type     | Description                                  |
|------------------|----------|----------------------------------------------|
| `name`           | `str`    | Name of the camera.                          |
| `animated`       | `bool`   | Enable/disable camera animation.             |
| `aspectRatio`    | `float`  | Image aspect ratio.                          |
| `focalLength`    | `float`  | Focal length in millimeters.                 |
| `frameHeight`    | `float`  | Frame height in millimeters.                 |
| `focalDistance`  | `float`  | Focal distance in scene units.               |
| `apertureRadius` | `float`  | Apeture radius in scene units.               |
| `shutterSpeed`   | `float`  | Shutter speed  in seconds (not implemented). |
| `ISOSpeed`       | `float`  | Film speed (not implemented).                |
| `nearPlane`      | `float`  | Near plane distance in scene units.          |
| `farPlane`       | `float`  | Far plane distance in scene units.           |
| `position`       | `float3` | Camera position in world space.              |
| `target`         | `float3` | Camera target in world space.                |
| `up`             | `float3` | Camera up vector in world space.             |

#### EnvMap

class falcor.**EnvMap**

| Property    | Type     | Description                                     |
|-------------|----------|-------------------------------------------------|
| `path`      | `str`    | File path of loaded environment map (readonly). |
| `rotation`  | `float3` | Rotation angles in degrees (XYZ).               |
| `intensity` | `float`  | Intensity (scalar multiplier).                  |

| Static method          | Description                           |
|------------------------|---------------------------------------|
| `createFromFile(path)` | Create a environment map from a file. |

#### Material

**DEPRECATED**: Use `StandardMaterial` instead.

#### StandardMaterial

enum falcor.**MaterialTextureSlot**

`BaseColor`, `Specular`, `Emissive`, `Normal`, `Transmission`, `Displacement`

enum falcor.**AlphaMode**

`Opaque`, `Mask`

enum falcor.**ShadingModel**

`MetalRough`, `SpecGloss`

class falcor.**StandardMaterial**

| Property               | Type           | Description                                          |
|------------------------|----------------|------------------------------------------------------|
| `name`                 | `str`          | Name of the material.                                |
| `shadingModel`         | `ShadingModel` | Shading model (readonly).                            |
| `baseColor`            | `float4`       | Base color (linear RGB) and opacity (alpha).         |
| `specularParams`       | `float4`       | Specular parameters.                                 |
| `roughness`            | `float`        | Roughness (0 = smooth, 1 = rough).                   |
| `metallic`             | `float`        | Metallic (0 = dielectric, 1 = conductive).           |
| `transmissionColor`    | `float3`       | Transmission color.                                  |
| `diffuseTransmission`  | `float`        | Diffuse transmission (0 = opaque, 1 = transparent).  |
| `specularTransmission` | `float`        | Specular transmission (0 = opaque, 1 = transparent). |
| `indexOfRefraction`    | `float`        | Index of refraction.                                 |
| `emissiveColor`        | `float3`       | Emissive color (linear RGB).                         |
| `emissiveFactor`       | `float`        | Multiplier for emissive color.                       |
| `alphaMode`            | `AlphaMode`    | Alpha mode (opaque or mask)                          |
| `alphaThreshold`       | `float`        | Alpha masking threshold (0-1).                       |
| `doubleSided`          | `bool`         | Enable double sided rendering.                       |
| `thinSurface`          | `bool`         | Enable thin surface rendering.                       |
| `nestedPriority`       | `int`          | Nested priority for nested dielectrics.              |
| `volumeAbsorption`     | `float3`       | Volume absorption coefficient.                       |
| `volumeScattering`     | `float3`       | Volume scattering coefficient.                       |
| `volumeAnisotropy`     | `float`        | Volume phase function anisotropy (g).                |
| `displacementScale`    | `float`        | Displacement mapping scale value.                    |
| `displacementOffset`   | `float`        | Displacement mapping offset value.                   |

| Method                                  | Description                                |
|-----------------------------------------|--------------------------------------------|
| `clearTexture(slot)`                    | Clears one of the texture slots.           |
| `loadTexture(slot, path, useSrgb=True)` | Load one of the texture slots from a file. |

#### HairMaterial

class falcor.**HairMaterial**

Interface identical to `StandardMaterial`

#### ClothMaterial

class falcor.**ClothMaterial**

Interface identical to `StandardMaterial`

#### Grid

class falcor.**Grid**

| Property     | Type    | Description                                           |
|--------------|---------|-------------------------------------------------------|
| `voxelCount` | `int`   | Total number of active voxels in the grid (readonly). |
| `minIndex`   | `int3`  | Minimum index stored in the grid (readonly).          |
| `maxIndex`   | `int3`  | Maximum index stored in the grid (readonly).          |
| `minValue`   | `float` | Minimum value stored in the grid (readonly).          |
| `maxValue`   | `float` | Maximum value stored in the grid (readonly).          |

| Method          | Description                                            |
|-----------------|--------------------------------------------------------|
| `getValue(ijk)` | Access the value of a voxel in the grid (index space). |

| Static method                                                | Description                                 |
|--------------------------------------------------------------|---------------------------------------------|
| `createSphere(radius, voxelSize, blendRange=2.0)`            | Create a sphere grid.                       |
| `createBox(width, height, depth, voxelSize, blendRange=2.0)` | Create a box grid.                          |
| `createFromFile(path, gridname)`                             | Create a grid from an OpenVDB/NanoVDB file. |

#### Volume

**DEPRECATED**: Use `GridVolume` instead.

#### GridVolume

enum falcor.GridVolume.**GridSlot**

`Density`, `Emission`

enum falcor.GridVolume.**EmissionMode**

`Direct`, `Blackbody`

class falcor.**GridVolume**

| Property              | Type           | Description                                             |
|-----------------------|----------------|---------------------------------------------------------|
| `name`                | `str`          | Name of the volume.                                     |
| `gridFrame`           | `int`          | Current frame in the grid sequence.                     |
| `gridFrameCount`      | `int`          | Total number of frames in the grid sequence (readonly). |
| `frameRate`           | `float`        | Frame rate for grid animation.                          |
| `playbackEnabled`     | `bool`         | Enable/disable grid animation playback.                 |
| `densityGrid`         | `Grid`         | Density grid.                                           |
| `densityScale`        | `float`        | Density scale factor.                                   |
| `emissionGrid`        | `Grid`         | Emission grid.                                          |
| `emissionScale`       | `float`        | Emission scale factor.                                  |
| `albedo`              | `float3`       | Scattering albedo.                                      |
| `anisotropy`          | `float`        | Phase function anisotropy (g).                          |
| `emissionMode`        | `EmissionMode` | Emission mode (Direct, Blackbody).                      |
| `emissionTemperature` | `float`        | Emission base temperature (K).                          |

| Method                                    | Description                                                                         |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| `loadGrid(slot, path, gridname)`          | Load a grid slot from an OpenVDB/NanoVDB file.                                      |
| `loadGridSequence(slot, paths, gridname)` | Load a grid slot from a sequence of OpenVDB/NanoVDB files.                          |
| `loadGridSequence(slot, path, gridname)`  | Load a grid slot from a sequence of OpenVDB/NanoVDB files contained in a directory. |

#### Light

class falcor.**Light**

| Property    | Type     | Description                     |
|-------------|----------|---------------------------------|
| `name`      | `str`    | Name of the light.              |
| `active`    | `bool`   | Enable/disable light.           |
| `animated`  | `bool`   | Enable/disable light animation. |
| `intensity` | `float3` | Intensity of the light.         |

class falcor.**PointLight** : falcor.**Light**

| Property        | Type     | Description                            |
|-----------------|----------|----------------------------------------|
| `position`      | `float3` | Position of the light in world space.  |
| `direction`     | `float3` | Direction of the light in world space. |
| `openingAngle`  | `float`  | Opening half-angle in radians.         |
| `penumbraAngle` | `float`  | Penumbra half-angle in radians.        |

class falcor.**DirectionalLight** : falcor.**Light**

| Property    | Type     | Description                            |
|-------------|----------|----------------------------------------|
| `direction` | `float3` | Direction of the light in world space. |

class falcor.**DistantLight** : falcor.**Light**

| Property    | Type     | Description                                   |
|-------------|----------|-----------------------------------------------|
| `direction` | `float3` | Direction of the light in world space.        |
| `angle`     | `float`  | Half-angle subtended by the light in radians. |

class falcor.**AnalyticAreaLight** : falcor.**Light**

class falcor.**RectLight** : falcor.**AnalyticAreaLight**

class falcor.**DiscLight** : falcor.**AnalyticAreaLight**

class falcor.**SphereLight** : falcor.**AnalyticAreaLight**

#### Transform

class falcor.**Transform**

| Property           | Type       | Description                                 |
|--------------------|------------|---------------------------------------------|
| `translation`      | `float3`   | Translation.                                |
| `rotationEuler`    | `float3`   | Euler rotation angles around XYZ (radians). |
| `rotationEulerDeg` | `float3`   | Euler rotation angles around XYZ (degrees). |
| `scaling`          | `float3`   | Scaling.                                    |
| `matrix`           | `float4x4` | Transformation matrix (readonly).           |

| Method                         | Description                      |
|--------------------------------|----------------------------------|
| `lookAt(position, target, up)` | Set up a look-at transformation. |

#### Animation

enum falcor.Animation.**InterpolationMode**

`Linear`, `Hermite`

enum falcor.Animation.**Behavior**

`Constant`, `Linear`, `Cycle`, `Oscillate`

class falcor.**Animation**

| Property               | Type                | Description                                                              |
|------------------------|---------------------|--------------------------------------------------------------------------|
| `name`                 | `str`               | Name of the animation (readonly).                                        |
| `nodeID`               | `int`               | Animated scene graph node (readonly).                                    |
| `duration`             | `float`             | Duration in seconds (readonly).                                          |
| `interpolationMode`    | `InterpolationMode` | Interpolation mode (linear, hermite).                                    |
| `preInfinityBehavior`  | `Behavior`          | Behavior before the first keyframe (constant, linear, cycle, oscillate). |
| `postInfinityBehavior` | `Behavior`          | Behavior after the last keyframe (constant, linear, cycle, oscillate).   |
| `enableWarping`        | `bool`              | Enable/disable warping, i.e. interpolating from last to first keyframe.  |

| Method                         | Description                                  |
|--------------------------------|----------------------------------------------|
| `addKeyframe(time, transform)` | Add a transformation keyframe at given time. |

#### TriangleMesh

class falcor.TriangleMesh.**Vertex**

| Field      | Type     | Description                |
|------------|----------|----------------------------|
| `position` | `float3` | Vertex position.           |
| `normal`   | `float3` | Vertex normal.             |
| `texCoord` | `float2` | Vertex texture coordinate. |

class falcor.**TriangleMesh**

| Property   | Type           | Description                  |
|------------|----------------|------------------------------|
| `name`     | `str`          | Name of the triangle mesh.   |
| `vertices` | `list(Vertex)` | List of vertices (readonly). |
| `indices`  | `list(int)`    | List of indices (readonly).  |

| Method                                  | Description                                         |
|-----------------------------------------|-----------------------------------------------------|
| `addVertex(position, normal, texCoord)` | Add a vertex to the mesh. Returns the vertex index. |
| `addTriangle(i0, i1, i2)`               | Add a triangle to the mesh.                         |

| Class Method                                         | Description                                                                                                                                       |
|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `createQuad(size=float2(1))`                         | Creates a quad mesh, centered at the origin with normal pointing in positive Y direction.                                                         |
| `createCube(size=float3(1))`                         | Creates a cube mesh, centered at the origin.                                                                                                      |
| `createSphere(radius=1, segmentsU=32, segmentsV=16)` | Creates a UV sphere mesh, centered at the origin with poles in positive/negative Y direction.                                                     |
| `createFromFile(path, smoothNormals=False)`          | Creates a triangle mesh from a file. If no normals are defined in the file, `smoothNormals` can be used generate smooth instead of facet normals. |

#### SceneBuiler

enum falcor.**SceneBuilderFlags**

| Enum                         | Description                                                                                                                                                                                           |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Default`                    | Use the default flags (0).                                                                                                                                                                            |
| `DontMergeMaterials`         | Don't merge materials that have the same properties. Use this option to preserve the original material names.                                                                                         |
| `UseOriginalTangentSpace`    | Use the original bitangents that were loaded with the mesh. By default, we will ignore them and use MikkTSpace to generate the tangent space. We will always generate bitangents if they are missing. |
| `AssumeLinearSpaceTextures`  | By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.                                                     |
| `DontMergeMeshes`            | Preserve the original list of meshes in the scene, don't merge meshes with the same material.                                                                                                         |
| `UseSpecGlossMaterials`      | Set materials to use Spec-Gloss shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.                                                                              |
| `UseMetalRoughMaterials`     | Set materials to use Metal-Rough shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.                                                                             |
| `NonIndexedVertices`         | Convert meshes to use non-indexed vertices. This requires more memory but may increase performance.                                                                                                   |
| `Force32BitIndices`          | Force 32-bit indices for all meshes. By default, 16-bit indices are used for small meshes.                                                                                                            |
| `RTDontMergeStatic`          | For raytracing, don't merge all static non-instanced meshes into single pre-transformed BLAS.                                                                                                         |
| `RTDontMergeDynamic`         | For raytracing, don't merge dynamic non-instanced meshes with identical transforms into single BLAS.                                                                                                  |
| `RTDontMergeInstanced`       | For raytracing, don't merge instanced meshes with identical instances into single BLAS.                                                                                                               |
| `FlattenStaticMeshInstances` | Flatten static mesh instances by duplicating mesh data and composing transformations. Animated instances are not affected. Can lead to a large increase in memory use.                                |
| `DontOptimizeGraph`          | Don't optimize the scene graph to remove unnecessary nodes.                                                                                                                                           |
| `DontOptimizeMaterials`      | Don't optimize materials by removing constant textures. The optimizations are lossless so should generally be enabled.                                                                                |
| `DontUseDisplacement`        | Don't use displacement mapping.                                                                                                                                                                       |
| `UseCache`                   | Enable scene caching. This caches the runtime scene representation on disk to reduce load time.                                                                                                       |
| `RebuildCache`               | Rebuild scene cache.                                                                                                                                                                                  |

class falcor.**SceneBuilder**

| Property         | Type                  | Description                                      |
|------------------|-----------------------|--------------------------------------------------|
| `flags`          | `SceneBuilderFlags`   | Scene builder flags (readonly).                  |
| `renderSettings` | `SceneRenderSettings` | Settings to determine how the scene is rendered. |
| `materials`      | `list(Material)`      | List of materials (readonly).                    |
| `volumes`        | `list(Volume)`        | **DEPRECATED**: Use `gridVolumes` instead.       |
| `gridVolumes`    | `list(GridVolume)`    | List of grid volumes (readonly).                 |
| `lights`         | `list(Light)`         | List of lights (readonly).                       |
| `cameras`        | `list(Camera)`        | List of cameras (readonly).                      |
| `animations`     | `list(Animation)`     | List of animations (readonly).                   |
| `envMap`         | `EnvMap`              | Environment map.                                 |
| `selectedCamera` | `Camera`              | Default selected camera.                         |
| `cameraSpeed`    | `float`               | Speed of the interactive camera.                 |

| Method                                        | Description                                                                                                     |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `importScene(path, dict, instances)`          | Load a scene from an asset file. `dict` contains optional data. `instances` is an optional list of `Transform`. |
| `addTriangleMesh(triangleMesh, material)`     | Add a triangle mesh to the scene and return its ID.                                                             |
| `addMaterial(material)`                       | Add a material and return its ID.                                                                               |
| `getMaterial(name)`                           | Return a material by name. The first material with matching name is returned or `None` if none was found.       |
| `loadMaterialTexture(material, slot, path)`   | Request loading a material texture asynchronously. Use `Material.loadTexture` for synchronous loading.          |
| `waitForMaterialTextureLoading()`             | Wait until all material textures are loaded.                                                                    |
| `addVolume(volume)`                           | **DEPRECATED**: Use `addGridVolume` instead.                                                                    |
| `addGridVolume(gridVolume)`                   | Add a grid volume and return its ID.                                                                            |
| `getVolume(name)`                             | **DEPRECATED**: Use `getGridVolume` instead.                                                                    |
| `getGridVolume(name)`                         | Return a grid volume by name. The first volume with matching name is returned or `None` if none was found.      |
| `addLight(light)`                             | Add a light and return its ID.                                                                                  |
| `getLight(name)`                              | Return a light by name. The first light with matching name is returned or `None` if none was found.             |
| `addCamera(camera)`                           | Add a camera and return its ID.                                                                                 |
| `addAnimation(animation)`                     | Add an animation.                                                                                               |
| `createAnimation(animatable, name, duration)` | Create an animation for an animatable object. Returns the new animation or `None` if one already exists.        |
| `addNode(name, transform, parent)`            | Add a node and return its ID.                                                                                   |
| `addMeshInstance(nodeID, meshID)`             | Add a mesh instance.                                                                                            |
| `addCustomPrimitive(userID, aabb)`            | Add a custom primitive. 'aabb' is an AABB specifying its bounds.                                                |
| `addSDFGridInstance(userID, sdfGridID)`       | Add a SDF grid instance.                                                                                        |
| `addSDFGrid(sdfGrid, maternal)`               | Add a SDF grid and returns its ID.                                                                              |


### Render Pass Helpers

#### IOSize

enum falcor.RenderPassHelpers.**IOSize**

| Enum              | Description                                                                                                               |
|-------------------|---------------------------------------------------------------------------------------------------------------------------|
| Default           | Use the default size. The size is determined based on whatever is bound (the system will use the window size by default). |
| Fixed             | Use fixed size in pixels.                                                                                                 |
| Full              | Use full window size.                                                                                                     |
| Half              | Use half window size.                                                                                                     |
| Quarter           | Use quarter window size.                                                                                                  |
| Double            | Use double window size.                                                                                                   |

Some render passes have options to set output resolution using the `IOSize` enum.
Usage examples (add these options to the scripting dictionary when creating the render pass):

```python
'outputSize': IOSize.Half
```

```python
'outputSize': IOSize.Fixed, 'fixedOutputSize': uint2(128,64)
```

### Render Passes

**Note:** These docs are not complete. See the source code for the full set of exported methods and properties.

#### AccumulatePass

enum falcor.**AccumulatePrecision**

`Double`, `Single`, `SingleCompensated`

class falcor.**AccumulatePass**

| Method    | Description                                                                               |
|-----------|-------------------------------------------------------------------------------------------|
| `reset()` | Reset accumulation. This is useful when the pass has been created with 'autoReset': False |

| Property                | Type        | Description                                                                   |
|-------------------------|-------------|-------------------------------------------------------------------------------|
| `outputSize`            | `IOSize`    | Set output resolution.                                                        |
| `fixedOutputSize`       | `uint2`     | Fixed output resolution in (width, height) pixels when using `IOSize.Fixed`.  |

#### ToneMapper

enum falcor.**ToneMapOp**

`Linear`, `Reinhard`, `ReinhardModified`, `HejiHableAlu`, `HableUc2`, `Aces`

class falcor.**ToneMapper**

| Property                | Type        | Description                                                                   |
|-------------------------|-------------|-------------------------------------------------------------------------------|
| `exposureCompensation`  | `float`     | Exposure compensation (applies in manual and auto exposure).                  |
| `autoExposure`          | `bool`      | Enable/disable auto exposure.                                                 |
| `exposureValue`         | `float`     | Exposure value in manual mode.                                                |
| `filmSpeed`             | `float`     | ISO film speed in manual mode.                                                |
| `whiteBalance`          | `bool`      | Enable/disable white balancing.                                               |
| `whitePoint`            | `float`     | White point in Kelvin.                                                        |
| `operator`              | `ToneMapOp` | Tone mapping operator.                                                        |
| `clamp`                 | `bool`      | Enable/disable clamping to [0..1] range.                                      |
| `outputSize`            | `IOSize`    | Set output resolution.                                                        |
| `fixedOutputSize`       | `uint2`     | Fixed output resolution in (width, height) pixels when using `IOSize.Fixed`.  |

#### SimplePostFX

class falcor.**SimplePostFX**

| Property                | Type        | Description                                                                   |
|-------------------------|-------------|-------------------------------------------------------------------------------|
| `outputSize`            | `IOSize`    | Set output resolution.                                                        |
| `fixedOutputSize`       | `uint2`     | Fixed output resolution in (width, height) pixels when using `IOSize.Fixed`.  |

#### GBuffer passes

class falcor.**GBufferRaster**
class falcor.**VBufferRaster**
class falcor.**GBufferRT**
class falcor.**VBufferRT**

| Property                | Type        | Description                                                                   |
|-------------------------|-------------|-------------------------------------------------------------------------------|
| `outputSize`            | `IOSize`    | Set output resolution.                                                        |
| `fixedOutputSize`       | `uint2`     | Fixed output resolution in (width, height) pixels when using `IOSize.Fixed`.  |

#### ImageLoader

class falcor.**ImageLoader**

| Property                | Type        | Description                                                                   |
|-------------------------|-------------|-------------------------------------------------------------------------------|
| `outputSize`            | `IOSize`    | Set output resolution.                                                        |

When `IOSize.Fixed` is used, the render pass output is at the native resolution of the loaded image. In all other modes the image is bilinearly rescaled to the desired output resolution.

#### GaussianBlur

class falcor.**GaussianBlur**

| Property      | Type    | Description             |
|---------------|---------|-------------------------|
| `kernelWidth` | `int`   | Kernel width in pixels. |
| `sigma`       | `float` | Sigma of gaussian.      |

#### SSAO

class falcor.**SSAO**

| Property       | Type    | Description              |
|----------------|---------|--------------------------|
| `kernelRadius` | `int`   | Kernel radius in pixels. |
| `sampleRadius` | `float` | Sampling radius.         |
| `distribution` | `int`   | Sampling distribution.   |

#### Skybox

class falcor.**SkyBox**

| Property | Type    | Description       |
|----------|---------|-------------------|
| `scale`  | `float` | Color multiplier. |
| `filter` | `int`   | Filtering mode.   |

#### CSM

class falcor.**CSM**

| Property             | Type    | Description |
|----------------------|---------|-------------|
| `cascadeCount`       | `int`   |             |
| `mapSize`            | `uint2` |             |
| `visibilityBitCount` | `int`   |             |
| `filter`             | `int`   |             |
| `sdsmLatency`        | `int`   |             |
| `partition`          | `int`   |             |
| `lambda`             | `float` |             |
| `minDistance`        | `float` |             |
| `maxDistance`        | `float` |             |
| `cascadeThreshold`   | `float` |             |
| `depthBias`          | `float` |             |
| `kernelWidth`        | `int`   |             |
| `maxAniso`           | `int`   |             |
| `bleedReduction`     | `float` |             |
| `positiveExp`        | `float` |             |
| `negativeExp`        | `float` |             |

#### FXAA

class falcor.**FXAA**

| Property           | Type    | Description |
|--------------------|---------|-------------|
| `qualitySubPix`    | `float` |             |
| `edgeThreshold`    | `float` |             |
| `edgeThresholdMin` | `float` |             |
| `earlyOut`         | `bool`  |             |

#### TAA

class falcor.**TAA**

| Property | Type    | Description |
|----------|---------|-------------|
| `alpha`  | `float` |             |
| `sigma`  | `float` |             |
