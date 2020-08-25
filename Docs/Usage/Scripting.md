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

#### Global variables

| Variable | Description             |
|----------|-------------------------|
| `m`      | Instance of `Renderer`. |

## Mogwai API

#### Global functions

| Function       | Description                      |
|----------------|----------------------------------|
| `help()`       | Print global help.               |
| `help(object)` | Print help of a specific object. |

#### Global variables

| Variable | Description                  |
|----------|------------------------------|
| `t`      | Instance of `Clock`.         |
| `fc`     | Instance of `FrameCapture`.  |
| `vc`     | Instance of `VideoCapture`.  |
| `tc`     | Instance of `TimingCapture`. |

#### Renderer

class falcor.**Renderer**

| Property      | Type          | Description                     |
|---------------|---------------|---------------------------------|
| `scene`       | `Scene`       | Active scene (readonly).        |
| `activeGraph` | `RenderGraph` | Active render graph (readonly). |
| `ui`          | `bool`        | Show/hide the UI.               |

| Method                                                      | Description                                                     |
|-------------------------------------------------------------|-----------------------------------------------------------------|
| `script(filename)`                                          | Run a script.                                                   |
| `loadScene(filename, buildFlags=SceneBuilderFlags.Default)` | Load a scene. See available build flags below.                  |
| `saveConfig(filename)`                                      | Save the current state to a config file.                        |
| `addGraph(graph)`                                           | Add a render graph.                                             |
| `removeGraph(graph)`                                        | Remove a render graph. `graph` can be a render graph or a name. |
| `getGraph(name)`                                            | Get a render graph by name.                                     |
| `graph(name)`                                               | **DEPRECATED:** Use `getGraph` instead.                         |
| `resizeSwapChain(width, height)`                            | **DEPRECATED:** Use global `resizeSwapChain` instead.           |

enum falcor.**SceneBuilderFlags**

| Enum                          | Description                                                                                                                                                                                           |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Default`                     | Use the default flags (0).                                                                                                                                                                            |
| `RemoveDuplicateMaterials`    | Deduplicate materials that have the same properties. The material name is ignored during the search.                                                                                                  |
| `UseOriginalTangentSpace`     | Use the original bitangents that were loaded with the mesh. By default, we will ignore them and use MikkTSpace to generate the tangent space. We will always generate bitangents if they are missing. |
| `AssumeLinearSpaceTextures`   | By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.                                                     |
| `DontMergeMeshes`             | Preserve the original list of meshes in the scene, don't merge meshes with the same material.                                                                                                         |
| `BuffersAsShaderResource`     | Generate the VBs and IB with the shader-resource-view bind flag.                                                                                                                                      |
| `UseSpecGlossMaterials`       | Set materials to use Spec-Gloss shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.                                                                              |
| `UseMetalRoughMaterials`      | Set materials to use Metal-Rough shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.                                                                             |
| `NonIndexedVertices`          | Convert meshes to use non-indexed vertices. This requires more memory but may increase performance.                                                                                                   |


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
| `stop()`         | Stop the clock (pause and reset)  |
| `step(frames=1)` | Step forward or backward in time. |

#### FrameCapture

The frame capture will always dump the marked graph output. You can use `graph.markOutput()` and `graph.unmarkOutput()` to control which outputs to dump.

The frame counter starts at zero in Falcor (it starts by default at one in Maya, so the frame numbers may be offset by one frame).

By default, the captures frames are stored to the executable directory. This can be changed by setting `fc.outputDir`.

**Note:** The frame counter is not advanced when time is paused. If you capture with time paused, the captured frame will be overwritten for every rendered frame. The workaround is to change the base filename between captures with `fc.capture()`, see example below.

class falcor.**FrameCapture**

| Property       | Type     | Description                                                                  |
|----------------|----------|------------------------------------------------------------------------------|
| `outputDir`    | `string` | Capture output directory.                                                    |
| `baseFilename` | `string` | Capture base filename. The frameID and output name will be appended to this. |
| `ui`           | `bool`   | Show/hide the UI.                                                            |

| Method                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `reset(graph)`             | Reset frame capturing for the given graph (or all graphs if set to `None`). |
| `capture()`                | Capture the current frame.                                                  |
| `addFrames(graph, frames)` | Add a list of frames to capture for the given graph.                        |
| `print()`                  | Print the requested frames to capture for all available graphs.             |
| `print(graph)`             | Print the requested frames to capture for the specified graph.              |
| `frames(graph, frames)`    | **DEPRECATED:** Use `addFrames` instead.                                    |

**Example:** *Capture list of frames with clock running and then exit*
```python
t.exitFrame(101)
fc.outputDir = "../../../Output"
fc.baseFilename = "Mogwai"
fc.addFrames(m.activeGraph, [20, 50, 100])
```

**Example:** *Capture frames with clock paused and then exit*
```python
t.pause()
fc.outputDir = "../../../Output"

frames = [20, 50, 100]
for i in range(101):
    renderFrame()
    if i in frames:
        fc.baseFilename = f"Mogwai-{i:04d}"
        fc.capture()
exit()
```


#### VideoCapture
The video capture will always capture the marked graph output. You can use `graph.markOutput()` and `graph.unmarkOutput()` to control which outputs to dump.

enum falcor.**Codec**

`Raw`, `H264`, `HVEC`, `MPEG2`, `MPEG4`

class falcor.**VideoCapture**

| Property       | Type     | Description                                                      |
|----------------|----------|------------------------------------------------------------------|
| `outputDir`    | `string` | Capture output directory.                                        |
| `baseFilename` | `string` | Capture base filename. The output name will be appended to this. |
| `ui`           | `bool`   | Show/hide the UI.                                                |
| `codec`        | `Codec`  | Video codec (`Raw`, `H264`, `HVEC`, `MPEG2`, `MPEG4`).           |
| `fps`          | `int`    | Video frame rate.                                                |
| `bitrate`      | `float`  | Video bitrate in Mpbs.                                           |
| `gopSize`      | `int`    | Video GOP size.                                                  |

| Method                     | Description                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------|
| `reset(graph)`             | Reset video capturing for the given graph (or all graphs if set to `None`).                           |
| `addRanges(graph, ranges)` | Add a list of frame ranges to capture for a given graph. `ranges` is a list of `(start, end)` tuples. |
| `print()`                  | Print the requested ranges to capture for all available graphs.                                       |
| `print(graph)`             | Print the requested ranges to capture for the specified graph.                                        |
| `ranges(graph, ranges)`    | **DEPRECATED:** Use `addRanges` instead.                                                              |

Example:
```python
# Video Capture
vc.outputDir = "."
vc.baseFilename = "Mogwai"
vc.codec = Codec.H264
vc.fps = 60
vc.bitrate = 4.0
vc.gopSize = 10
vc.addRanges(m.activeGraph, [[30, 300]])
```

#### TimingCapture

class falcor.**TimingCapture**

| Method                       | Description                                      |
|------------------------------|--------------------------------------------------|
| `captureFrameTime(filename)` | Start writing frame times to the given filename. |

Example:
```python
# Timing Capture
tc.captureFrameTime("timecapture.csv")
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

| Property | Type     | Description               |
|----------|----------|---------------------------|
| `name`   | `string` | Name of the render graph. |

| Method                         | Description                                                                        |
|--------------------------------|------------------------------------------------------------------------------------|
| `RenderGraph(name)`            | Create a new render graph.                                                         |
| `addPass(pass, name)`          | Add a render pass to the graph.                                                    |
| `removePass(name)`             | Remove a render pass from the graph.                                               |
| `updatePass(name, dict)`       | Update a render pass with new configuration options in `dict`.                     |
| `getPass(name)`                | Get a pass by name.                                                                |
| `addEdge(src, dst)`            | Add an edge to the render graph.                                                   |
| `removeEdge(src, dst)`         | Remove an edge from the render graph.                                              |
| `autoGenEdges(executionOrder)` | TODO document                                                                      |
| `markOutput(name)`             | Mark an output to be selectable in Mogwai and writing files when capturing frames. |
| `unmarkOutput(name)`           | Unmark an output.                                                                  |
| `getOutput(index)`             | Get an output by index.                                                            |
| `getOutput(name)`              | Get an output by name.                                                             |

#### RenderPass

class falcor.**RenderPass**

| Method                   | Description                                               |
|--------------------------|-----------------------------------------------------------|
| `RenderPass(name, dict)` | **DEPRECATED:** Use global `createPass` function instead. |

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

### Scene API

#### SceneRenderSettings

class falcor.**SceneRenderSettings**

| Property            | Type   | Description                                           |
|---------------------|--------|-------------------------------------------------------|
| `useEnvLight`       | `bool` | Enable/disable distant lighting from environment map. |
| `useAnalyticLights` | `bool` | Enable/disable lighting from analytic lights.         |
| `useEmissiveLights` | `bool` | Enable/disable lighting from emissive lights.         |

#### Scene

class falcor.**Scene**

| Property         | Type                  | Description                                      |
|------------------|-----------------------|--------------------------------------------------|
| `animated`       | `bool`                | Enable/disable scene animations.                 |
| `renderSettings` | `SceneRenderSettings` | Settings to determine how the scene is rendered. |
| `camera`         | `Camera`              | Camera.                                          |
| `cameraSpeed`    | `float`               | Speed of the interactive camera.                 |
| `envMap`         | `EnvMap`              | Environment map.                                 |
| `materials`      | `list(Material)`      | List of materials                                |

| Method                               | Description                                            |
|--------------------------------------|--------------------------------------------------------|
| `animate(enable)`                    | **DEPRECATED:** Use `animated` instead.                |
| `animateCamera(enabled)`             | **DEPRECATED:** Use `animated` on `Camera` instead.    |
| `animateLight(index, enabled)`       | **DEPRECATED:** Use `animated` on `Light` instead.     |
| `setEnvMap(filename)`                | Load an environment map from an image.                 |
| `getLight(index)`                    | Return a light by index.                               |
| `getLight(name)`                     | Return a light by name.                                |
| `light(index)`                       | **DEPRECATED:** Use `getLight` instead.                |
| `light(name)`                        | **DEPRECATED:** Use `getLight` instead.                |
| `getMaterial(index)`                 | Return a material by index.                            |
| `getMaterial(name)`                  | Return a material by name.                             |
| `material(index)`                    | **DEPRECATED:** Use `getMaterial` instead.             |
| `material(name)`                     | **DEPRECATED:** Use `getMaterial` instead.             |
| `addViewpoint()`                     | Add current camera's viewpoint to the viewpoint list.  |
| `addViewpoint(position, target, up)` | Add a viewpoint to the viewpoint list.                 |
| `removeViewpoint()`                  | Remove selected viewpoint.                             |
| `selectViewpoint(index)`             | Select a specific viewpoint and move the camera to it. |
| `viewpoint()`                        | **DEPRECATED:** Use `addViewpoint` instead.            |
| `viewpoint(index)`                   | **DEPRECATED:** Use `selectViewpoint` instead.         |

#### Camera

class falcor.**Camera**

| Property         | Type     | Description                                  |
|------------------|----------|----------------------------------------------|
| `name`           | `str`    | Name of the camera (readonly).               |
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

| Property    | Type     | Description                                    |
|-------------|----------|------------------------------------------------|
| `filename`  | `str`    | Filename of loaded environment map (readonly). |
| `rotation`  | `float3` | Rotation angles in degrees (XYZ).              |
| `intensity` | `float`  | Intensity (scalar multiplier).                 |

#### Material

enum falcor.**MaterialTextureSlot**

`BaseColor`, `Specular`, `Emissive`, `Normal`, `Occlusion`, `SpecularTransmission`

class falcor.**Material**

| Property               | Type     | Description                                           |
|------------------------|----------|-------------------------------------------------------|
| `name`                 | `str`    | Name of the material (readonly).                      |
| `baseColor`            | `float4` | Base color (linear RGB) and opacity.                  |
| `specularParams`       | `float4` | Specular parameters (occlusion, roughness, metallic). |
| `roughness`            | `float`  | Roughness (0 = smooth, 1 = rough).                    |
| `metallic`             | `float`  | Metallic (0 = dielectric, 1 = conductive).            |
| `specularTransmission` | `float`  | Specular transmission (0 = opaque, 1 = transparent).  |
| `indexOfRefraction`    | `float`  | Index of refraction.                                  |
| `emissiveColor`        | `float3` | Emissive color (linear RGB).                          |
| `emissiveFactor`       | `float`  | Multiplier for emissive color.                        |
| `alphaMode`            | `int`    | Alpha mode (0 = opaque, 1 = masked).                  |
| `alphaThreshold`       | `float`  | Alpha masking threshold (0-1).                        |
| `doubleSided`          | `bool`   | Enable double sided rendering.                        |
| `nestedPriority`       | `int`    | Nested priority for nested dielectrics.               |

| Method                                      | Description                                |
|---------------------------------------------|--------------------------------------------|
| `clearTexture(slot)`                        | Clears one of the texture slots.           |
| `loadTexture(slot, filename, useSrgb=True)` | Load one of the texture slots from a file. |

#### Light

class falcor.**Light**

| Property    | Type     | Description                     |
|-------------|----------|---------------------------------|
| `name`      | `str`    | Name of the light (readonly).   |
| `active`    | `bool`   | Enable/disable light.           |
| `animated`  | `bool`   | Enable/disable light animation. |
| `color`     | `float3` | Color of the light.             |
| `intensity` | `float`  | Intensity of the light.         |

class falcor.**DirectionalLight**

| Property    | Type     | Description                            |
|-------------|----------|----------------------------------------|
| `name`      | `str`    | Name of the light (readonly).          |
| `color`     | `float3` | Color of the light.                    |
| `intensity` | `float`  | Intensity of the light.                |
| `direction` | `float3` | Direction of the light in world space. |

class falcor.**DistantLight**

| Property    | Type     | Description                                   |
|-------------|----------|-----------------------------------------------|
| `name`      | `str`    | Name of the light (readonly).                 |
| `color`     | `float3` | Color of the light.                           |
| `intensity` | `float`  | Intensity of the light.                       |
| `direction` | `float3` | Direction of the light in world space.        |
| `angle`     | `float`  | Half-angle subtended by the light in radians. |

class falcor.**PointLight**

| Property        | Type     | Description                            |
|-----------------|----------|----------------------------------------|
| `name`          | `str`    | Name of the light (readonly).          |
| `color`         | `float3` | Color of the light.                    |
| `intensity`     | `float`  | Intensity of the light.                |
| `position`      | `float3` | Position of the light in world space.  |
| `direction`     | `float3` | Direction of the light in world space. |
| `openingAngle`  | `float`  | Opening half-angle in radians.         |
| `penumbraAngle` | `float`  | Penumbra half-angle in radians.        |

class falcor.**AnalyticAreaLight**

| Property    | Type     | Description                   |
|-------------|----------|-------------------------------|
| `name`      | `str`    | Name of the light (readonly). |
| `color`     | `float3` | Color of the light.           |
| `intensity` | `float`  | Intensity of the light.       |

### Render Passes

#### AccumulatePass

enum falcor.**AccumulatePrecision**

`Double`, `Single`, `SingleCompensated`

class falcor.**AccumulatePass**

| Method    | Description                                                                               |
|-----------|-------------------------------------------------------------------------------------------|
| `reset()` | Reset accumulation. This is useful when the pass has been created with 'autoReset': False |

#### ToneMapper

enum falcor.**ToneMapOp**

`Linear`, `Reinhard`, `ReinhardModified`, `HejiHableAlu`, `HableUc2`, `Aces`

class falcor.**ToneMapper**

| Property                | Type        | Description                                                  |
|-------------------------|-------------|--------------------------------------------------------------|
| `exposureCompensation`  | `float`     | Exposure compensation (applies in manual and auto exposure). |
| `autoExposure`          | `bool`      | Enable/disable auto exposure.                                |
| `exposureValue`         | `float`     | Exposure value in manual mode.                               |
| `filmSpeed`             | `float`     | ISO film speed in manual mode.                               |
| `whiteBalance`          | `bool`      | Enable/disable white balancing.                              |
| `whitePoint`            | `float`     | White point in Kelvin.                                       |
| `operator`              | `ToneMapOp` | Tone mapping operator.                                       |
| `clamp`                 | `bool`      | Enable/disable clamping to [0..1] range.                     |

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
