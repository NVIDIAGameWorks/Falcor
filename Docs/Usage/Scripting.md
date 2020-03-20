### [Index](../index.md) | [Usage](./index.md) | Scripting

--------

# Mogwai Scripting Reference

## Sample API

#### Global functions

| Function                      | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| `loadRenderPassLibrary(name)` | Load a render pass library.                                              |
| `renderFrame()`               | Render a frame. If the clock is not paused, it will advance by one tick. |
| `exit()`                      | Terminate the application.                                               |
| `cls()`                       | Clear the console.                                                       |

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

| Method                           | Description                                                     |
|----------------------------------|-----------------------------------------------------------------|
| `script(filename)`               | Run a script.                                                   |
| `loadScene(filename)`            | Load a scene.                                                   |
| `scene()`                        | Get the scene.                                                  |
| `envMap(filename)`               | Load an environment map (temporary workaround).                 |
| `saveConfig(filename)`           | Save the current state to a config file.                        |
| `addGraph(graph)`                | Add a render graph.                                             |
| `removeGraph(graph)`             | Remove a render graph. `graph` can be a render graph or a name. |
| `graph(name)`                    | Get a render graph by name.                                     |
| `activeGraph()`                  | Get the currently active render graph.                          |
| `resizeSwapChain(width, height)` | Resize the window/swapchain.                                    |
| `ui(show)`                       | Show/hide the UI.                                               |

#### Clock

class falcor.**Clock**

| Method                 | Description                        |
|------------------------|------------------------------------|
| `now()`                | Get the current time in _seconds_. |
| `now(seconds)`         | Set the current time in _seconds_. |
| `frame()`              | Get the current frame number.      |
| `frame(frameID)`       | Set the current frame number.      |
| `pause()`              | Pause the clock.                   |
| `play()`               | Resume the clock.                  |
| `stop()`               | Stop the clock (pause and reset)   |
| `step(frames = 1)`     | Stop forward or backward in time.  |
| `framerate()`          | Get the framerate.                 |
| `framerate(framerate)` | Set the framerate.                 |
| `exitTime(seconds)`    | Exit at specific time.             |
| `exitFrame(n)`	     | Exit at specific frame.            |

#### FrameCapture

The frame capture will always dump the marked graph output. You can use `graph.markOutput()` and `graph.unmarkOutput()` to control which outputs to dump.

The frame counter starts at zero in Falcor (it starts by default at one in Maya, so the frame numbers may be offset by one frame).

By default, the captures frames are stored to the executable directory. This can be changed by `fc.outputDir()`.

**Note:** The frame counter is not advanced when time is paused. If you capture with time paused, the captured frame will be overwritten for every rendered frame. The workaround is to change the base filename between captures with `fc.capture()`, see example below.


class falcor.**FrameCapture**

| Method                  | Description                                                                |
|-------------------------|----------------------------------------------------------------------------|
| `outputDir(dir)`        | Set the output directory.                                                  |
| `baseFilename(name)`    | Set the base filename. The frameID and output name will be appended to it. |
| `capture()`             | Capture the current frame.                                                 |
| `frames(graph, frames)` | Set a list of frames to capture for the given graph.                       |
| `print()`               | Print the requested frames to capture for all available graphs.            |
| `print(graph)`          | Print the requested frames to capture for the specified graph.             |
| `ui(show)`              | Show/hide the UI.                                                          |

**Example:** *Capture list of frames with clock running and then exit*
```python
t.exitFrame(101)
fc.outputDir("../../../Output")
fc.baseFilename("Mogwai")
fc.frames(m.activeGraph(), [20, 50, 100])
```

**Example:** *Capture frames with clock paused and then exit*
```python
t.pause()
fc.outputDir("../../../Output")

frames = [20, 50, 100]
for i in range(101):
    renderFrame()
    if i in frames:
        fc.baseFilename(f"Mogwai-{i:04d}")
        fc.capture()
exit()
```


#### VideoCapture
The video capture will always capture the marked graph output. You can use `graph.markOutput()` and `graph.unmarkOutput()` to control which outputs to dump.

enum falcor.**Codec**

`Raw`, `H264`, `HVEC`, `MPEG2`, `MPEG4`

class falcor.**VideoCapture**

| Method                  | Description                                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| `outputDir(dir)`        | Set the output directory.                                                                             |
| `baseFilename(name)`    | Set the base filename. The output name will be appended to it.                                        |
| `codec(codec)`          | Set the codec (`Raw`, `H264`, `HVEC`, `MPEG2`, `MPEG4`).                                              |
| `codec()`               | Get the codec.                                                                                        |
| `fps(fps)`              | Set tha framerate.                                                                                    |
| `fps()`                 | Get the framerate.                                                                                    |
| `bitrate(bitrate)`      | Set the bitrate in Mpbs.                                                                              |
| `bitrate()`             | Get the bitrate.                                                                                      |
| `gopSize(size)`         | Set the GOP size.                                                                                     |
| `gopSize()`             | Get the GOP size.                                                                                     |
| `ranges(graph, ranges)` | Set a list of frame ranges to capture for a given graph. `ranges` is a list of `(start, end)` tuples. |
| `print()`               | Print the requested frames to capture for all available graphs.                                       |
| `print(graph)`          | Print the requested frames to capture for the specified graph.                                        |
| `ui(show)`              | Show/hide the UI.                                                                                     |

Example:
```python
# Video Capture
vc.outputDir(".")
vc.baseFilename("Mogwai")
vc.codec(Codec.H264)
vc.fps(60)
vc.bitrate(4.0)
vc.gopSize(10)
g = m.activeGraph()
vc.ranges(g, [[30, 300]])
```

#### TimingCapture

class falcor.**TimingCapture**

| Method                       | Description                                      |
|------------------------------|--------------------------------------------------|
| `captureFrameTime(filename)` | Start writing frame times to the given filename. |

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

| Method                         | Description                                                                        |
|--------------------------------|------------------------------------------------------------------------------------|
| `RenderGraph(name)`            | Create a new render graph.                                                         |
| `name()`                       | Get the name.                                                                      |
| `name(name)`                   | Set the name.                                                                      |
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

| Method                   | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| `RenderPass(name, dict)` | Create a new render pass with configuration options in `dict`. |

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

#### Scene

class falcor.**Scene**

| Method                         | Description                       |
|--------------------------------|-----------------------------------|
| `animate(enable)`              | Enable/disable scene animations.  |
| `animateCamera(enabled)`       | Enable/disable camera animations. |
| `animateLight(index, enabled)` | Enable/disable light animations.  |
| `camera()`                     | Return the camera.                |
| `light(index)`                 | Return a light by index.          |
| `light(name)`                  | Return a light by name.           |
| `material(index)`              | Return a material by index.       |
| `material(name)`               | Return a material by name.        |
| `removeViewpoint(index)`       | Remove a viewpoint.               |
| `viewpoint()`                  | Create a new viewpoint.           |
| `viewpoint(index)`             | Load a viewpoint.                 |

#### Camera

class falcor.**Camera**

| Property         | Type    | Description                                  |
|------------------|---------|----------------------------------------------|
| `name`           | `str`   | Name of the camera (readonly).               |
| `aspectRatio`    | `float` | Image aspect ratio.                          |
| `focalLength`    | `float` | Focal length in millimeters.                 |
| `frameHeight`    | `float` | Frame height in millimeters.                 |
| `focalDistance`  | `float` | Focal distance in scene units.               |
| `apertureRadius` | `float` | Apeture radius in scene units.               |
| `shutterSpeed`   | `float` | Shutter speed  in seconds (not implemented). |
| `ISOSpeed`       | `float` | Film speed (not implemented).                |
| `nearPlane`      | `float` | Near plane distance in scene units.          |
| `farPlane`       | `float` | Far plane distance in scene units.           |
| `position`       | `vec3`  | Camera position in world space.              |
| `target`         | `vec3`  | Camera target in world space.                |
| `up`             | `vec`   | Camera up vector in world space.             |

#### Material

class falcor.**Material**

| Property               | Type    | Description                                           |
|------------------------|---------|-------------------------------------------------------|
| `name`                 | `str`   | Name of the material (readonly).                      |
| `baseColor`            | `vec4`  | Base color (linear RGB) and opacity.                  |
| `specularParams`       | `vec4`  | Specular parameters (occlusion, roughness, metallic). |
| `specularTransmission` | `float` | Specular transmission (0 = opaque, 1 = transparent).  |
| `indexOfRefraction`    | `float` | Index of refraction.                                  |
| `emissiveColor`        | `vec3`  | Emissive color (linear RGB).                          |
| `emissiveFactor`       | `float` | Multiplier for emissive color.                        |
| `alphaMode`            | `int`   | Alpha mode (0 = opaque, 1 = masked).                  |
| `alphaThreshold`       | `float` | Alpha masking threshold (0-1).                        |
| `doubleSided`          | `bool`  | Enable double sided rendering.                        |
| `nestedPriority`       | `int`   | Nested priority for nested dielectrics.               |

#### Light

class falcor.**Light**

| Property    | Type    | Description                   |
|-------------|---------|-------------------------------|
| `name`      | `str`   | Name of the light (readonly). |
| `color`     | `vec3`  | Color of the light.           |
| `intensity` | `float` | Intensity of the light.       |

class falcor.**DirectionalLight**

| Property    | Type    | Description                            |
|-------------|---------|----------------------------------------|
| `name`      | `str`   | Name of the light (readonly).          |
| `color`     | `vec3`  | Color of the light.                    |
| `intensity` | `float` | Intensity of the light.                |
| `direction` | `vec3`  | Direction of the light in world space. |

class falcor.**PointLight**

| Property        | Type    | Description                            |
|-----------------|---------|----------------------------------------|
| `name`          | `str`   | Name of the light (readonly).          |
| `color`         | `vec3`  | Color of the light.                    |
| `intensity`     | `float` | Intensity of the light.                |
| `position`      | `vec3`  | Position of the light in world space.  |
| `direction`     | `vec3`  | Direction of the light in world space. |
| `openingAngle`  | `float` | Opening half-angle in radians.         |
| `penumbraAngle` | `float` | Penumbra half-angle in radians.        |

class falcor.**AnalyticAreaLight**

| Property    | Type    | Description                   |
|-------------|---------|-------------------------------|
| `name`      | `str`   | Name of the light (readonly). |
| `color`     | `vec3`  | Color of the light.           |
| `intensity` | `float` | Intensity of the light.       |

### Render Passes

#### GaussianBlur

class falcor.**GaussianBlur**

| Property      | Type    | Description             |
|---------------|---------|-------------------------|
| `kernelWidth` | `int`   | Kernel width in pixels. |
| `sigma`       | `float` | Sigma of gaussian.      |

#### ToneMapper

enum falcor.**ToneMapOp**

`Linear`, `Reinhard`, `ReinhardModified`, `HejiHableAlu`, `HableUc2`, `Aces`

class falcor.**ToneMapper**

| Property                | Type        | Description                                                  |
|-------------------------|-------------|--------------------------------------------------------------|
| `exposureCompenstation` | `float`     | Exposure compensation (applies in manual and auto exposure). |
| `autoExposure`          | `bool`      | Enable/disable auto exposure.                                |
| `exposureValue`         | `float`     | Exposure value in manual mode.                               |
| `filmSpeed`             | `float`     | ISO film speed in manual mode.                               |
| `whiteBalance`          | `bool`      | Enable/disable white balancing.                              |
| `whitePoint`            | `float`     | White point in Kelvin.                                       |
| `operator`              | `ToneMapOp` | Tone mapping operator.                                       |
| `clamp`                 | `bool`      | Enable/disable clamping to [0..1] range.                     |

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
| `mapSize`            | `uvec2` |             |
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
