### [Index](../index.md) | [Usage](./index.md) | Scenes

--------

# Scenes

The Scene class manages all of the data required to describe a scene. This includes:
- Scene Graph
- Meshes
- Lights
- Cameras
- Animations (Skinned and Paths)
- Environment Map
- Acceleration Structures (DirectX Raytracing only)

Once a Scene instance has been created, no objects may be added or removed, but properties of objects such as lights and cameras may be modified.

## Cameras

Scenes can contain multiple cameras but only one is active. The first camera loaded from a model file will become the active camera. If no cameras are loaded, a default camera will be created automatically.

Scenes also provide functionality for controlling cameras with keyboard/mouse in the following modes:
- First Person
- Orbiting
- Six Degrees of Freedom

The mode defaults to First Person and can be changed using:
```c++
void Scene::setCameraController(CameraControllerType type);
```

### Controls

| Key | Description | Modes |
| --- |-------------|-------|
| **W/A/S/D** | Forward, Left, Back, Right | First Person, 6DoF |
| **Q/E** | Down, Up | 6DoF |
| **Left-Click Drag** | Turn Camera | First Person, 6DoF |
| **Left-Click Drag** | Orbit Camera | Orbiter |
| **Right-Click Drag** | Roll Camera | 6DoF |
| **Shift** | Increase movement speed | First Person, 6DoF |
| **Ctrl** | Decrease movement speed | First Person, 6DoF |


## Integration

**Note: If you are using Mogwai, most of this is handled automatically. You may skip to the Shaders section below.**

### Per-Frame Updates

If your scene contains animations, or you require the camera controller, you ***must*** ensure the following scene functions are called from your `Renderer`:
```c++
void YourRenderer::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    if(mpScene) mpScene->update(pRenderContext, gpFramework->getGlobalClock().now());
}

bool YourRenderer::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mpScene ? mpScene->onKeyEvent(mouseEvent) : false;
}

bool YourRenderer::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene ? mpScene->onMouseEvent(mouseEvent) : false;
}
```

`Scene::update()` returns a set of flags indicating which objects in the scene has changed. This is useful if your renderer or technique needs to reset values, update resources, etc based on scene changes. See `Scene::UpdateFlags` in `Scene.h` for more details.

### Acceleration Structures

The `Scene` class creates and manages raytracing acceleration structures internally. When needed, bottom-level acceleration structures are updated in `Scene::update()`, and top-level acceleration structures are updated in `Scene::raytrace()`. Raytracing resources will not be created if `Scene::raytrace()` is not called.

Acceleration structures can be updated either by recreating them entirely, or refitting the existing one. By default, for best general-case performance:
- Top-level acceleration structures are **rebuilt**
- Bottom-level acceleration structures are **refit**

This can be changed using:
```c++
void Scene::setTlasUpdateMode(UpdateMode mode);
void Scene::setBlasUpdateMode(UpdateMode mode);
```

## Rendering

The scene can be rendered using functions from the `Scene` class. There is no longer a `SceneRenderer` class.

To rasterize, use:
```c++
void Scene::rasterize(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, RenderFlags flags = RenderFlags::None);
```

To raytrace, use:
```c++
void Scene::raytrace(RenderContext* pContext, RtProgram* pProgram, const std::shared_ptr<RtProgramVars>& pVars, uint3 dispatchDims);
```

## Shaders

The GPU data structure for each scene is described in `Scene/Scene.slang`, and is accessed through the global `gScene`. There are also a few helper functions to simplify data lookup.

To access the scene in your shaders, you must import `Scene/Scene.slang` at the top of your file:
```
import Scene.Scene;
```

**IMPORTANT:** Currently, the scene GPU data structure requires defines to work correctly. These defines can be retrieved from the scene itself when compiling shaders.
```c++
SceneBuilder::SharedPtr pBuilder = SceneBuilder::create(path, flags);
Scene::SharedPtr pScene = pBuilder->getScene();

GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Shader.3d.slang", "vsMain", "psMain");
pProgram->addDefines(pScene->getSceneDefines()); // First add defines from the scene
GraphicsVars::SharedPtr pProgramVars = GraphicsVars::create(pProgram->getReflector()); // Then use
```

### Rasterization

The output of the default vertex shader includes two parameters: `instanceID`, and `materialID` which can be used to look up data for the current mesh being rendered.
See interfaces in `Scene/Scene.slang`.

For basic usage, it is not necessary to perform the lookups yourself. A helper function defined in `Scene/Raster.slang` can load and prepare data for you.

```c++
import Scene.Raster;

float4 main(VSOut vertexOut, float4 pixelCrd : SV_POSITION, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    float3 viewDir = normalize(gScene.camera.getPosition() - vOut.posW);
    let lod = ImplicitLodTextureSampler();
    ShadingData sd = prepareShadingData(vertexOut, triangleIndex, viewDir, lod);
    ...
}
```

### Raytracing

Use the helper function in `Scene/Raytracing.slang` called `getGeometryInstanceID()` to calculate the equivalent of what `instanceID` would be in raster, which can be used in the same way to look up geometry and material data.

```c++
import Scene.Raytracing;

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -WorldRayDirection(), lod);
    ...
}
```
