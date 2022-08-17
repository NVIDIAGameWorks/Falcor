### [Index](../index.md) | [Usage](./index.md) | Custom Intersection Primitives

--------

# Custom Primitives
Custom intersection primitives are supported when using DXR. To render them, you must:
1. Define bounding boxes containing the objects.
2. Implement intersection shaders for each type of primitive you want to render.
3. Implement additional hit shaders to handle custom primitives.

## Adding Custom Primitives to the Scene
Currently, custom Primitives must be manually added through the `SceneBuilder`:
```c++
auto pSceneBuilder = SceneBuilder::create(path);
pSceneBuilder->addCustomPrimitive(0, AABB(float3(-0.5f), float3(0.5f)));
```

The interface contains two arguments: the first is an index to which intersection shader should handle this primitive (more details below), and the second is an AABB object describing the bounds of the primitive.

## Shader Setup
In Falcor, hit groups for custom intersection primitives are declared separately from those handling triangle meshes. Intersection shaders themselves are separated as well, even though DXR considered them part of the hit group.

```c++
RtProgram::Desc desc;
// Shaders for triangle meshes
desc.addShaderLibrary("Shaders.rt.slang").setRayGen("rayGen");
desc.addHitGroup(0, "scatterClosestHit", "").addMiss(0, "scatterMiss");
// Shaders for custom primitives
desc.addIntersection(0, "myIntersection");
desc.addAABBHitGroup(0, "isectScatterClosestHit", "");
```

`addIntersection()` accepts an index identifier argument similar to `hitIndex` and `missIndex` in `addHitGroup()` and `addMiss()` respectively. When adding custom primitives in the `SceneBuilder`, the `typeID` selects which intersection shader will be run, corresponding to how it's declared in `addIntersection()`. For example, every custom primitive declared with `addCustomPrimitive(0, ...)`, will be handled by the intersection shader declared as `addIntersection(0, ...")`, and so on.

**Users must also write additional hit groups (Closest Hit and/or Any Hit shaders) to handle intersection shaders, and declare them using `addAABBHitGroup()`.** This is because intersection shaders can output user-defined hit attributes to hit shaders, while hit shaders for triangle meshes always use `BuiltInTriangleIntersectionAttributes`. Using `BuiltInTriangleIntersectionAttributes` in custom intersection shaders and hit groups is valid, but you must still implement separate hit shaders at this time. **All intersection shaders and hit shaders handling custom primitives must share the same attributes struct.**

Typically, each "AABB Hit Group" should match the behavior of its corresponding "Triangle Hit Group', as the ray index (`RayContributionToHitGroupIndex`) specified in the `TraceRay()` argument in the shader affects both hit group types the same. For example, both hit groups at ray index 0 would be scatter rays, both hit groups at index 1 would be shadow rays, etc. There are no separate miss shaders for custom intersections.

## Writing Shaders
For details on the HLSL syntax for intersection shaders, see the DXR documentation [here](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#intersection-shader).

On the shader side, user-defined Custom Primitives are represented by their world-space min and max points, their typeID, and their instance index. This is all stored in the scene data structure as `StructuredBuffer<CustomPrimitiveData> customPrimitives;`.

The instance index differentiates between primitives of the same type. It is zero-based and counted separately for each type. With `typeID` and `instanceIdx`, users can uniquely identify custom primitives and look up additional data needed to render them.

To access the scene data for the current custom primitive being processed, use
```c++
gScene.customPrimitives[GeometryIndex()];
```


