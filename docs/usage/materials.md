### [Index](../index.md) | [Usage](./index.md) | Materials

--------

# Materials

Starting with Falcor 5.0 there is a new material system that allows creation and rendering of different materials
types at runtime. In previous versions, a fixed diffuse+GGX material model was assumed.

All materials and their resources are managed by the `MaterialSystem` C++ class and matching Slang module.
The material system is owned by `Scene` object and bound together with its other resources when rendering.

In order to access the material system in shader code, the following module must be imported:

```c++
import Scene.Shading;
```

## Data layout

For efficient access, each material is described by a data blob of fixed size (currently 128B).
The data blob consists of a header (see `MaterialHeader` declared in `Falcor/Scene/Material/MaterialData.slang`)
followed by a data payload. The header always exists and holds material type ID and auxiliary flags etc.

The format of the data payload is opaque and depends on the material type. If a material needs more data
than what fits in the payload, it can store additional sideband data in a buffer.

All resources (textures, samplers, buffers) are accessed via bindless resource IDs, where the actual GPU
resources are managed by `MaterialSystem` and the IDs are allocated when resources are added.
These bindless IDs are stored as part of the data payload, so that a material is self-contained and fully
desribed by its data blob.

## Material class (host side)

On the host side, all materials are derived from the `Material` base class declared in `Scene/Material/Material.h`.

In order to add a new material, a new class should be added that inherits from `Material` and implements
its pure virtual member functions. The most important ones are:

```c++
// Called once per frame to prepare the material for rendering.
Material::UpdateFlags Material::update(MaterialSystem* pOwner);

// Returns the material data blob.
MaterialDataBlob Material::getDataBlob() const;
```

The base class holds the `MaterialHeader` struct and the derived material class is responsible
for holding the data payload. The `getDataBlob()` returns the final data blob, which will be uploaded to the
GPU by the material system for access on the shader side.

## Python bindings

To allow creation of materials and setting of the parameters from Python scripts (including `.pyscene` files),
each material class is expected to export Python bindings.

These bindings are defined in the `FALCOR_SCRIPT_BINDING` block, usually placed at the bottom of the material's `.cpp` file.

Example usage:

```c++
glass = StandardMaterial("WindowGlass")
glass.roughness = 0
glass.metallic = 0
glass.indexOfRefraction = 1.52
glass.specularTransmission = 1
glass.doubleSided = True
glass.nestedPriority = 2
glass.volumeAbsorption = float3(2.0, 1.0, 1.5)
```

For more examples of how material's are created from Python, refer to the test scenes in `media/TestScenes/`
(this directory is automatically fetched during project setup).

## Material module (shader side)

On the shader side, each material class has a corresponding Slang module stored in `Falcor/Rendering/Materials/`.
These modules implement the `IMaterial` Slang interface (see `Rendering/Materials/IMaterial.slang`).

The main purpose of the material module is to:
1. hold the material data, and
2. hold the code for setting up a BSDF at a shading point.

The latter is referred to as "pattern generation", which may involve sampling textures, evaluating
procedural functions, and any other setup needed for shading.

The first data field in the material module has to be the material header. This should be followed by the
material payload as declared for the material type. For example, the standard material is declared:

```c++
struct StandardMaterial : IMaterial
{
    MaterialHeader header;
    BasicMaterialData data;
    ...
};
```

An instance of the material is created by calling the material system as follows:

```c++
IMaterial material = gScene.materials.getMaterial(materialID);
```

Internally, this function accesses the material header to fetch the material type, and then it calls
Slang's `createDynamicObject<..>` function to create an instance of the right type.
The opaque material data blob is cast to the data types used in the material module, so its fields are
directly accessible internally in the material module.

## BSDF module (shader side)

Each material module has an associated BSDF type, which implements the `IBSDF` Slang interface.
For example, `StandardMaterial` has an associated `StandardBSDF` type.

An instance of the BSDF type is created for a specific shading point in the scene, and it exposes
interfaces for evaluating and sampling the BSDF at that point.
The `IBSDF` interface also has functions for querying additional BSDF properties
at the shading point, such as albedo, emission, etc.

A BSDF instance is created by calling the following function on the material:

```c++
ShadingData sd = ...       // struct describing the shading point
ITextureSampler lod = ...  // method for texture level-of-detail computation

IBSDF bsdf = material.setupBSDF(gScene.materials, sd, lod);
```

Internally, the `setupBSDF` function accesses the material system to fetch/evaluate all resources needed at the
shading point. It returns an instance of the material's associated BSDF type,
which the caller can then use to evaluate or sample the BSDF at the shading point.

Since creating the material followed by instantiating the BSDF is very common,
there is a convenience function `getBSDF()` on `MaterialSystem` that does both operations in one step:

```c++
IBSDF bsdf = gScene.materials.getBSDF(sd, lod);
```

In the above interfaces, a `ShadingData` struct is needed to describe the shading point.
This is generated at a hit point by calling the `prepareShadingData()` function.
This function is responsible for setting up the shading frame (normal, tangent, bitangent)
including evaluating normal mapping and material opacity for alpha testing.

In addition to this, a `ITextureSampler` instance is needed to describe how textures should
be sampled (if there are any). The caller is responsible for deciding this based on which
method for texture LOD it is using (e.g. ray cones, ray differentials, fixed mip level, etc).
See available choices in `Scene/Material/TextureSampler.slang`).
