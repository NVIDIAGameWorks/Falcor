### [Index](./index.md) | Known Issues

--------

# Known Issues

## USD Support

Falcor's USD importer implements a subset of the full [USD specification](https://graphics.pixar.com/usd/release/index.html). The following provides an overview of some of the USD features not supported by the Falcor USD importer.

### General
- `UsdPhysics` is unsupported.
- `UsdVariantSets` do not function properly.
- Animation of camera parameters (aperture size, etc.) is not supported.
- Only per-vertex skinning data is supported. In particular, constant skinning data is not supported.
- A maximum of 4 bones per vertex are supported. Additional bones will be ignored, and a warning issued.

### UsdGeom

- `UsdGeomCapsule`, `UsdGeomCone`, `UsdGeomCube`, `UsdGeomCylinder`, `UsdGeomHermiteCurves`, `UsdGeomNurbsCurves`, `UsdGeomNurbsPatch`, and `UsdGeomPoints` are unsupported
- Instancing of `UsdGeomBasisCurves` is not supported.
- `DisplayColor` interpolation modes are ignored.
- Pivot transformations are not supported.

### UsdLux
- `UsdLuxShapingAPI` and `UsdLuxShadowAPI` are not supported.
- `UsdLuxCylinderLight` and `UsdLuxPortalLight` are not supported.
- Instancing of lights is not supported.

### UsdPreviewSurface

- Only the `metallic` workflow is supported. A warning will be issued when `UsdPreviewSurface` shaders that use the `specular` workflow are encountered.
- The `clearcoat`, `clearcloatRoughness`, `occlusion` and `displacement` inputs of `UsdPreviewSurface` are not supported.
- The `bias`, `scale`, `wrapS`, and `wrapT` inputs of `UsdUVTexture` are not supported.
- `Transform2d` nodes are not supported.
