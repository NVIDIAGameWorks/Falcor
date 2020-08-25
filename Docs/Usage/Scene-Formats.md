### [Index](../index.md) | [Usage](./index.md) | Scene Formats

--------

# Scene Formats

Falcor uses [Assimp](https://github.com/assimp/assimp) as its core asset loader and can load all file formats Assimp supports by default.

From assets, Falcor will import:
- Scene Graph
- Meshes
- Materials
    - Diffuse Texture
        - Metal-Rough Shading Model (Default)
            - RGB: Base Color
        - Spec-Gloss Shading Model (Default for OBJ only)
            - RGB: Diffuse Color
    - Specular Parameters Texture
        - Metal-Rough Shading Model (Default)
            - R: Occlusion
            - G: Roughness
            - B: Metallic
        - Spec-Gloss Shading Model (Default for OBJ only)
            - RGB: Specular Color
            - A: Glossiness
    - Normals Texture
    - Occlusion Texture (Used for Spec-Gloss shading model only)
    - Emissive Color/Texture
- Cameras
- Point lights
- Directional lights
- Keyframe animations
- Skinned animations


## Python Scene Files

You can also leverage Falcor's scripting system to set values in the scene on load that are not supported by standard file formats. These are also written in Python (using `.pyscene` file extension), but are formatted differently than normal Falcor scripts.

### Usage

The first line must be a Python comment containing only a path to the base asset to be loaded. File paths in Python scene files may be relative to the file itself, in addition to standard Falcor data directories.

```python
# BistroInterior.fbx
```

The asset will be loaded and will be bound to an object called `scene`. Through this object, you have access to any script bindings accessible through Scenes. See the [scripting documentation](./Scripting.md) for a full list of functions and properties.

Example:

```python
# BistroInterior.fbx
# Line above loads BistroInterior.fbx from the same folder as the script

scene.setEnvMap("BistroInterior.hdr") # Set the environment map to "BistroInterior.hdr" located in the same folder as the script

bottle_wine = scene.getMaterial("TransparentGlassWine") # Get a material from the scene by name

# Set material properties
bottle_wine.indexOfRefraction = 1.55
bottle_wine.specularTransmission = 1
bottle_wine.doubleSided = True
bottle_wine.nestedPriority = 2
bottle_wine.volumeAbsorption = float3(0.7143, 1.1688, 1.7169)
```
