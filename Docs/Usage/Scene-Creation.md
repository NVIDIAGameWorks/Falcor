### [Index](../index.md) | [Usage](./index.md) | Scene Creation

--------

# Scene Creation

Scenes are created using the `SceneBuilder` class. Falcor uses Assimp as its core model loading functionality and therefore supports all formats that Assimp does. For a full list, please see refer to the Assimp project documentation: https://github.com/assimp/assimp.

In additional to mesh geometry, Falcor also supports loading skinned animations, animations paths, cameras, and lights (point/directional/emissives) from model files.

For details on how to use a `Scene` after creation, see the [Scenes](./Scenes.md) page.

## Loading A Model From File
Basic usage for loading a model from a file:
```c++
SceneBuilder::SharedPtr pBuilder = SceneBuilder::create("path/to/model.file");
Scene::SharedPtr pScene = pBuilder->getScene();
```

Model loading functions also provide an optional parameter for creating multiple instances of a model from file. The following example loads a model from file and creates two instances at positions [-5, 0, 0] and [5, 0, 0].
```c++
SceneBuilder::InstanceMatrices instances = {
    glm::translate(float3(-5.0f, 0.0f, 0.0f)),
    glm::translate(float3(5.0f, 0.0f, 0.0f))
};

SceneBuilder::SharedPtr pBuilder = SceneBuilder::create("path/to/model.file", SceneBuilder::Flags::Default. instances);
Scene::SharedPtr pScene = pBuilder->getScene();
```

Additional scene objects not loaded as part of a model file can also be added using the `SceneBuilder` interface before the scene is created.
```c++
SceneBuilder::SharedPtr pBuilder = SceneBuilder::create("path/to/model.file");
pBuilder->addLight(PointLight::create());
Scene::SharedPtr pScene = pBuilder->getScene();
```

## Creating a Scene From Custom Geometry
In some cases, creating a scene from programmatically defined geometry is preferable. `SceneBuilder` provides a `Mesh` struct interface to pass mesh data to the builder. The builder immediately makes an internal copy of mesh data so the memory does not need to be kept valid until final scene creation.
```c++
SceneBuilder::SharedPtr pBuilder = SceneBuilder::create();

SceneBuilder::Mesh mesh;
// ... Fill out mesh struct ...
size_t meshId = pBuilder->addMesh(mesh);
size_t nodeId = pBuilder->addNode(Node( /* Instance matrix */ ));
pBuilder->addMeshInstance(nodeId, meshId);

Scene::SharedPtr pScene = pBuilder->getScene();
```

Just adding a mesh to the scene is not enough to render it. You must also define a transform node in the scene graph (which in a simple case would just be the mesh's world matrix), then add a mesh instance that associates the mesh geometry with the transform.
