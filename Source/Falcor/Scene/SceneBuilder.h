/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Scene.h"
#include "Transform.h"
#include "TriangleMesh.h"
#include "Material/MaterialTextureLoader.h"
#include "VertexAttrib.slangh"

namespace Falcor
{
    class dlldecl SceneBuilder
    {
    public:
        using SharedPtr = std::shared_ptr<SceneBuilder>;

        using MaterialList = std::vector<Material::SharedPtr>;
        using VolumeList = std::vector<Volume::SharedPtr>;
        using GridList = std::vector<Grid::SharedPtr>;
        using CameraList = std::vector<Camera::SharedPtr>;
        using LightList = std::vector<Light::SharedPtr>;
        using AnimationList = std::vector<Animation::SharedPtr>;

        /** Flags that control how the scene will be built. They can be combined together.
        */
        enum class Flags
        {
            None                        = 0x0,    ///< None
            DontMergeMaterials          = 0x1,    ///< Don't merge materials that have the same properties. Use this option to preserve the original material names.
            UseOriginalTangentSpace     = 0x2,    ///< Use the original tangent space that was loaded with the mesh. By default, we will ignore it and use MikkTSpace to generate the tangent space. We will always generate tangent space if it is missing.
            AssumeLinearSpaceTextures   = 0x4,    ///< By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.
            DontMergeMeshes             = 0x8,    ///< Preserve the original list of meshes in the scene, don't merge meshes with the same material. This flag only applies to scenes imported by 'AssimpImporter'.
            UseSpecGlossMaterials       = 0x10,   ///< Set materials to use Spec-Gloss shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.
            UseMetalRoughMaterials      = 0x20,   ///< Set materials to use Metal-Rough shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.
            NonIndexedVertices          = 0x40,   ///< Convert meshes to use non-indexed vertices. This requires more memory but may increase performance.
            Force32BitIndices           = 0x80,   ///< Force 32-bit indices for all meshes. By default, 16-bit indices are used for small meshes.
            RTDontMergeStatic           = 0x100,  ///< For raytracing, don't merge all static meshes into single pre-transformed BLAS.
            RTDontMergeDynamic          = 0x200,  ///< For raytracing, don't merge all dynamic meshes with identical transforms into single BLAS.

            Default = None
        };

        /** Mesh description.
            This struct is used by the importers to add new meshes.
            The description is then processed by the scene builder into an optimized runtime format.
            The frequency of each vertex attribute is specified individually, but note that an index list is always required.
        */
        struct Mesh
        {
            enum class AttributeFrequency
            {
                None,
                Constant,       ///< Constant value for mesh. The element count must be 1.
                Uniform,        ///< One value per face. The element count must match `faceCount`.
                Vertex,         ///< One value per vertex. The element count must match `vertexCount`.
                FaceVarying,    ///< One value per vertex per face. The element count must match `indexCount`.
            };

            template<typename T>
            struct Attribute
            {
                const T* pData = nullptr;
                AttributeFrequency frequency = AttributeFrequency::None;
            };

            std::string name;                           ///< The mesh's name.
            uint32_t faceCount = 0;                     ///< The number of primitives the mesh has.
            uint32_t vertexCount = 0;                   ///< The number of vertices the mesh has.
            uint32_t indexCount = 0;                    ///< The number of indices the mesh has.
            const uint32_t* pIndices = nullptr;         ///< Array of indices. The element count must match `indexCount`. This field is required.
            Vao::Topology topology = Vao::Topology::Undefined; ///< The primitive topology of the mesh
            Material::SharedPtr pMaterial;              ///< The mesh's material. Can't be nullptr.

            Attribute<float3> positions;                ///< Array of vertex positions. This field is required.
            Attribute<float3> normals;                  ///< Array of vertex normals. This field is required.
            Attribute<float4> tangents;                 ///< Array of vertex tangents. This field is optional. If set to nullptr, or if BuildFlags::UseOriginalTangentSpace is not set, the tangent space will be generated using MikkTSpace.
            Attribute<float2> texCrds;                  ///< Array of vertex texture coordinates. This field is optional. If set to nullptr, all texCrds will be set to (0,0).
            Attribute<uint4> boneIDs;                   ///< Array of bone IDs. This field is optional. If it's set, that means that the mesh is animated, in which case boneWeights is required.
            Attribute<float4> boneWeights;              ///< Array of bone weights. This field is optional. If it's set, that means that the mesh is animated, in which case boneIDs is required.

            bool isFrontFaceCW = false;                 ///< Indicate whether front-facing side has clockwise winding in object space.
            bool useOriginalTangentSpace = false;       ///< Indicate whether to use the original tangent space that was loaded with the mesh. By default, we will ignore it and use MikkTSpace to generate the tangent space.

            template<typename T>
            T get(const Attribute<T>& attribute, uint32_t face, uint32_t vert) const
            {
                if (attribute.pData)
                {
                    switch (attribute.frequency)
                    {
                    case AttributeFrequency::Constant:
                        return attribute.pData[0];
                    case AttributeFrequency::Uniform:
                        return attribute.pData[face];
                    case AttributeFrequency::Vertex:
                        return attribute.pData[pIndices[face * 3 + vert]];
                    case AttributeFrequency::FaceVarying:
                        return attribute.pData[face * 3 + vert];
                    default:
                        should_not_get_here();
                    }
                }
                return T{};
            }

            template<typename T>
            size_t getAttributeCount(const Attribute<T>& attribute)
            {
                switch (attribute.frequency)
                {
                case AttributeFrequency::Constant:
                    return 1;
                case AttributeFrequency::Uniform:
                    return faceCount;
                case AttributeFrequency::Vertex:
                    return vertexCount;
                case AttributeFrequency::FaceVarying:
                    return 3 * faceCount;
                default:
                    should_not_get_here();
                }
                return 0;
            }

            float3 getPosition(uint32_t face, uint32_t vert) const { return get(positions, face, vert); }
            float3 getNormal(uint32_t face, uint32_t vert) const { return get(normals, face, vert); }
            float4 getTangent(uint32_t face, uint32_t vert) const { return get(tangents, face, vert); }
            float2 getTexCrd(uint32_t face, uint32_t vert) const { return get(texCrds, face, vert); }

            struct Vertex
            {
                float3 position;
                float3 normal;
                float4 tangent;
                float2 texCrd;
                uint4 boneIDs;
                float4 boneWeights;
            };

            Vertex getVertex(uint32_t face, uint32_t vert) const
            {
                Vertex v = {};
                v.position = get(positions, face, vert);
                v.normal = get(normals, face, vert);
                v.tangent = get(tangents, face, vert);
                v.texCrd = get(texCrds, face, vert);
                v.boneIDs = get(boneIDs, face, vert);
                v.boneWeights = get(boneWeights, face, vert);
                return v;
            }

            bool hasBones() const
            {
                return boneWeights.pData || boneIDs.pData;
            }
        };

        /** Pre-processed mesh data.
            This data is formatted such that it can directly be copied
            to the global scene buffers.
        */
        struct ProcessedMesh
        {
            std::string name;
            Vao::Topology topology = Vao::Topology::Undefined;
            Material::SharedPtr pMaterial;

            uint64_t indexCount = 0;            ///< Number of indices, or zero if non-indexed.
            bool use16BitIndices = false;       ///< True if the indices are in 16-bit format.
            bool isFrontFaceCW = false;         ///< Indicate whether front-facing side has clockwise winding in object space.
            std::vector<uint32_t> indexData;    ///< Vertex indices in either 32-bit or 16-bit format packed tightly, or empty if non-indexed.
            std::vector<StaticVertexData> staticData;
            std::vector<DynamicVertexData> dynamicData;
        };

        /** Curve description.
        */
        struct Curve
        {
            template<typename T>
            struct Attribute
            {
                const T* pData = nullptr;
            };

            std::string name;                           ///< The curve's name.
            uint32_t degree = 1;                        ///< Polynomial degree of the curve; linear (1) by default.
            uint32_t vertexCount = 0;                   ///< The number of vertices.
            uint32_t indexCount = 0;                    ///< The number of indices (i.e., tube segments).
            const uint32_t* pIndices = nullptr;         ///< Array of indices. The element count must match `indexCount`. This field is required.
            Material::SharedPtr pMaterial;              ///< The curve's material. Can't be nullptr.

            Attribute<float3> positions;                ///< Array of vertex positions. This field is required.
            Attribute<float> radius;                    ///< Array of sphere radius. This field is required.
            Attribute<float3> tangents;                 ///< Array of vertex tangents. This field is required.
            Attribute<float3> normals;                  ///< Array of vertex normals. This field is optional.
            Attribute<float2> texCrds;                  ///< Array of vertex texture coordinates. This field is optional. If set to nullptr, all texCrds will be set to (0,0).
        };

        /** Pre-processed curve data.
            This data is formatted such that it can directly be copied
            to the global scene buffers.
        */
        struct ProcessedCurve
        {
            std::string name;
            Vao::Topology topology = Vao::Topology::LineStrip;
            Material::SharedPtr pMaterial;

            std::vector<uint32_t> indexData;
            std::vector<StaticCurveVertexData> staticData;
        };

        static const uint32_t kInvalidNode = Scene::kInvalidNode;

        struct Node
        {
            std::string name;
            glm::mat4 transform;
            glm::mat4 localToBindPose;   // For bones
            uint32_t parent = kInvalidNode;
        };

        using InstanceMatrices = std::vector<glm::mat4>;

        /** Create a new object
        */
        static SharedPtr create(Flags mFlags = Flags::Default);

        /** Create a new builder and import a scene/model file
            \param filename The filename to load
            \param flags The build flags
            \param instances A list of instance matrices to load. This is optional, by default a single instance will be load
            \return A new object with the imported file already initialized. If an import error occurred, a nullptr will be returned
        */
        static SharedPtr create(const std::string& filename, Flags buildFlags = Flags::Default, const InstanceMatrices& instances = InstanceMatrices());

        /** Import a scene/model file
            \param filename The filename to load
            \param instances A list of instance matrices to load. This is optional, by default a single instance will be load
            \return true if the import succeeded, otherwise false
        */
        bool import(const std::string& filename, const InstanceMatrices& instances = InstanceMatrices(), const Dictionary& dict = Dictionary());

        /** Get the scene. Make sure to add all the objects before calling this function
            \return nullptr if something went wrong, otherwise a new Scene object
        */
        Scene::SharedPtr getScene();

        /** Get the build flags
        */
        Flags getFlags() const { return mFlags; }

        /** Set the render settings.
        */
        void setRenderSettings(const Scene::RenderSettings& renderSettings) { mRenderSettings = renderSettings; }

        /** Get the render settings.
        */
        Scene::RenderSettings& getRenderSettings() { return mRenderSettings; }

        /** Get the render settings.
        */
        const Scene::RenderSettings& getRenderSettings() const { return mRenderSettings; }

        // Meshes

        /** Add a mesh.
            Throws an exception if something went wrong.
            \param mesh The mesh to add.
            \return The ID of the mesh in the scene. Note that all of the instances share the same mesh ID.
        */
        uint32_t addMesh(const Mesh& mesh);

        /** Add a triangle mesh.
            \param The triangle mesh to add.
            \param pMaterial The material to use for the mesh.
            \return The ID of the mesh in the scene.
        */
        uint32_t addTriangleMesh(const TriangleMesh::SharedPtr& pTriangleMesh, const Material::SharedPtr& pMaterial);

        /** Pre-process a mesh into the data format that is used in the global scene buffers.
            Throws an exception if something went wrong.
            \param mesh The mesh to pre-process.
            \return The pre-processed mesh.
        */
        ProcessedMesh processMesh(const Mesh& mesh) const;

        /** Add a pre-processed mesh.
            \param mesh The pre-processed mesh.
            \return The ID of the mesh in the scene. Note that all of the instances share the same mesh ID.
        */
        uint32_t addProcessedMesh(const ProcessedMesh& mesh);

        // Procedural primitives, including custom primitives, curves, etc.

        // Custom primitives

        /** Add an AABB defining a custom primitive.
            \param[in] typeID The intersection shader ID that will be run on this primitive.
            \param[in] aabb An AABB describing the bounds of the primitive.
        */
        void addCustomPrimitive(uint32_t typeID, const AABB& aabb);

        // Curves

        /** Add a curve.
            Throws an exception if something went wrong.
            \param curve The curve to add.
            \return The ID of the curve in the scene. Note that all of the instances share the same curve ID.
        */
        uint32_t addCurve(const Curve& curve);

        /** Pre-process a curve into the data format that is used in the global scene buffers.
            Throws an exception if something went wrong.
            \param curve The curve to pre-process.
            \return The pre-processed curve.
        */
        ProcessedCurve processCurve(const Curve& curve) const;

        /** Add a pre-processed curve.
            \param curve The pre-processed curve.
            \return The ID of the curve in the scene. Note that all of the instances share the same curve ID.
        */
        uint32_t addProcessedCurve(const ProcessedCurve& curve);

        // Materials

        /** Get the list of materials.
        */
        const MaterialList& getMaterials() const { return mMaterials; }

        /** Get a material by name.
            Note: This returns the first material found with a matching name.
            \param name Material name.
            \return Returns the first material with a matching name or nullptr if none was found.
        */
        Material::SharedPtr getMaterial(const std::string& name) const;

        /** Add a material.
            \param pMaterial The material.
            \return The ID of the material in the scene.
        */
        uint32_t addMaterial(const Material::SharedPtr& pMaterial);

        /** Request loading a material texture.
            \param[in] pMaterial Material to load texture into.
            \param[in] slot Slot to load texture into.
            \param[in] filename Texture filename.
        */
        void loadMaterialTexture(const Material::SharedPtr& pMaterial, Material::TextureSlot slot, const std::string& filename);

        // Volumes

        /** Get the list of volumes.
        */
        const VolumeList& getVolumes() const { return mVolumes; }

        /** Get a volume by name.
            Note: This returns the first volume found with a matching name.
            \param name Volume name.
            \return Returns the first volume with a matching name or nullptr if none was found.
        */
        Volume::SharedPtr getVolume(const std::string& name) const;

        /** Add a volume.
            \param pMaterial The volume.
            \return The ID of the volume in the scene.
        */
        uint32_t addVolume(const Volume::SharedPtr& pVolume);

        // Lights

        /** Get the list of lights.
        */
        const LightList& getLights() const { return mLights; }

        /** Add a light source
            \param pLight The light object.
            \return The light ID
        */
        uint32_t addLight(const Light::SharedPtr& pLight);

        // Environment map

        /** Get the environment map.
        */
        const EnvMap::SharedPtr& getEnvMap() const { return mpEnvMap; }

        /** Set the environment map.
            \param[in] pEnvMap Environment map. Can be nullptr.
        */
        void setEnvMap(EnvMap::SharedPtr pEnvMap) { mpEnvMap = pEnvMap; }

        // Cameras

        /** Get the list of cameras.
        */
        const CameraList& getCameras() const { return mCameras; }

        /** Add a camera.
            \param pCamera Camera to be added.
            \return The camera ID
        */
        uint32_t addCamera(const Camera::SharedPtr& pCamera);

        /** Get the selected camera.
        */
        const Camera::SharedPtr& getSelectedCamera() const { return mpSelectedCamera; }

        /** Set the selected camera.
            \param pCamera Camera to use as selected camera (needs to be added first).
        */
        void setSelectedCamera(const Camera::SharedPtr& pCamera);

        /** Get the camera speed.
        */
        float getCameraSpeed() const { return mCameraSpeed; }

        /** Set the camera speed.
        */
        void setCameraSpeed(float speed) { mCameraSpeed = speed; }

        // Animations

        /** Get the list of animations.
        */
        const AnimationList& getAnimations() const { return mAnimations; }

        /** Add an animation
            \param pAnimation The animation
        */
        void addAnimation(const Animation::SharedPtr& pAnimation);

        /** Create an animation for an animatable object.
            \param pAnimatable Animatable object.
            \param name Name of the animation.
            \param duration Duration of the animation in seconds.
            \return Returns a new animation or nullptr if an animation already exists.
        */
        Animation::SharedPtr createAnimation(Animatable::SharedPtr pAnimatable, const std::string& name, double duration);

        // Scene graph

        /** Adds a node to the graph.
            \return The node ID.
        */
        uint32_t addNode(const Node& node);

        /** Add a mesh instance to a node
        */
        void addMeshInstance(uint32_t nodeID, uint32_t meshID);

        /** Add a curve instance to a node.
        */
        void addCurveInstance(uint32_t nodeID, uint32_t curveID);

        /** Check if a scene node is animated. This check is done recursively through parent nodes.
            \return Returns true if node is animated.
        */
        bool isNodeAnimated(uint32_t nodeID) const;

        /** Set the animation interpolation mode for a given scene node. This sets the mode recursively for all parent nodes.
        */
        void setNodeInterpolationMode(uint32_t nodeID, Animation::InterpolationMode interpolationMode, bool enableWarping);

    private:
        SceneBuilder(Flags buildFlags);

        struct InternalNode : Node
        {
            InternalNode() = default;
            InternalNode(const Node& n) : Node(n) {}
            std::vector<uint32_t> children;     ///< Node IDs of all child nodes.
            std::vector<uint32_t> meshes;       ///< Mesh IDs of all meshes this node transforms.
            std::vector<uint32_t> curves;       ///< Curve IDs of all curves this node transforms.
        };

        struct MeshSpec
        {
            std::string name;
            Vao::Topology topology = Vao::Topology::Undefined;
            uint32_t materialId = 0;            ///< Global material ID.
            uint32_t staticVertexOffset = 0;    ///< Offset into the shared 'staticData' array. This is calculated in createGlobalBuffers().
            uint32_t staticVertexCount = 0;     ///< Number of static vertices.
            uint32_t dynamicVertexOffset = 0;   ///< Offset into the shared 'dynamicData' array. This is calculated in createGlobalBuffers().
            uint32_t dynamicVertexCount = 0;    ///< Number of dynamic vertices.
            uint32_t indexOffset = 0;           ///< Offset into the shared 'indexData' array. This is calculated in createGlobalBuffers().
            uint32_t indexCount = 0;            ///< Number of indices, or zero if non-indexed.
            uint32_t vertexCount = 0;           ///< Number of vertices.
            bool use16BitIndices = false;       ///< True if the indices are in 16-bit format.
            bool hasDynamicData = false;        ///< True if mesh has dynamic vertices.
            bool isStatic = false;              ///< True if mesh is non-instanced and static (not dynamic or animated).
            bool isFrontFaceCW = false;         ///< Indicate whether front-facing side has clockwise winding in object space.
            AABB boundingBox;                   ///< Mesh bounding-box in object space.
            std::vector<uint32_t> instances;    ///< Node IDs of all instances of this mesh.

            // Pre-processed vertex data.
            std::vector<uint32_t> indexData;    ///< Vertex indices in either 32-bit or 16-bit format packed tightly, or empty if non-indexed.
            std::vector<StaticVertexData> staticData;
            std::vector<DynamicVertexData> dynamicData;

            uint32_t getTriangleCount() const
            {
                assert(topology == Vao::Topology::TriangleList);
                return (indexCount > 0 ? indexCount : vertexCount) / 3;
            }

            uint32_t getIndex(const size_t i) const
            {
                assert(i < indexCount);
                return use16BitIndices ? reinterpret_cast<const uint16_t*>(indexData.data())[i] : indexData[i];
            }
        };

        // TODO: Add support for dynamic curves
        struct CurveSpec
        {
            std::string name;
            Vao::Topology topology;
            uint32_t materialId = 0;            ///< Global material ID.
            uint32_t staticVertexOffset = 0;    ///< Offset into the shared 'staticData' array. This is calculated in createCurveGlobalBuffers().
            uint32_t staticVertexCount = 0;     ///< Number of static curve vertices.
            uint32_t indexOffset = 0;           ///< Offset into the shared 'indexData' array. This is calculated in createCurveGlobalBuffers().
            uint32_t indexCount = 0;            ///< Number of indices.
            uint32_t vertexCount = 0;           ///< Number of vertices.
            uint32_t degree = 1;                ///< Polynomial degree of curve; linear (1) by default.
            std::vector<uint32_t> instances;    ///< Node IDs of all instances of this curve.

            // Pre-processed curve vertex data.
            std::vector<uint32_t> indexData;    ///< Vertex indices in 32-bit.
            std::vector<StaticCurveVertexData> staticData;
        };

        // Geometry data
        struct BuffersData
        {
            std::vector<uint32_t> indexData;                ///< Vertex indices for all meshes in either 32-bit or 16-bit format packed tightly, decided per mesh.
            std::vector<PackedStaticVertexData> staticData; ///< Vertex attributes for all meshes in packed format.
            std::vector<DynamicVertexData> dynamicData;     ///< Additional vertex attributes for dynamic (skinned) meshes.
        } mBuffersData;

        struct CurveBuffersData
        {
            std::vector<uint32_t> indexData;                ///< Vertex indices for all curves in 32-bit.
            std::vector<StaticCurveVertexData> staticData;  ///< Vertex attributes for all curves.
        } mCurveBuffersData;

        using SceneGraph = std::vector<InternalNode>;
        using MeshList = std::vector<MeshSpec>;
        using MeshGroup = Scene::MeshGroup;
        using MeshGroupList = std::vector<MeshGroup>;
        using CurveList = std::vector<CurveSpec>;

        Scene::SharedPtr mpScene;

        SceneGraph mSceneGraph;
        const Flags mFlags;
        std::string mFilename;

        Scene::RenderSettings mRenderSettings;

        MeshList mMeshes;
        MeshGroupList mMeshGroups; ///< Groups of meshes. Each group represents all the geometries in a BLAS for ray tracing.

        std::vector<ProceduralPrimitiveData> mProceduralPrimitives;           ///< GPU Data struct of procedural primitive metadata.
        std::unordered_map<uint32_t, uint32_t> mProceduralPrimInstanceCount;  ///< Map typeId to instance count.
        std::vector<AABB> mCustomPrimitiveAABBs;                              ///< User-defined custom primitive AABBs.
        CurveList mCurves;

        MaterialList mMaterials;
        std::unique_ptr<MaterialTextureLoader> mpMaterialTextureLoader;

        VolumeList mVolumes;
        GridList mGrids;
        std::unordered_map<Grid::SharedPtr, uint32_t> mGridIDs;

        CameraList mCameras;
        Camera::SharedPtr mpSelectedCamera;
        LightList mLights;
        EnvMap::SharedPtr mpEnvMap;
        std::vector<Animation::SharedPtr> mAnimations;
        float mCameraSpeed = 1.0f;

        // Mesh helpers

        /** Split a mesh by the given axis-aligned splitting plane.
            \return Pair of optional mesh IDs for the meshes on the left and right side, respectively.
        */
        std::pair<std::optional<uint32_t>, std::optional<uint32_t>> splitMesh(uint32_t meshID, const int axis, const float pos);

        void splitIndexedMesh(const MeshSpec& mesh, MeshSpec& leftMesh, MeshSpec& rightMesh, const int axis, const float pos);
        void splitNonIndexedMesh(const MeshSpec& mesh, MeshSpec& leftMesh, MeshSpec& rightMesh, const int axis, const float pos);

        // Mesh group helpers
        size_t countTriangles(const MeshGroup& meshGroup) const;
        AABB calculateBoundingBox(const MeshGroup& meshGroup) const;
        bool needsSplit(const MeshGroup& meshGroup, size_t& triangleCount) const;
        MeshGroupList splitMeshGroupSimple(MeshGroup& meshGroup) const;
        MeshGroupList splitMeshGroupMedian(MeshGroup& meshGroup) const;
        MeshGroupList splitMeshGroupMidpointMeshes(MeshGroup& meshGroup);

        // Post processing
        void removeUnusedMeshes();
        void pretransformStaticMeshes();
        void calculateMeshBoundingBoxes();
        void createMeshGroups();
        void optimizeGeometry();
        void createGlobalBuffers();
        void createCurveGlobalBuffers();
        void removeDuplicateMaterials();
        void collectVolumeGrids();
        void quantizeTexCoords();

        // Scene setup
        uint32_t createMeshData();
        void createMeshVao(uint32_t drawCount);
        void createCurveData();
        void createCurveVao();
        void mapCurvesToProceduralPrimitives(uint32_t typeID);
        void createRaytracingAABBData();
        void createNodeList();
        void createMeshBoundingBoxes();
        void calculateCurveBoundingBoxes();

        void pushProceduralPrimitive(uint32_t typeID, uint32_t instanceIdx, uint32_t AABBOffset, uint32_t AABBCount);
    };

    enum_class_operators(SceneBuilder::Flags);
}
