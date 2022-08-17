/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "SceneCache.h"
#include "SceneIDs.h"
#include "Transform.h"
#include "TriangleMesh.h"
#include "VertexAttrib.slangh"
#include "SceneTypes.slang"
#include "Material/MaterialTextureLoader.h"

#include "Core/Macros.h"
#include "Core/API/VAO.h"
#include "Utils/Math/AABB.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include "Utils/Scripting/Dictionary.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    class FALCOR_API SceneBuilder
    {
    public:
        using SharedPtr = std::shared_ptr<SceneBuilder>;

        /** Flags that control how the scene will be built. They can be combined together.
        */
        enum class Flags
        {
            None                            = 0x0,      ///< None
            DontMergeMaterials              = 0x1,      ///< Don't merge materials that have the same properties. Use this option to preserve the original material names.
            UseOriginalTangentSpace         = 0x2,      ///< Use the original tangent space that was loaded with the mesh. By default, we will ignore it and use MikkTSpace to generate the tangent space. We will always generate tangent space if it is missing.
            AssumeLinearSpaceTextures       = 0x4,      ///< By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.
            DontMergeMeshes                 = 0x8,      ///< Preserve the original list of meshes in the scene, don't merge meshes with the same material. This flag only applies to scenes imported by 'AssimpImporter'.
            UseSpecGlossMaterials           = 0x10,     ///< Set materials to use Spec-Gloss shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.
            UseMetalRoughMaterials          = 0x20,     ///< Set materials to use Metal-Rough shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.
            NonIndexedVertices              = 0x40,     ///< Convert meshes to use non-indexed vertices. This requires more memory but may increase performance.
            Force32BitIndices               = 0x80,     ///< Force 32-bit indices for all meshes. By default, 16-bit indices are used for small meshes.
            RTDontMergeStatic               = 0x100,    ///< For raytracing, don't merge all static non-instanced meshes into single pre-transformed BLAS.
            RTDontMergeDynamic              = 0x200,    ///< For raytracing, don't merge dynamic non-instanced meshes with identical transforms into single BLAS.
            RTDontMergeInstanced            = 0x400,    ///< For raytracing, don't merge instanced meshes with identical instances into single BLAS.
            FlattenStaticMeshInstances      = 0x800,    ///< Flatten static mesh instances by duplicating mesh data and composing transformations. Animated instances are not affected. Can lead to a large increase in memory use.
            DontOptimizeGraph               = 0x1000,   ///< Don't optimize the scene graph to remove unnecessary nodes.
            DontOptimizeMaterials           = 0x2000,   ///< Don't optimize materials by removing constant textures. The optimizations are lossless so should generally be enabled.
            DontUseDisplacement             = 0x4000,   ///< Don't use displacement mapping.
            UseCompressedHitInfo            = 0x8000,   ///< Use compressed hit info (on scenes with triangle meshes only).
            TessellateCurvesIntoPolyTubes   = 0x10000,  ///< Tessellate curves into poly-tubes (the default is linear swept spheres).

            UseCache                        = 0x10000000, ///< Enable scene caching. This caches the runtime scene representation on disk to reduce load time.
            RebuildCache                    = 0x20000000, ///< Rebuild scene cache.

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
            Attribute<float> curveRadii;                ///< Array of vertex curve radii. This field is optional.
            Attribute<uint4> boneIDs;                   ///< Array of bone IDs. This field is optional. If it's set, that means that the mesh is animated, in which case boneWeights is required.
            Attribute<float4> boneWeights;              ///< Array of bone weights. This field is optional. If it's set, that means that the mesh is animated, in which case boneIDs is required.

            bool isFrontFaceCW = false;                 ///< Indicate whether front-facing side has clockwise winding in object space.
            bool useOriginalTangentSpace = false;       ///< Indicate whether to use the original tangent space that was loaded with the mesh. By default, we will ignore it and use MikkTSpace to generate the tangent space.
            bool mergeDuplicateVertices = true;         ///< Indicate whether to merge identical vertices and adjust indices.
            NodeID skeletonNodeId{ NodeID::Invalid() }; ///< For skinned meshes, the node ID of the skeleton's world transform. If invalid, the skeleton is based on the mesh's own world position (Assimp behavior pre-multiplies instance transform).

            template<typename T>
            uint32_t getAttributeIndex(const Attribute<T>& attribute, uint32_t face, uint32_t vert) const
            {
                switch (attribute.frequency)
                {
                case AttributeFrequency::Constant:
                    return 0;
                case AttributeFrequency::Uniform:
                    return face;
                case AttributeFrequency::Vertex:
                    return pIndices[face * 3 + vert];
                case AttributeFrequency::FaceVarying:
                    return face * 3 + vert;
                default:
                    FALCOR_UNREACHABLE();
                }
                return Scene::kInvalidAttributeIndex;
            }

            template<typename T>
            T get(const Attribute<T>& attribute, uint32_t index) const
            {
                if (attribute.pData)
                {
                    return attribute.pData[index];
                }
                return T{};
            }

            template<typename T>
            T get(const Attribute<T>& attribute, uint32_t face, uint32_t vert) const
            {
                if (attribute.pData)
                {
                    return get(attribute, getAttributeIndex(attribute, face, vert));
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
                    FALCOR_UNREACHABLE();
                }
                return 0;
            }

            float3 getPosition(uint32_t face, uint32_t vert) const { return get(positions, face, vert); }
            float3 getNormal(uint32_t face, uint32_t vert) const { return get(normals, face, vert); }
            float4 getTangent(uint32_t face, uint32_t vert) const { return get(tangents, face, vert); }
            float2 getTexCrd(uint32_t face, uint32_t vert) const { return get(texCrds, face, vert); }
            float getCurveRadii(uint32_t face, uint32_t vert) const { return get(curveRadii, face, vert); }

            struct Vertex
            {
                float3 position;
                float3 normal;
                float4 tangent;
                float2 texCrd;
                float curveRadius;
                uint4 boneIDs;
                float4 boneWeights;
            };

            struct VertexAttributeIndices
            {
                uint32_t positionIdx;
                uint32_t normalIdx;
                uint32_t tangentIdx;
                uint32_t texCrdIdx;
                uint32_t curveRadiusIdx;
                uint32_t boneIDsIdx;
                uint32_t boneWeightsIdx;
            };

            Vertex getVertex(uint32_t face, uint32_t vert) const
            {
                Vertex v = {};
                v.position = get(positions, face, vert);
                v.normal = get(normals, face, vert);
                v.tangent = get(tangents, face, vert);
                v.texCrd = get(texCrds, face, vert);
                v.curveRadius = get(curveRadii, face, vert);
                v.boneIDs = get(boneIDs, face, vert);
                v.boneWeights = get(boneWeights, face, vert);
                return v;
            }

            Vertex getVertex(const VertexAttributeIndices& attributeIndices)
            {
                Vertex v = {};
                v.position = get(positions, attributeIndices.positionIdx);
                v.normal = get(normals, attributeIndices.normalIdx);
                v.tangent = get(tangents, attributeIndices.tangentIdx);
                v.texCrd = get(texCrds, attributeIndices.texCrdIdx);
                v.curveRadius = get(curveRadii, attributeIndices.curveRadiusIdx);
                v.boneIDs = get(boneIDs, attributeIndices.boneIDsIdx);
                v.boneWeights = get(boneWeights, attributeIndices.boneWeightsIdx);
                return v;
            }

            VertexAttributeIndices getAttributeIndices(uint32_t face, uint32_t vert)
            {
                VertexAttributeIndices v = {};
                v.positionIdx = getAttributeIndex(positions, face, vert);
                v.normalIdx = getAttributeIndex(normals, face, vert);
                v.tangentIdx = getAttributeIndex(tangents, face, vert);
                v.texCrdIdx = getAttributeIndex(texCrds, face, vert);
                v.curveRadiusIdx = getAttributeIndex(curveRadii, face, vert);
                v.boneIDsIdx = getAttributeIndex(boneIDs, face, vert);
                v.boneWeightsIdx = getAttributeIndex(boneWeights, face, vert);
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
            NodeID skeletonNodeId{ NodeID::Invalid() }; ///< Forwarded from Mesh struct.

            uint64_t indexCount = 0;            ///< Number of indices, or zero if non-indexed.
            bool use16BitIndices = false;       ///< True if the indices are in 16-bit format.
            bool isFrontFaceCW = false;         ///< Indicate whether front-facing side has clockwise winding in object space.
            std::vector<uint32_t> indexData;    ///< Vertex indices in either 32-bit or 16-bit format packed tightly, or empty if non-indexed.
            std::vector<StaticVertexData> staticData;
            std::vector<SkinningVertexData> skinningData;
        };

        using MeshAttributeIndices = std::vector<Mesh::VertexAttributeIndices>;

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

        struct Node
        {
            std::string name;
            rmcv::mat4 transform;
            rmcv::mat4 meshBind;          // For skinned meshes. World transform at bind time.
            rmcv::mat4 localToBindPose;   // For bones. Inverse bind transform.
            NodeID parent{ NodeID::Invalid() };
        };

        using InstanceMatrices = std::vector<rmcv::mat4>;

        /** Create a new object
        */
        static SharedPtr create(Flags mFlags = Flags::Default);

        /** Create a new builder and import a scene/model file
            \param path The file path to load
            \param flags The build flags
            \param instances A list of instance matrices to load. This is optional, by default a single instance will be load
            \return A new object with the imported file already initialized, or throws an ImporterError if importing went wrong.
        */
        static SharedPtr create(const std::filesystem::path& path, Flags buildFlags = Flags::Default, const InstanceMatrices& instances = InstanceMatrices());

        /** Import a scene/model file
            \param path The file path to load
            \param instances A list of instance matrices to load. This is optional, by default a single instance will be load
            Throws an ImporterError if something went wrong.
        */
        void import(const std::filesystem::path& path, const InstanceMatrices& instances = InstanceMatrices(), const Dictionary& dict = Dictionary());

        /** Get the scene. Make sure to add all the objects before calling this function
            \return nullptr if something went wrong, otherwise a new Scene object
        */
        Scene::SharedPtr getScene();

        /** Get the build flags
        */
        Flags getFlags() const { return mFlags; }

        /** Set the render settings.
        */
        void setRenderSettings(const Scene::RenderSettings& renderSettings) { mSceneData.renderSettings = renderSettings; }

        /** Get the render settings.
        */
        Scene::RenderSettings& getRenderSettings() { return mSceneData.renderSettings; }

        /** Get the render settings.
        */
        const Scene::RenderSettings& getRenderSettings() const { return mSceneData.renderSettings; }

        /** Set the metadata.
        */
        void setMetadata(const Scene::Metadata& metadata) { mSceneData.metadata = metadata; }

        /** Get the metadata.
        */
        Scene::Metadata& getMetadata() { return mSceneData.metadata; }

        /** Get the metadata.
        */
        const Scene::Metadata& getMetadata() const { return mSceneData.metadata; }

        // Meshes

        /** Add a mesh.
            Throws an exception if something went wrong.
            \param mesh The mesh to add.
            \return The ID of the mesh in the scene. Note that all of the instances share the same mesh ID.
        */
        MeshID addMesh(const Mesh& mesh);

        /** Add a triangle mesh.
            \param The triangle mesh to add.
            \param pMaterial The material to use for the mesh.
            \return The ID of the mesh in the scene.
        */
        MeshID addTriangleMesh(const TriangleMesh::SharedPtr& pTriangleMesh, const Material::SharedPtr& pMaterial);

        /** Pre-process a mesh into the data format that is used in the global scene buffers.
            Throws an exception if something went wrong.
            \param mesh The mesh to pre-process.
            \param pAttributeIndices Optional. If specified, the attribute indices used to create the final mesh vertices will be saved here.
            \return The pre-processed mesh.
        */
        ProcessedMesh processMesh(const Mesh& mesh, MeshAttributeIndices* pAttributeIndices = nullptr) const;

        /** Generate tangents for a mesh.
            \param mesh The mesh to generate tangents for. If successful, the tangent attribute on the mesh will be set to the output vector.
            \param tangents Output for generated tangents.
        */
        void generateTangents(Mesh& mesh, std::vector<float4>& tangents) const;

        /** Add a pre-processed mesh.
            \param mesh The pre-processed mesh.
            \return The ID of the mesh in the scene. Note that all of the instances share the same mesh ID.
        */
        MeshID addProcessedMesh(const ProcessedMesh& mesh);

        /** Set mesh vertex cache for animation.
            \param[in] cachedCurves The mesh vertex cache data (will be moved from).
        */
        void setCachedMeshes(std::vector<CachedMesh>&& cachedMeshes);

        // Custom primitives

        /** Add an AABB defining a custom primitive.
            \param[in] userID User-defined ID that can be used to identify different sub-types of custom primitives.
            \param[in] aabb An AABB describing the bounds of the primitive.
        */
        void addCustomPrimitive(uint32_t userID, const AABB& aabb);

        // Curves

        /** Add a curve.
            Throws an exception if something went wrong.
            \param curve The curve to add.
            \return The ID of the curve in the scene. Note that all of the instances share the same curve ID.
        */
        CurveID addCurve(const Curve& curve);

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
        CurveID addProcessedCurve(const ProcessedCurve& curve);

        /** Set curve vertex cache for animation.
            \param[in] cachedCurves The dynamic curve vertex cache data.
        */
        void setCachedCurves(std::vector<CachedCurve>&& cachedCurves) { mSceneData.cachedCurves = std::move(cachedCurves); }

        // SDFs

        /** Add an SDF grid.
            \param pSDFGrid The SDF grid.
            \param pMaterial The material to be used by this SDF grid.
            \return The ID of the SDG grid desc in the scene.
        */
        SdfDescID addSDFGrid(const SDFGrid::SharedPtr& pSDFGrid, const Material::SharedPtr& pMaterial);

        // Materials

        /** Get the list of materials.
        */
        const std::vector<Material::SharedPtr>& getMaterials() const { return mSceneData.pMaterials->getMaterials(); }

        /** Get a material by name.
            Note: This returns the first material found with a matching name.
            \param name Material name.
            \return Returns the first material with a matching name or nullptr if none was found.
        */
        Material::SharedPtr getMaterial(const std::string& name) const { return mSceneData.pMaterials->getMaterialByName(name); }

        /** Add a material.
            \param pMaterial The material.
            \return The ID of the material in the scene.
        */
        MaterialID addMaterial(const Material::SharedPtr& pMaterial);

        /** Request loading a material texture.
            \param[in] pMaterial Material to load texture into.
            \param[in] slot Slot to load texture into.
            \param[in] path Texture file path.
        */
        void loadMaterialTexture(const Material::SharedPtr& pMaterial, Material::TextureSlot slot, const std::filesystem::path& path);

        /** Wait until all material textures are loaded.
        */
        void waitForMaterialTextureLoading();

        // Volumes

        /** Get the list of grid volumes.
        */
        const std::vector<GridVolume::SharedPtr>& getGridVolumes() const { return mSceneData.gridVolumes; }

        /** Get a grid volume by name.
            Note: This returns the first volume found with a matching name.
            \param name Volume name.
            \return Returns the first volume with a matching name or nullptr if none was found.
        */
        GridVolume::SharedPtr getGridVolume(const std::string& name) const;

        /** Add a grid volume.
            \param pGridVolume The grid volume.
            \param nodeID The node to attach the volume to (optional).
            \return The ID of the volume in the scene.
        */
        VolumeID addGridVolume(const GridVolume::SharedPtr& pGridVolume, NodeID nodeID = NodeID{ NodeID::Invalid() } );

        // Lights

        /** Get the list of lights.
        */
        const std::vector<Light::SharedPtr>& getLights() const { return mSceneData.lights; }

        /** Get a light by name.
            Note: This returns the first light found with a matching name.
            \param name Light name.
            \return Returns the first light with a matching name or nullptr if none was found.
        */
        Light::SharedPtr getLight(const std::string& name) const;

        /** Add a light source
            \param pLight The light object.
            \return The light ID
        */
        LightID addLight(const Light::SharedPtr& pLight);

        /** DEMO21: Load global light profile.
        */
        void loadLightProfile(const std::string& filename, bool normalize = true);

        // Environment map

        /** Get the environment map.
        */
        const EnvMap::SharedPtr& getEnvMap() const { return mSceneData.pEnvMap; }

        /** Set the environment map.
            \param[in] pEnvMap Environment map. Can be nullptr.
        */
        void setEnvMap(EnvMap::SharedPtr pEnvMap) { mSceneData.pEnvMap = pEnvMap; }

        // Cameras

        /** Get the list of cameras.
        */
        const std::vector<Camera::SharedPtr>& getCameras() const { return mSceneData.cameras; }

        /** Add a camera.
            \param pCamera Camera to be added.
            \return The camera ID
        */
        CameraID addCamera(const Camera::SharedPtr& pCamera);

        /** Get the selected camera.
        */
        Camera::SharedPtr getSelectedCamera() const;

        /** Set the selected camera.
            \param pCamera Camera to use as selected camera (needs to be added first).
        */
        void setSelectedCamera(const Camera::SharedPtr& pCamera);

        /** Get the camera speed.
        */
        float getCameraSpeed() const { return mSceneData.cameraSpeed; }

        /** Set the camera speed.
        */
        void setCameraSpeed(float speed) { mSceneData.cameraSpeed = speed; }

        // Animations

        /** Get the list of animations.
        */
        const std::vector<Animation::SharedPtr>& getAnimations() const { return mSceneData.animations; }

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
        NodeID addNode(const Node& node);

        /** Get how many nodes have been added to the scene graph.
            \return The node count.
        */
        uint32_t getNodeCount() const { return uint32_t(mSceneGraph.size()); }

        /** Add a mesh instance to a node
        */
        void addMeshInstance(NodeID nodeID, MeshID meshID);

        /** Add a curve instance to a node.
        */
        void addCurveInstance(NodeID nodeID, CurveID curveID);

        /** Add an SDF grid instance to a node.
        */
        void addSDFGridInstance(NodeID nodeID, SdfDescID sdfGridID);

        /** Check if a scene node is animated. This check is done recursively through parent nodes.
            \return Returns true if node is animated.
        */
        bool isNodeAnimated(NodeID nodeID) const;

        /** Set the animation interpolation mode for a given scene node. This sets the mode recursively for all parent nodes.
        */
        void setNodeInterpolationMode(NodeID nodeID, Animation::InterpolationMode interpolationMode, bool enableWarping);

    private:
        SceneBuilder(Flags buildFlags);

        struct InternalNode : Node
        {
            InternalNode() = default;
            InternalNode(const Node& n) : Node(n) {}
            std::vector<NodeID> children;          ///< Node IDs of all child nodes.
            std::vector<MeshID> meshes;            ///< Mesh IDs of all meshes this node transforms.
            std::vector<CurveID> curves;           ///< Curve IDs of all curves this node transforms.
            std::vector<SdfGridID> sdfGrids;       ///< SDF grid IDs of all SDF grids this node transforms.
            std::vector<Animatable*> animatable;   ///< Pointers to all animatable objects attached to this node.
            bool dontOptimize = false;             ///< Whether node should be ignored in optimization passes

            /** Returns true if node has any attached scene objects.
            */
            bool hasObjects() const { return !meshes.empty() || !curves.empty() || !sdfGrids.empty() || !animatable.empty(); }
        };

        struct MeshSpec
        {
            std::string name;
            Vao::Topology topology = Vao::Topology::Undefined;
            MaterialID materialId{ 0 };             ///< Global material ID.
            uint32_t staticVertexOffset = 0;        ///< Offset into the shared 'staticData' array. This is calculated in createGlobalBuffers().
            uint32_t staticVertexCount = 0;         ///< Number of static vertices.
            uint32_t skinningVertexOffset = 0;      ///< Offset into the shared 'skinningData' array. This is calculated in createGlobalBuffers().
            uint32_t skinningVertexCount = 0;       ///< Number of skinned vertices.
            uint32_t prevVertexOffset = 0;          ///< Offset into the shared `prevVertices` array. This is calculated in createGlobalBuffers().
            uint32_t prevVertexCount = 0;           ///< Number of previous vertices stored. This can be the static or skinned vertex count depending on animation type.
            uint32_t indexOffset = 0;               ///< Offset into the shared 'indexData' array. This is calculated in createGlobalBuffers().
            uint32_t indexCount = 0;                ///< Number of indices, or zero if non-indexed.
            uint32_t vertexCount = 0;               ///< Number of vertices.
            NodeID  skeletonNodeID{ NodeID::Invalid() }; ///< Node ID of skeleton world transform. Forwarded from Mesh struct.
            bool use16BitIndices = false;           ///< True if the indices are in 16-bit format.
            bool hasSkinningData = false;           ///< True if mesh has skinned vertices.
            bool isStatic = false;                  ///< True if mesh is non-instanced and static (not dynamic or animated).
            bool isFrontFaceCW = false;             ///< Indicate whether front-facing side has clockwise winding in object space.
            bool isDisplaced = false;               ///< True if mesh has displacement map.
            bool isAnimated = false;                ///< True if mesh has vertex animations.
            AABB boundingBox;                       ///< Mesh bounding-box in object space.
            std::vector<NodeID> instances;          ///< Node IDs of all instances of this mesh.

            // Pre-processed vertex data.
            std::vector<uint32_t> indexData;    ///< Vertex indices in either 32-bit or 16-bit format packed tightly, or empty if non-indexed.
            std::vector<StaticVertexData> staticData;
            std::vector<SkinningVertexData> skinningData;

            uint32_t getTriangleCount() const
            {
                FALCOR_ASSERT(topology == Vao::Topology::TriangleList);
                return (indexCount > 0 ? indexCount : vertexCount) / 3;
            }

            uint32_t getIndex(const size_t i) const
            {
                FALCOR_ASSERT(i < indexCount);
                return use16BitIndices ? reinterpret_cast<const uint16_t*>(indexData.data())[i] : indexData[i];
            }

            bool isSkinned() const
            {
                return hasSkinningData;
            }

            bool isDynamic() const
            {
                return isSkinned() || isAnimated;
            }
        };

        // TODO: Add support for dynamic curves
        struct CurveSpec
        {
            std::string name;
            Vao::Topology topology;
            MaterialID materialId{ 0 };         ///< Global material ID.
            uint32_t staticVertexOffset = 0;    ///< Offset into the shared 'staticData' array. This is calculated in createCurveGlobalBuffers().
            uint32_t staticVertexCount = 0;     ///< Number of static curve vertices.
            uint32_t indexOffset = 0;           ///< Offset into the shared 'indexData' array. This is calculated in createCurveGlobalBuffers().
            uint32_t indexCount = 0;            ///< Number of indices.
            uint32_t vertexCount = 0;           ///< Number of vertices.
            uint32_t degree = 1;                ///< Polynomial degree of curve; linear (1) by default.
            std::vector<NodeID> instances;      ///< Node IDs of all instances of this curve.

            // Pre-processed curve vertex data.
            std::vector<uint32_t> indexData;    ///< Vertex indices in 32-bit.
            std::vector<StaticCurveVertexData> staticData;
        };

        using SceneGraph = std::vector<InternalNode>;
        using MeshList = std::vector<MeshSpec>;
        using MeshGroup = Scene::MeshGroup;
        using MeshGroupList = std::vector<MeshGroup>;
        using CurveList = std::vector<CurveSpec>;

        Scene::SceneData mSceneData;
        Scene::SharedPtr mpScene;
        SceneCache::Key mSceneCacheKey;
        bool mWriteSceneCache = false;  ///< True if scene cache should be written after import.

        SceneGraph mSceneGraph;
        const Flags mFlags;

        MeshList mMeshes;
        MeshGroupList mMeshGroups; ///< Groups of meshes. Each group represents all the geometries in a BLAS for ray tracing.

        CurveList mCurves;

        std::unique_ptr<MaterialTextureLoader> mpMaterialTextureLoader;
        GpuFence::SharedPtr mpFence;

        // Helpers
        bool doesNodeHaveAnimation(NodeID nodeID) const;
        void updateLinkedObjects(NodeID oldNodeID, NodeID newNodeID);
        bool collapseNodes(NodeID parentNodeID, NodeID childNodeID);
        bool mergeNodes(NodeID dstNodeID, NodeID srcNodeID);
        void flipTriangleWinding(MeshSpec& mesh);
        void updateSDFGridID(SdfGridID oldID, SdfGridID newID);

        /** Split a mesh by the given axis-aligned splitting plane.
            \return Pair of optional mesh IDs for the meshes on the left and right side, respectively.
        */
        std::pair<std::optional<MeshID>, std::optional<MeshID>> splitMesh(MeshID meshID, const int axis, const float pos);

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
        void prepareDisplacementMaps();
        void prepareSceneGraph();
        void prepareMeshes();
        void removeUnusedMeshes();
        void flattenStaticMeshInstances();
        void optimizeSceneGraph();
        void pretransformStaticMeshes();
        void unifyTriangleWinding();
        void calculateMeshBoundingBoxes();
        void createMeshGroups();
        void optimizeGeometry();
        void sortMeshes();
        void createGlobalBuffers();
        void createCurveGlobalBuffers();
        void optimizeMaterials();
        void removeDuplicateMaterials();
        void collectVolumeGrids();
        void quantizeTexCoords();
        void removeDuplicateSDFGrids();

        // Scene setup
        void createMeshData();
        void createMeshInstanceData(uint32_t& tlasInstanceIndex);
        void createCurveData();
        void createCurveInstanceData(uint32_t& tlasInstanceIndex);
        void createSceneGraph();
        void createMeshBoundingBoxes();
        void calculateCurveBoundingBoxes();

        friend class SceneCache;
    };

    FALCOR_ENUM_CLASS_OPERATORS(SceneBuilder::Flags);
}
