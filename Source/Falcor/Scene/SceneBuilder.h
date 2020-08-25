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
#include "VertexAttrib.slangh"

namespace Falcor
{
    class dlldecl SceneBuilder
    {
    public:
        using SharedPtr = std::shared_ptr<SceneBuilder>;

        /** Flags that control how the scene will be built. They can be combined together.
        */
        enum class Flags
        {
            None                        = 0x0,    ///< None
            RemoveDuplicateMaterials    = 0x1,    ///< Deduplicate materials that have the same properties. The material name is ignored during the search.
            UseOriginalTangentSpace     = 0x2,    ///< Use the original tangent space that was loaded with the mesh. By default, we will ignore it and use MikkTSpace to generate the tangent space. We will always generate tangent space if it is missing.
            AssumeLinearSpaceTextures   = 0x4,    ///< By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.
            DontMergeMeshes             = 0x8,    ///< Preserve the original list of meshes in the scene, don't merge meshes with the same material.
            BuffersAsShaderResource     = 0x10,   ///< Generate the VBs and IB with the shader-resource-view bind flag.
            UseSpecGlossMaterials       = 0x20,   ///< Set materials to use Spec-Gloss shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.
            UseMetalRoughMaterials      = 0x40,   ///< Set materials to use Metal-Rough shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else.
            NonIndexedVertices          = 0x80,   ///< Convert meshes to use non-indexed vertices. This requires more memory but may increase performance.

            Default = None
        };

        /** Mesh description
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

        /** Adds a node to the graph.
            Note that if the node contains data other then the transform matrix (such as meshes or lights), you'll need to add those objects before adding the node.
            \return The node ID.
        */
        uint32_t addNode(const Node& node);

        /** Check if a scene node is animated. This check is done recursively through parent nodes.
            \return Returns true if node is animated.
        */
        bool isNodeAnimated(uint32_t nodeID) const;

        /** Set the animation interpolation mode for a given scene node. This sets the mode recursively for all parent nodes.
        */
        void setNodeInterpolationMode(uint32_t nodeID, Animation::InterpolationMode interpolationMode, bool enableWarping);

        /** Add a mesh instance to a node
        */
        void addMeshInstance(uint32_t nodeID, uint32_t meshID);

        /** Add a mesh. This function will throw an exception if something went wrong.
            \param meshDesc The mesh's description.
            \return The ID of the mesh in the scene. Note that all of the instances share the same mesh ID.
        */
        uint32_t addMesh(const Mesh& meshDesc);

        /** Add a light source
            \param pLight The light object.
            \return The light ID
        */
        uint32_t addLight(const Light::SharedPtr& pLight);

        /** Get the number of attached lights
        */
        size_t getLightCount() const { return mLights.size(); }

        /** Set a light-probe
            \param pProbe The environment map. You can set it to null to disable environment mapping
        */
        void setLightProbe(const LightProbe::SharedPtr& pProbe) { mpLightProbe = pProbe; }

        /** Set an environment map.
            \param[in] pEnvMap Environment map. Can be nullptr.
        */
        void setEnvMap(EnvMap::SharedPtr pEnvMap) { mpEnvMap = pEnvMap; }

        /** Add a camera.
            \param pCamera Camera to be added.
            \return The camera ID
        */
        uint32_t addCamera(const Camera::SharedPtr& pCamera);

        /** Get the number of attached cameras
        */
        size_t getCameraCount() const { return mCameras.size(); }

        /** Select a camera.
            \param name The name of the camera to select.
        */
        void setCamera(const std::string name);

        /** Get the build flags
        */
        Flags getFlags() const { return mFlags; }

        /** Add an animation
            \param animation The animation
        */
        void addAnimation(const Animation::SharedPtr& pAnimation);

        /** Set the camera's speed
        */
        void setCameraSpeed(float speed) { mCameraSpeed = speed; }

    private:
        SceneBuilder(Flags buildFlags);

        struct InternalNode : Node
        {
            InternalNode() = default;
            InternalNode(const Node& n) : Node(n) {}
            std::vector<uint32_t> children;
            std::vector<uint32_t> meshes;
        };

        struct MeshSpec
        {
            MeshSpec() = default;
            Vao::Topology topology;
            uint32_t materialId = 0;
            uint32_t indexOffset = 0;
            uint32_t staticVertexOffset = 0;
            uint32_t dynamicVertexOffset = 0;
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;
            bool hasDynamicData = false;
            std::vector<uint32_t> instances; // Node IDs
        };

        // Geometry data
        struct BuffersData
        {
            std::vector<uint32_t> indices;
            std::vector<PackedStaticVertexData> staticData;
            std::vector<DynamicVertexData> dynamicData;
        } mBuffersData;

        using SceneGraph = std::vector<InternalNode>;
        using MeshList = std::vector<MeshSpec>;

        bool mDirty = true;
        Scene::SharedPtr mpScene;

        SceneGraph mSceneGraph;
        const Flags mFlags;

        MeshList mMeshes;
        std::vector<Material::SharedPtr> mMaterials;
        std::unordered_map<const Material*, uint32_t> mMaterialToId;

        std::vector<Camera::SharedPtr> mCameras;
        std::vector<Light::SharedPtr> mLights;
        LightProbe::SharedPtr mpLightProbe;
        EnvMap::SharedPtr mpEnvMap;
        std::vector<Animation::SharedPtr> mAnimations;
        uint32_t mSelectedCamera = 0;
        float mCameraSpeed = 1.0f;

        uint32_t addMaterial(const Material::SharedPtr& pMaterial, bool removeDuplicate);
        Vao::SharedPtr createVao(uint16_t drawCount);

        uint32_t createMeshData(Scene* pScene);
        void createGlobalMatricesBuffer(Scene* pScene);
        void calculateMeshBoundingBoxes(Scene* pScene);
        void createAnimationController(Scene* pScene);
        std::string mFilename;
    };

    enum_class_operators(SceneBuilder::Flags);
}
