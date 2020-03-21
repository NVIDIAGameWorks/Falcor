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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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

        /** Flags that control how the scene will be built. They can be combined together
        */
        enum class Flags
        {
            None                        = 0x0,    ///< None
            RemoveDuplicateMaterials    = 0x1,    ///< Deduplicate materials that have the same properties. The material name is ignored during the search
            UseOriginalTangentSpace     = 0x2,    ///< Use the original bitangents that were loaded with the mesh. By default, we will ignore them and use MikkTSpace to generate the tangent space. We will always generate bitangents if they are missing
            AssumeLinearSpaceTextures   = 0x4,    ///< By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.
            DontMergeMeshes             = 0x8,    ///< Preserve the original list of meshes in the scene, don't merge meshes with the same material
            BuffersAsShaderResource     = 0x10,   ///< Generate the VBs and IB with the shader-resource-view bind flag
            UseSpecGlossMaterials       = 0x20,   ///< Set materials to use Spec-Gloss shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else
            UseMetalRoughMaterials      = 0x40,   ///< Set materials to use Metal-Rough shading model. Otherwise default is Spec-Gloss for OBJ, Metal-Rough for everything else

            Default = None
        };

        /** Mesh description
        */
        struct Mesh
        {
            std::string name;                           // The mesh's name
            uint32_t vertexCount = 0;                   // The number of vertices the mesh has
            uint32_t indexCount = 0;                    // The number of indices the mesh has. Can't be zero - the scene doesn't support non-indexed meshes. If you'd like us to support non-indexed meshes, please open an issue
            const uint32_t* pIndices    = nullptr;      // Array of indices. The element count must match `indexCount`
            const vec3* pPositions      = nullptr;      // Array of vertex positions. The element count must match `vertexCount`. This field is required
            const vec3* pNormals        = nullptr;      // Array of vertex normals. The element count must match `vertexCount`.   This field is required
            const vec3* pBitangents     = nullptr;      // Array of vertex bitangent. The element count must match `vertexCount`. Optional. If set to nullptr, or if BuildFlags::UseOriginalTangentSpace is not set, the tangents will be generated using MikkTSpace
            const vec2* pTexCrd         = nullptr;      // Array of vertex texture coordinates. The element count must match `vertexCount`. This field is required
            const vec3* pLightMapUVs    = nullptr;      // Array of light-map UVs. The element count must match `vertexCount`. This field is optional
            const uvec4* pBoneIDs       = nullptr;      // Array of bone IDs. The element count must match `vertexCount`. This field is optional. If it's set, that means that the mesh is animated, in which case pBoneWeights can't be nullptr
            const vec4*  pBoneWeights   = nullptr;      // Array of bone weights. The element count must match `vertexCount`. This field is optional. If it's set, that means that the mesh is animated, in which case pBoneIDs can't be nullptr
            Vao::Topology topology = Vao::Topology::Undefined; // The primitive topology of the mesh
            Material::SharedPtr pMaterial;              // The mesh's material. Can't be nullptr
        };

        static const uint32_t kInvalidNode = Scene::kInvalidNode;
        struct Node
        {
            std::string name;
            mat4 transform;
            mat4 localToBindPose;   // For bones
            size_t parent = kInvalidNode;
        };

        using InstanceMatrices = std::vector<mat4>;

        /** Construct a new object
        */
        SceneBuilder(Flags buildFlags = Flags::Default);

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
        bool import(const std::string& filename, const InstanceMatrices& instances = InstanceMatrices());

        /** Get the scene. Make sure to add all the objects before calling this function
            \return nullptr if something went wrong, otherwise a new Scene object
        */
        Scene::SharedPtr getScene();

        /** Adds a node to the graph
            Note that if the node contains data other then the transform matrix (such as meshes or lights), you'll need to add those objects before adding the node.
        */
        size_t addNode(const Node& node);

        /** Add a mesh instance to a node
        */
        void addMeshInstance(size_t nodeID, size_t meshID);

        /** Add a mesh. This function will throw an exception if something went wrong
            \param mesh The mesh's desc
            \param flags The build flags
            \return The ID of the mesh in the scene. Note that all of the instances share the same mesh ID
        */
        size_t addMesh(const Mesh& mesh);

        /** Add a light source
            \param pLight The light object. Can't be nullptr
            \return The light ID
        */
        size_t addLight(const Light::SharedPtr& pLight, size_t nodeID = kInvalidNode);

        /** Get the number of attached lights
        */
        size_t getLightCount() const { return mLights.size(); }

        /** Set a light-probe
            \param pProbe The environment map. You can set it to null to disable environment mapping
        */
        void setLightProbe(LightProbe::ConstSharedPtrRef pProbe) { mpLightProbe = pProbe; }

        /** Set an environment map.
            \param[in] pEnvMap Texture to use as environment map. Can be nullptr.
        */
        void setEnvironmentMap(Texture::ConstSharedPtrRef pEnvMap) { mpEnvMap = pEnvMap; }

        /** Set the camera
        */
        void setCamera(const Camera::SharedPtr& pCamera, size_t nodeID = kInvalidNode);

        /** Get the build flags
        */
        Flags getFlags() const { return mFlags; }

        /** Add an animation
            \param meshID The mesh ID the animation should be applied to
            \param animation The animation
            \return The ID of the animation. The ID is relative to number of animations which are associated with the specified mesh, it's not a global ID
        */
        size_t addAnimation(size_t meshID, Animation::ConstSharedPtrRef pAnimation);

        /** Set the camera's speed
        */
        void setCameraSpeed(float speed) { mCameraSpeed = speed; }

        /** Check if a camera exists
        */
        bool hasCamera() const { return mCamera.pObject != nullptr; }
    private:
        struct InternalNode : Node
        {
            InternalNode() = default;
            InternalNode(const Node& n) : Node(n) {}
            std::vector<size_t> children;
            std::vector<size_t> meshes;
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
            std::vector<Animation::SharedPtr> animations;
        };

        // Geometry data
        struct BuffersData
        {
            std::vector<uint32_t> indices;
            std::vector<StaticVertexData> staticData;
            std::vector<DynamicVertexData> dynamicData;
        } mBuffersData;

        using SceneGraph = std::vector<InternalNode>;
        using MeshList = std::vector<MeshSpec>;

        bool mDirty = true;
        Scene::SharedPtr mpScene;

        SceneGraph mSceneGraph;
        Flags mFlags;

        MeshList mMeshes;
        std::vector<Material::SharedPtr> mMaterials;
        std::unordered_map<const Material*, uint32_t> mMaterialToId;

        Scene::AnimatedObject<Camera> mCamera;
        std::vector<Scene::AnimatedObject<Light>> mLights;
        LightProbe::SharedPtr mpLightProbe;
        Texture::SharedPtr mpEnvMap;
        float mCameraSpeed = 1.0f;

        uint32_t addMaterial(const Material::SharedPtr& pMaterial, bool removeDuplicate);
        Vao::SharedPtr createVao(uint16_t drawCount);

        uint32_t createMeshData(Scene* pScene);
        void createGlobalMatricesBuffer(Scene* pScene);
        void calculateMeshBoundingBoxes(Scene* pScene);
        void createAnimationController(Scene* pScene);
        std::string mFilename;
    };

    inline std::string to_string(SceneBuilder::Flags flags)
    {
#define t2s(t_) case SceneBuilder::Flags::t_: return #t_;
        switch (flags)
        {
            t2s(None);
            t2s(RemoveDuplicateMaterials);
            t2s(UseOriginalTangentSpace);
            t2s(AssumeLinearSpaceTextures);
            t2s(DontMergeMeshes);
            t2s(BuffersAsShaderResource);
            t2s(UseSpecGlossMaterials);
            t2s(UseMetalRoughMaterials);
        default:
            should_not_get_here();
            return "";
        }
#undef t2s
    }

    enum_class_operators(SceneBuilder::Flags);
}
