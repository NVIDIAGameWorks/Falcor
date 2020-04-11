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
#include "Core/API/VAO.h"
#include "Animation/Animation.h"
#include "Lights/Light.h"
#include "Lights/LightProbe.h"
#include "Camera/Camera.h"
#include "Material/Material.h"
#include "Utils/Math/AABB.h"
#include "Animation/AnimationController.h"
#include "Camera/CameraController.h"
#include "Experimental/Scene/Lights/LightCollection.h"
#include "SceneTypes.slang"

namespace Falcor
{
    class RtProgramVars;

    /** DXR Scene and Resources Layout:
        - BLAS creation logic is similar to Falcor 3.0, and are grouped in the following order:
            1) For non-instanced meshes, group them if they use the same scene graph transform matrix. One BLAS is created per group.
                a) It is possible a non-instanced mesh has no other meshes to merge with. In that case, the mesh goes in its own BLAS.
            2) For instanced meshes, one BLAS is created per mesh.

        - TLAS Construction:
            - Hit shaders use InstanceID() and GeometryIndex() to identify what was hit.
            - InstanceID is set like a starting offset so that (InstanceID + GeometryIndex) maps to unique indices.
            - Shader table has one hit group per mesh. InstanceContribution is set accordingly for correct lookup.

        Acceleration Structure Layout Example (Scene with 8 meshes, 10 instances total):

                               ----------------------------------------------------------------
                               |                            Value(s)                          |
        ---------------------------------------------------------------------------------------
        | InstanceID           |  0                    |  4  |  5  |  6  |  7  |  8  |  9     |  7 INSTANCE_DESCs in TLAS
        | InstanceContribution |  0                    |  4  |  5  |  6  |  7  |  7  |  7     |  Helps look up one hit group per MESH
        | BLAS Geometry Index  |  0  ,  1  ,  2  ,  3  |  0  |  0  |  0  |  0                 |  5 BLAS's containing 8 meshes total
        ---------------------------------------------------------------------------------------
        | Notes                | Meshes merged into    | One instance    | Multiple instances |
        |                      | one BLAS              | per mesh        | of a mesh          |
        --------------------------------------------------------------------------------------|

        - "InstanceID() + GeometryIndex()" is used for indexing into MeshInstanceData.
        - This is wrapped in getGlobalHitID() in Raytracing.slang.
    */

    class dlldecl Scene : public std::enable_shared_from_this<Scene>
    {
    public:
        using SharedPtr = std::shared_ptr<Scene>;
        using ConstSharedPtrRef = const SharedPtr&;
        using LightList = std::vector<Light::SharedPtr>;
        static const uint32_t kMaxBonesPerVertex = 4;
        static const FileDialogFilterVec kFileExtensionFilters;

        static SharedPtr create(const std::string& filename);

        // #SCENE: we should get rid of this. We can't right now because we can't create a structured-buffer of materials (MaterialData contains textures)
        Shader::DefineList getSceneDefines() const;

        enum class RenderFlags
        {
            None                    = 0x0,
            UserRasterizerState     = 0x1,  ///< Use the rasterizer state currently bound to `pState`. If this flag is not set, the default rasterizer state will be used.
                                            ///< Note that we need to change the rasterizer state during rendering because some meshes have a negative scale factor, and hence the triangles will have a different winding order.
                                            ///< If such meshes exist, overriding the state may result in incorrect rendering output
        };

        /** Flags indicating if and what was updated in the scene
        */
        enum class UpdateFlags
        {
            None                        = 0x0,  ///< Nothing happened
            MeshesMoved                 = 0x1,  ///< Meshes moved
            CameraMoved                 = 0x2,  ///< The camera moved
            CameraPropertiesChanged     = 0x4,  ///< Some camera properties changed, excluding position
            LightsMoved                 = 0x8,  ///< Lights were moved
            LightIntensityChanged       = 0x10, ///< Light intensity changed
            LightPropertiesChanged      = 0x20, ///< Other light changes not included in LightIntensityChanged and LightsMoved
            SceneGraphChanged           = 0x40, ///< Any transform in the scene graph changed.
            LightCollectionChanged      = 0x80, ///< Light collection changed (mesh lights)
            MaterialsChanged            = 0x100,///< Materials changed

            All                         = -1
        };

        /** Settings for how the scene is updated
        */
        enum class UpdateMode
        {
            Rebuild,    ///< Recreate acceleration structure when updates are needed
            Refit       ///< Update acceleration structure when updates are needed
        };

        enum class CameraControllerType
        {
            FirstPerson,
            Orbiter,
            SixDOF
        };

        /** Access the scene's camera to change properties, or use elsewhere.
        */
        const Camera::SharedPtr& getCamera() { return mCamera.pObject; }

        /** Attach a new camera to the scene
        */
        void setCamera(const Camera::SharedPtr& pCamera) { mCamera.pObject = pCamera; }

        /** Set the camera's aspect ratio
        */
        void setCameraAspectRatio(float ratio);

        /** Set a camera controller type
        */
        void setCameraController(CameraControllerType type);

        /** Get the camera controller type
        */
        CameraControllerType getCameraControllerType() const { return mCamCtrlType; }

        /** Toggle whether the camera is animated.
        */
        void toggleCameraAnimation(bool animate) { mCamera.animate = animate; }

        /** Reset the camera.
            This function will place the camera at the center of scene and optionally set the depth range to some reasonable pre-determined values
        */
        void resetCamera(bool resetDepthRange = true);

        /** Set the camera's speed
        */
        void setCameraSpeed(float speed) { mCameraSpeed = speed; }

        /** Get the camera's speed
        */
        float getCameraSpeed() const { return mCameraSpeed; }

        /** Add the current camera's viewpoint to the list of viewpoints.
        */
        void addViewpoint();

        deprecate("4.0.1", "Use addViewpoint() instead.")
        void saveNewViewpoint() { addViewpoint(); }

        /** Add a new viewpoint to the list of viewpoints.
        */
        void addViewpoint(const float3& position, const float3& target, const float3& up);

        /** Remove the currently active viewpoint.
        */
        void removeViewpoint();

        /** Select a viewpoint and move the camera to it.
        */
        void selectViewpoint(uint32_t index);

        deprecate("4.0.1", "Use selectViewpoint() instead.")
        void gotoViewpoint(uint32_t index) { selectViewpoint(index); }

        /** Returns true if there are saved viewpoints (used for dumping to config)
        */
        bool hasSavedViewpoints() { return mViewpoints.size() > 1; }

        /** Get the number of meshes
        */
        uint32_t getMeshCount() const { return (uint32_t)mMeshDesc.size(); }

        /** Get a mesh desc
        */
        const MeshDesc& getMesh(uint32_t meshID) const { return mMeshDesc[meshID]; }

        /** Get the number of mesh instances
        */
        uint32_t getMeshInstanceCount() const { return (uint32_t)mMeshInstanceData.size(); }

        /** Get a mesh instance desc
        */
        const MeshInstanceData& getMeshInstance(uint32_t instanceID) const { return mMeshInstanceData[instanceID]; }

        /** Get the number of materials in the scene
        */
        uint32_t getMaterialCount() const { return (uint32_t)mMaterials.size(); }

        /** Get a material
        */
        Material::ConstSharedPtrRef getMaterial(uint32_t materialID) const { return mMaterials[materialID]; }

        /** Get a material by name
        */
        Material::SharedPtr getMaterialByName(const std::string &name) const;

        /** Get the scene bounds
        */
        const BoundingBox& getSceneBounds() const { return mSceneBB; }

        /** Get a mesh's bounds
        */
        const BoundingBox& getMeshBounds(uint32_t meshID) const { return mMeshBBs[meshID]; }

        /** Get the number of lights in the scene
        */
        uint32_t getLightCount() const { return (uint32_t)mLights.size(); }

        /** Get a light
        */
        Light::ConstSharedPtrRef getLight(uint32_t lightID) const { return mLights[lightID].pObject; }

        /** Get a light by name
        */
        Light::SharedPtr getLightByName(const std::string &name) const;

        /** Get the light collection representing all the mesh lights in the scene.
            The light collection is created lazily on the first call. It needs a render context.
            to run the initialization shaders.
            \param[in] pContext Render context.
            \return Returns the light collection.
        */
        LightCollection::ConstSharedPtrRef getLightCollection(RenderContext* pContext);

        /** Get the light probe or nullptr if it doesn't exist.
        */
        const LightProbe::SharedPtr& getLightProbe() const { return mpLightProbe; }

        /** Toggle whether the specified light is animated.
        */
        void toggleLightAnimation(int index, bool animate) { mLights[index].animate = animate; }

        /** Get/Set how the scene's TLASes are updated when raytracing.
            TLASes are REBUILT by default
        */
        void setTlasUpdateMode(UpdateMode mode) { mTlasUpdateMode = mode; }
        UpdateMode getTlasUpdateMode() { return mTlasUpdateMode; }

        /** Get/Set how the scene's BLASes are updated when raytracing.
            BLASes are REFIT by default
        */
        void setBlasUpdateMode(UpdateMode mode) { mBlasUpdateMode = mode; }
        UpdateMode getBlasUpdateMode() { return mBlasUpdateMode; }

        /** Update the scene. Call this once per frame to update the camera location, animations, etc.
            \param pContext
            \param currentTime The current time in seconds
        */
        UpdateFlags update(RenderContext* pContext, double currentTime);

        /** Get the changes that happened during the last update
            The flags only change during an `update()` call, if something changed between calling `update()` and `getUpdates()`, the returned result will not reflect it
        */
        UpdateFlags getUpdates() const { return mUpdates; }

        /** Render the scene using the rasterizer
        */
        void render(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, RenderFlags flags = RenderFlags::None);

        /** Render the scene using raytracing
        */
        void raytrace(RenderContext* pContext, RtProgram* pProgram, const std::shared_ptr<RtProgramVars>& pVars, uint3 dispatchDims);

        /** Render the UI
        */
        void renderUI(Gui::Widgets& widget);

        /** Bind a sampler to the materials
        */
        void bindSamplerToMaterials(Sampler::ConstSharedPtrRef pSampler);

        /** Get the scene's VAO
        */
        const Vao::SharedPtr& getVao() const { return mpVao; }

        /** Set an environment map.
            \param[in] pEnvMap Texture to use as environment map. Can be nullptr.
        */
        void setEnvironmentMap(Texture::ConstSharedPtrRef pEnvMap);

        /** Load an environment from an image.
            \param[in] filename Texture filename.
        */
        void loadEnvironmentMap(const std::string& filename);

        /** Get the environment map
        */
        Texture::ConstSharedPtrRef getEnvironmentMap() const { return mpEnvMap; }

        /** Handle mouse events
        */
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Handle keyboard events
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Get the filename that the scene was loaded from
        */
        const std::string& getFilename() const { return mFilename; }

        /** Get the animation controller.
        */
        const AnimationController* getAnimationController() const { return mpAnimationController.get(); }

        /** Toggle all animations on or off.
        */
        void toggleAnimations(bool animate);

        /** Get the parameter block with all scene resources.
            Note that the camera is not bound automatically.
        */
        ParameterBlock::ConstSharedPtrRef getParameterBlock() const { return mpSceneBlock; }

        /** Set the BLAS geometry index into the local vars for each geometry.
            This is a workaround before GeometryIndex() is supported in shaders.
        */
        void setGeometryIndexIntoRtVars(const std::shared_ptr<RtProgramVars>& pVars);

        /** Set the scene ray tracing resources into a shader var.
            The acceleration structure is created lazily, which requires the render context.
            \param[in] pContext Render context.
            \param[in] var Shader variable to set data into, usually the root var.
            \param[in] rayTypeCount Number of ray types in raygen program. Not needed for inline raytracing.
        */
        void setRaytracingShaderData(RenderContext* pContext, const ShaderVar& var, uint32_t rayTypeCount = 1);

        std::string getScript(const std::string& sceneVar);

    private:
        friend class SceneBuilder;
        friend class AnimationController;

        static constexpr uint32_t kStaticDataBufferIndex = 0;
        static constexpr uint32_t kPrevVertexBufferIndex = kStaticDataBufferIndex + 1;
        static constexpr uint32_t kDrawIdBufferIndex = kPrevVertexBufferIndex + 1;
        static constexpr uint32_t kVertexBufferCount = kDrawIdBufferIndex + 1;

        static SharedPtr create();

        /** Create scene parameter block and retrieve pointers to buffers
        */
        void initResources();

        /** Uploads scene data to parameter block
        */
        void uploadResources();

        /** Uploads a single material.
        */
        void uploadMaterial(uint32_t materialID);

        /** Update the scene's global bounding box.
        */
        void updateBounds();

        /** Update mesh instance flags
        */
        void updateMeshInstanceFlags();

        /** Do any additional initialization required after scene data is set and draw lists are determined.
        */
        void finalize();

        /** Create the draw list for rasterization
        */
        void createDrawList();

        /** Sort meshes into groups by transform. Updates mMeshInstances and mMeshGroups.
        */
        void sortMeshes();

        /** Initialize geometry descs for each BLAS
        */
        void initGeomDesc();

        /** Generate bottom level acceleration structures for all meshes
        */
        void buildBlas(RenderContext* pContext);

        /** Generate data for creating a TLAS.
            #SCENE TODO: Add argument to build descs based off a draw list
        */
        void fillInstanceDesc(std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry);

        /** Generate top level acceleration structure for the scene. Automatically determines whether to build or refit.
            \param[in] rayCount Number of ray types in the shader. Required to setup how instances index into the Shader Table
        */
        void buildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry);

        UpdateFlags updateCamera(bool forceUpdate);
        UpdateFlags updateLights(bool forceUpdate);
        UpdateFlags updateMaterials(bool forceUpdate);

        void updateGeometryStats();

        struct GeometryStats
        {
            size_t uniqueTriangleCount = 0;     ///< Number of unique triangles. A triangle can exist in multiple instances.
            size_t uniqueVertexCount = 0;       ///< Number of unique vertices. A vertex can be referenced by multiple triangles/instances.
            size_t instancedTriangleCount = 0;  ///< Number of instanced triangles. This is the total number of rendered triangles.
            size_t instancedVertexCount = 0;    ///< Number of instanced vertices. This is the total number of vertices in the rendered triangles.
        };

        template<typename Object>
        struct AnimatedObject
        {
            typename Object::SharedPtr pObject;
            bool animate = true;
            uint32_t nodeID = kInvalidNode;
            bool update(const AnimationController* pAnimCtrl, bool force);
            bool hasGlobalTransform() const { return nodeID != kInvalidNode; }
            void setIntoObject(const float3& pos, const float3& up, const float3& lookAt);
            bool enabled(bool force) const;
        };

        Scene();

        // Scene Geometry
        Vao::SharedPtr mpVao;
        struct DrawArgs
        {
            Buffer::SharedPtr pBuffer;
            uint32_t count = 0;
        } mDrawClockwiseMeshes, mDrawCounterClockwiseMeshes;

        static const uint32_t kInvalidNode = -1;

        struct Node
        {
            Node() = default;
            Node(const std::string& n, uint32_t p, const glm::mat4& t, const glm::mat4& l2b) : parent(p), name(n), transform(t), localToBindSpace(l2b) {};
            std::string name;
            uint32_t parent = kInvalidNode;
            glm::mat4 transform;  // The node's transformation matrix
            glm::mat4 localToBindSpace; // Local to bind space transformation
        };

        struct MeshGroup
        {
            std::vector<uint32_t> meshList;     ///< List of meshId's that are part of the group.
        };

        // #SCENE We don't need those vectors on the host
        std::vector<MeshDesc> mMeshDesc;                    ///< Copy of GPU buffer (mpMeshes)
        std::vector<MeshInstanceData> mMeshInstanceData;    ///< Copy of GPU buffer (mpMeshInstances)
        std::vector<MeshGroup> mMeshGroups;                 ///< Groups of meshes with identical transforms. Each group maps to a BLAS for ray tracing.
        std::vector<Node> mSceneGraph;                      ///< For each index i, the array element indicates the parent node. Indices are in relation to mLocalToWorldMatrices

        std::vector<Material::SharedPtr> mMaterials;        ///< Bound to parameter block
        std::vector<AnimatedObject<Light>> mLights;         ///< Bound to parameter block
        LightCollection::SharedPtr mpLightCollection;       ///< Bound to parameter block
        LightProbe::SharedPtr mpLightProbe;                 ///< Bound to parameter block
        Texture::SharedPtr mpEnvMap;                        ///< Not bound to anything, not rendered automatically. Can be used to render a skybox

        // Scene Metadata (CPU Only)
        std::vector<BoundingBox> mMeshBBs;                          ///< Bounding boxes for meshes (not instances)
        std::vector<std::vector<uint32_t>> mMeshIdToInstanceIds;    ///< Mapping of what instances belong to which mesh
        BoundingBox mSceneBB;                                       ///< Bounding boxes of the entire scene
        std::vector<bool> mMeshHasDynamicData;                      ///< Whether a Mesh has dynamic data, meaning it is skinned
        GeometryStats mGeometryStats;                               ///< Geometry statistics for the scene.

        // Resources
        Buffer::SharedPtr mpMeshesBuffer;
        Buffer::SharedPtr mpMeshInstancesBuffer;
        Buffer::SharedPtr mpMaterialsBuffer;
        Buffer::SharedPtr mpLightsBuffer;
        ParameterBlock::SharedPtr mpSceneBlock;

        // Camera
        CameraControllerType mCamCtrlType = CameraControllerType::FirstPerson;
        CameraController::SharedPtr mpCamCtrl;
        AnimatedObject<Camera> mCamera;
        float mCameraSpeed = 1.0f;

        // Saved Camera Viewpoints
        struct Viewpoint
        {
            float3 position;
            float3 target;
            float3 up;
        };
        std::vector<Viewpoint> mViewpoints;
        uint32_t mCurrentViewpoint = 0;

        // Rendering
        RasterizerState::SharedPtr mpFrontClockwiseRS;
        UpdateFlags mUpdates = UpdateFlags::All;
        AnimationController::UniquePtr mpAnimationController;

        // Raytracing Data
        UpdateMode mTlasUpdateMode = UpdateMode::Rebuild;   ///< How the TLAS should be updated when there are changes in the scene
        UpdateMode mBlasUpdateMode = UpdateMode::Refit;     ///< How the BLAS should be updated when there are changes to meshes

        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> mInstanceDescs; ///< Shared between TLAS builds to avoid reallocating CPU memory

        struct TlasData
        {
            Buffer::SharedPtr pTlas;
            ShaderResourceView::SharedPtr pSrv;         ///< Shader Resource View for binding the TLAS
            Buffer::SharedPtr pInstanceDescs;           ///< Buffer holding instance descs for the TLAS
            UpdateMode updateMode = UpdateMode::Rebuild; ///< Update mode this TLAS was created with.
        };

        std::unordered_map<uint32_t, TlasData> mTlasCache;  ///< Top Level Acceleration Structure for scene data cached per shader ray count
                                                            ///< Number of ray types in program affects Shader Table indexing
        Buffer::SharedPtr mpTlasScratch;                    ///< Scratch buffer used for TLAS builds. Can be shared as long as instance desc count is the same, which for now it is.
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO mTlasPrebuildInfo; ///< This can be reused as long as the number of instance descs doesn't change.

        struct BlasData
        {
            Buffer::SharedPtr pBlas;
            Buffer::SharedPtr pScratchBuffer;

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDescs;
            bool hasSkinnedMesh = false;                ///< Whether the BLAS contains a skinned mesh, which means the BLAS may need to be updated
            UpdateMode updateMode = UpdateMode::Refit;  ///< Update mode this BLAS was created with.
        };

        std::vector<BlasData> mBlasData;    ///< All data related to the scene's BLASes
        bool mHasSkinnedMesh = false;       ///< Whether the scene has a skinned mesh at all.

        std::string mFilename;
    };

    enum_class_operators(Scene::RenderFlags);
    enum_class_operators(Scene::UpdateFlags);
}
