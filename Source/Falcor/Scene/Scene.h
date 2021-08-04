/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/VAO.h"
#include "Animation/Animation.h"
#include "Lights/Light.h"
#include "Lights/LightCollection.h"
#include "Lights/EnvMap.h"
#include "Camera/Camera.h"
#include "Material/Material.h"
#include "Volume/Volume.h"
#include "Volume/Grid.h"
#include "Utils/Math/AABB.h"
#include "Animation/AnimationController.h"
#include "Animation/AnimatedVertexCache.h"
#include "Camera/CameraController.h"
#include "Displacement/DisplacementUpdateTask.slang"
#include "SceneTypes.slang"
#include "HitInfo.h"

// Indicating the implementation of curve back-face culling is in anyhit shaders or intersection shaders.
// Currently, the performance numbers on BabyCheetah scene with 20 indirect bounces are 77ms (with anyhit) and 73ms (without anyhit).
// It will be removed once we have conclusions on performance.
#define CURVE_BACKFACE_CULLING_USING_ANYHIT 0

namespace Falcor
{
    class RtProgramVars;

    /** This class is the main scene representation.
        It holds all scene resources such as geometry, cameras, lights, and materials.

        DXR Scene and Resources Layout:
        - BLAS creation logic:
            1) For static non-instanced meshes, pre-transform and group them into single BLAS.
                a) This can be overridden by the 'RTDontMergeStatic' scene build flag.
            2) For dynamic non-instanced meshes, group them if they use the same scene graph transform matrix. One BLAS is created per group.
                a) This can be overridden by the 'RTDontMergeDynamic' scene build flag.
                b) It is possible a non-instanced mesh has no other meshes to merge with. In that case, the mesh goes in its own BLAS.
            3) For instanced meshes, one BLAS is created per group of mesh with identical instances.
                a) This can be overridden by the 'RTDontMergeInstanced' scene build flag.
            4) Procedural primitives are placed in their own BLAS at the end.

        - TLAS Construction:
            - Hit shaders use InstanceID() and GeometryIndex() to identify what was hit.
            - InstanceID is set like a starting offset so that (InstanceID + GeometryIndex) maps to unique indices.
            - Shader table has one hit group per geometry and ray type. InstanceContribution is set accordingly for correct lookup.

        Acceleration Structure Layout Example (Scene with 11 geometries, 16 instances total):

                               ----------------------------------------------------------------------------------------------
                               |                                         Value(s)                                           |
        ---------------------------------------------------------------------------------------------------------------------
        | InstanceID           |  0                    |  4  |  5  |  6  |  7  |  8  |  9     |  10          |  13          |   9 INSTANCE_DESCs in TLAS
        | InstanceContribution |  0                    |  4  |  5  |  6  |  7  |  7  |  7     |  8           |  8           |   Helps look up one hit group per geometry and ray type
        | BLAS Geometry Index  |  0  ,  1  ,  2  ,  3  |  0  |  0  |  0  |  0                 |  0 , 1 , 2   |  0 , 1 , 2   |   6 BLAS's containing 11 geometries in total
        ---------------------------------------------------------------------------------------------------------------------
        | Notes                | Four geometries in    | One instance    | Multiple instances | Two instances of three      |
        |                      | one BLAS              | per geom/BLAS   | of same geom/BLAS  | geometries in one BLAS      |
        --------------------------------------------------------------------------------------------------------------------|

        - "InstanceID() + GeometryIndex()" is used for indexing into MeshInstanceData for hits on triangle meshes.
        - This is wrapped in getGeometryInstanceID() in Raytracing.slang.
    */
    class dlldecl Scene : public std::enable_shared_from_this<Scene>
    {
    public:
        using SharedPtr = std::shared_ptr<Scene>;
        using GeometryType = PrimitiveTypeFlags;

        using UpdateCallback = std::function<void(const Scene::SharedPtr& pScene, double currentTime)>;

        static const uint32_t kMaxBonesPerVertex = 4;
        static const uint32_t kInvalidBone = -1;
        static const uint32_t kInvalidGrid = -1;
        static const uint32_t kInvalidNode = Animatable::kInvalidNode;
        static const uint32_t kInvalidIndex = -1;

        static const FileDialogFilterVec& getFileExtensionFilters();

        /** Create scene from file.
            \param[in] filename Import the scene from this file.
            \return Scene object, or nullptr if an error occured.
        */
        static SharedPtr create(const std::string& filename);

        /** Get scene defines.
            These defines must be set on all programs that access the scene.
            The defines are static and it's sufficient to set them once after loading.
            \return List of shader defines.
        */
        Shader::DefineList getSceneDefines() const;

        /** Render settings determining how the scene is rendered.
            This is used primarily by the path tracer renderers.
        */
        struct RenderSettings
        {
            bool useEnvLight = true;        ///< Enable lighting from environment map.
            bool useAnalyticLights = true;  ///< Enable lighting from analytic lights.
            bool useEmissiveLights = true;  ///< Enable lighting from emissive lights.
            bool useVolumes = true;         ///< Enable rendering of heterogeneous volumes.

            bool operator==(const RenderSettings& other) const
            {
                return (useEnvLight == other.useEnvLight) &&
                    (useAnalyticLights == other.useAnalyticLights) &&
                    (useEmissiveLights == other.useEmissiveLights) &&
                    (useVolumes == other.useVolumes);
            }

            bool operator!=(const RenderSettings& other) const { return !(*this == other); }
        };

        /** Optional importer-provided rendering metadata
         */
        struct Metadata
        {
            std::optional<float> fNumber;                       ///< Lens aperture.
            std::optional<float> filmISO;                       ///< Film speed.
            std::optional<float> shutterSpeed;                  ///< (Reciprocal) shutter speed.
            std::optional<uint32_t> samplesPerPixel;            ///< Number of primary samples per pixel.
            std::optional<uint32_t> maxDiffuseBounces;          ///< Maximum number of diffuse bounces.
            std::optional<uint32_t> maxSpecularBounces;         ///< Maximum number of specular bounces.
            std::optional<uint32_t> maxTransmissionBounces;     ///< Maximum number of transmission bounces.
            std::optional<uint32_t> maxVolumeBounces;           ///< Maximum number of volume bounces.
        };

        /** Flags indicating if and what was updated in the scene.
        */
        enum class UpdateFlags
        {
            None                        = 0x0,      ///< Nothing happened
            MeshesMoved                 = 0x1,      ///< Meshes moved
            CameraMoved                 = 0x2,      ///< The camera moved
            CameraPropertiesChanged     = 0x4,      ///< Some camera properties changed, excluding position
            CameraSwitched              = 0x8,      ///< Selected a different camera
            LightsMoved                 = 0x10,     ///< Lights were moved
            LightIntensityChanged       = 0x20,     ///< Light intensity changed
            LightPropertiesChanged      = 0x40,     ///< Other light changes not included in LightIntensityChanged and LightsMoved
            SceneGraphChanged           = 0x80,     ///< Any transform in the scene graph changed.
            LightCollectionChanged      = 0x100,    ///< Light collection changed (mesh lights)
            MaterialsChanged            = 0x200,    ///< Materials changed
            EnvMapChanged               = 0x400,    ///< Environment map changed
            EnvMapPropertiesChanged     = 0x800,    ///< Environment map properties changed (check EnvMap::getChanges() for more specific information)
            LightCountChanged           = 0x1000,   ///< Number of active lights changed
            RenderSettingsChanged       = 0x2000,   ///< Render settings changed
            VolumesMoved                = 0x4000,   ///< Volumes were moved
            VolumePropertiesChanged     = 0x8000,   ///< Volume properties changed
            VolumeGridsChanged          = 0x10000,  ///< Volume grids changed
            VolumeBoundsChanged         = 0x20000,  ///< Volume bounds changed
            CurvesMoved                 = 0x40000,  ///< Curves moved.
            CustomPrimitivesMoved       = 0x80000,  ///< Custom primitives moved.
            GeometryChanged             = 0x100000, ///< Scene geometry changed (added/removed).
            DisplacementChanged         = 0x200000, ///< Displacement mapping parameters changed.

            All                         = -1
        };

        /** Settings for how the scene ray tracing acceleration structures are updated.
        */
        enum class UpdateMode
        {
            Rebuild,    ///< Recreate acceleration structure when updates are needed.
            Refit       ///< Update acceleration structure when updates are needed.
        };

        enum class CameraControllerType
        {
            FirstPerson,
            Orbiter,
            SixDOF
        };

        /** Statistics.
        */
        struct SceneStats
        {
            // Geometry stats
            uint64_t meshCount = 0;                     ///< Number of meshes.
            uint64_t meshInstanceCount = 0;             ///< Number if mesh instances.
            uint64_t meshInstanceOpaqueCount = 0;       ///< Number if mesh instances that are opaque.
            uint64_t transformCount = 0;                ///< Number of transform matrices.
            uint64_t uniqueTriangleCount = 0;           ///< Number of unique triangles. A triangle can exist in multiple instances.
            uint64_t uniqueVertexCount = 0;             ///< Number of unique vertices. A vertex can be referenced by multiple triangles/instances.
            uint64_t instancedTriangleCount = 0;        ///< Number of instanced triangles. This is the total number of rendered triangles.
            uint64_t instancedVertexCount = 0;          ///< Number of instanced vertices. This is the total number of vertices in the rendered triangles.
            uint64_t indexMemoryInBytes = 0;            ///< Total memory in bytes used by the index buffer.
            uint64_t vertexMemoryInBytes = 0;           ///< Total memory in bytes used by the vertex buffer.
            uint64_t geometryMemoryInBytes = 0;         ///< Total memory in bytes used by the geometry data (meshes, curves, custom primitives, instances etc.).
            uint64_t animationMemoryInBytes = 0;        ///< Total memory in bytes used by the animation system (transforms, skinning buffers).

            // Curve stats
            uint64_t curveCount = 0;                    ///< Number of curves.
            uint64_t curveInstanceCount = 0;            ///< Number of curve instances.
            uint64_t uniqueCurveSegmentCount = 0;       ///< Number of unique curve segments (linear tube segments by default). A segment can exist in multiple instances.
            uint64_t uniqueCurvePointCount = 0;         ///< Number of unique curve points. A point can be referenced by multiple segments/instances.
            uint64_t instancedCurveSegmentCount = 0;    ///< Number of instanced curve segments (linear tube segments by default). This is the total number of rendered segments.
            uint64_t instancedCurvePointCount = 0;      ///< Number of instanced curve points. This is the total number of end points in the rendered segments.
            uint64_t curveIndexMemoryInBytes = 0;       ///< Total memory in bytes used by the curve index buffer.
            uint64_t curveVertexMemoryInBytes = 0;      ///< Total memory in bytes used by the curve vertex buffer.

            // Custom primitive stats
            uint64_t customPrimitiveCount = 0;          ///< Number of custom primitives.

            // Material stats
            uint64_t materialCount = 0;                 ///< Number of materials.
            uint64_t materialOpaqueCount = 0;           ///< Number of materials that are opaque.
            uint64_t materialMemoryInBytes = 0;         ///< Total memory in bytes used by the material data.
            uint64_t textureCount = 0;                  ///< Number of unique textures. A texture can be referenced by multiple materials.
            uint64_t textureCompressedCount = 0;        ///< Number of unique compressed textures.
            uint64_t textureTexelCount = 0;             ///< Total number of texels in all textures.
            uint64_t textureMemoryInBytes = 0;          ///< Total memory in bytes used by the textures.

            // Raytracing stats
            uint64_t blasGroupCount = 0;                ///< Number of BLAS groups. There is one BLAS buffer per group.
            uint64_t blasCount = 0;                     ///< Number of BLASes.
            uint64_t blasCompactedCount = 0;            ///< Number of compacted BLASes.
            uint64_t blasOpaqueCount = 0;               ///< Number of BLASes that contain only opaque geometry.
            uint64_t blasGeometryCount = 0;             ///< Number of geometries.
            uint64_t blasOpaqueGeometryCount = 0;       ///< Number of geometries that are opaque.
            uint64_t blasMemoryInBytes = 0;             ///< Total memory in bytes used by the BLASes.
            uint64_t blasScratchMemoryInBytes = 0;      ///< Additional memory in bytes kept around for BLAS updates etc.
            uint64_t tlasCount = 0;                     ///< Number of TLASes.
            uint64_t tlasMemoryInBytes = 0;             ///< Total memory in bytes used by the TLASes.
            uint64_t tlasScratchMemoryInBytes = 0;      ///< Additional memory in bytes kept around for TLAS updates etc.

            // Light stats
            uint64_t activeLightCount = 0;              ///< Number of active lights.
            uint64_t totalLightCount = 0;               ///< Number of lights in the scene.
            uint64_t pointLightCount = 0;               ///< Number of point lights.
            uint64_t directionalLightCount = 0;         ///< Number of directional lights.
            uint64_t rectLightCount = 0;                ///< Number of rect lights.
            uint64_t sphereLightCount = 0;              ///< Number of sphere lights.
            uint64_t distantLightCount = 0;             ///< Number of distant lights.
            uint64_t lightsMemoryInBytes = 0;           ///< Total memory in bytes used by the analytic lights.
            uint64_t envMapMemoryInBytes = 0;           ///< Total memory in bytes used by the environment map.
            uint64_t emissiveMemoryInBytes = 0;         ///< Total memory in bytes used by the emissive lights.

            // Volume stats
            uint64_t volumeCount = 0;               ///< Number of volumes.
            uint64_t volumeMemoryInBytes = 0;       ///< Total memory in bytes used by the volumes.

            // Grid stats
            uint64_t gridCount = 0;                 ///< Number of grids.
            uint64_t gridVoxelCount = 0;            ///< Total number of voxels in all grids.
            uint64_t gridMemoryInBytes = 0;         ///< Total memory in bytes used by the grids.

            /** Get the total memory usage.
            */
            uint64_t getTotalMemory() const
            {
                return indexMemoryInBytes + vertexMemoryInBytes + geometryMemoryInBytes + animationMemoryInBytes +
                    curveIndexMemoryInBytes + curveVertexMemoryInBytes + materialMemoryInBytes + textureMemoryInBytes +
                    blasMemoryInBytes + blasScratchMemoryInBytes + tlasMemoryInBytes + tlasScratchMemoryInBytes +
                    lightsMemoryInBytes + envMapMemoryInBytes + emissiveMemoryInBytes +
                    volumeMemoryInBytes + gridMemoryInBytes;
            }

            /** Convert to python dict.
            */
            pybind11::dict toPython() const;
        };

        const SceneStats& getSceneStats() const { return mSceneStats; }

        /** Get the render settings.
        */
        const RenderSettings& getRenderSettings() const { return mRenderSettings; }

        /** Get the render settings.
        */
        RenderSettings& getRenderSettings() { return mRenderSettings; }

        /** Set the render settings.
        */
        void setRenderSettings(const RenderSettings& renderSettings) { mRenderSettings = renderSettings; }

        /** Returns true if environment map is available and should be used as the background.
        */
        bool useEnvBackground() const;

        /** Returns true if environment map is available and should be used as a distant light.
        */
        bool useEnvLight() const;

        /** Returns true if there are active analytic lights and they should be used for lighting.
        */
        bool useAnalyticLights() const;

        /** Returns true if there are active emissive lights and they should be used for lighting.
        */
        bool useEmissiveLights() const;

        /** Returns true if there are active volumes and they should be rendererd.
        */
        bool useVolumes() const;

        /** Get the metadata.
        */
        const Metadata& getMetadata() { return mMetadata; }

        /** Get the scene update callback.
        */
        UpdateCallback getUpdateCallback() const { return mUpdateCallback; }

        /** Set the scene update callback.
        */
        void setUpdateCallback(UpdateCallback updateCallback) { mUpdateCallback = updateCallback; }

        /** Access the scene's currently selected camera to change properties or to use elsewhere.
        */
        const Camera::SharedPtr& getCamera() { return mCameras[mSelectedCamera]; }

        /** Get a list of all cameras in the scene.
        */
        const std::vector<Camera::SharedPtr>& getCameras() { return mCameras; };

        /** Select a different camera to use. The camera must already exist in the scene.
        */
        void setCamera(const Camera::SharedPtr& pCamera);

        /** Set the currently selected camera's aspect ratio.
        */
        void setCameraAspectRatio(float ratio);

        /** Set the camera controller type.
        */
        void setCameraController(CameraControllerType type);

        /** Get the camera controller type.
        */
        CameraControllerType getCameraControllerType() const { return mCamCtrlType; }

        /** Reset the currently selected camera.
            This function will place the camera at the center of scene and optionally set the depth range to some reasonable pre-determined values
        */
        void resetCamera(bool resetDepthRange = true);

        /** Set the camera's speed.
        */
        void setCameraSpeed(float speed);

        /** Get the camera's speed.
        */
        float getCameraSpeed() const { return mCameraSpeed; }

        /** Add the currently selected camera's viewpoint to the list of viewpoints.
        */
        void addViewpoint();

        /** Select a camera to be used by index.
        */
        void selectCamera(uint32_t index);

        /** Select a camera to be used by name.
        */
        void selectCamera(std::string name);

        /** Add a new viewpoint to the list of viewpoints.
        */
        void addViewpoint(const float3& position, const float3& target, const float3& up, uint32_t cameraIndex = 0);

        /** Remove the currently active viewpoint.
        */
        void removeViewpoint();

        /** Select a viewpoint and move the camera to it.
        */
        void selectViewpoint(uint32_t index);

        /** Returns true if there are saved viewpoints (used for dumping to config).
        */
        bool hasSavedViewpoints() { return mViewpoints.size() > 1; }

        /** Get the number of geometries in the scene.
            This includes all types of geometry that exist in the ray tracing acceleration structures.
            \return Total number of geometries.
        */
        uint32_t getGeometryCount() const;

        /** Get the type of a given geometry.
            \param[in] geometryID Global geometry ID.
            \return The type of the given geometry.
        */
        GeometryType getGeometryType(uint32_t geometryID) const;

        /** Check if scene has any geometry of the given type.
            \param[in] type The type to check for.
            \return True if scene has any geometry of this type.
        */
        bool hasGeometryType(GeometryType type) const { return is_set(getPrimitiveTypes(), (PrimitiveTypeFlags)type); }

        /** Get the number of triangle meshes.
        */
        uint32_t getMeshCount() const { return (uint32_t)mMeshDesc.size(); }

        /** Get a mesh desc.
        */
        const MeshDesc& getMesh(uint32_t meshID) const { return mMeshDesc[meshID]; }

        /** Get the number of mesh instances.
        */
        uint32_t getMeshInstanceCount() const { return (uint32_t)mMeshInstanceData.size(); }

        /** Get a mesh instance desc.
        */
        const MeshInstanceData& getMeshInstance(uint32_t instanceID) const { return mMeshInstanceData[instanceID]; }

        /** Get the number of displaced mesh instances.
            Note: All displaced mesh instances are at the end of the mesh instance list.
        */
        uint32_t getDisplacedMeshInstanceCount() const { return mDisplacedMeshInstanceCount; }

        /** Get the number of curves.
        */
        uint32_t getCurveCount() const { return (uint32_t)mCurveDesc.size(); }

        /** Get a curve desc.
        */
        const CurveDesc& getCurve(uint32_t curveID) const { return mCurveDesc[curveID]; }

        /** Get the number of curve instances.
        */
        uint32_t getCurveInstanceCount() const { return (uint32_t)mCurveInstanceData.size(); }

        /** Get a curve instance desc.
        */
        const CurveInstanceData& getCurveInstance(uint32_t instanceID) const { return mCurveInstanceData[instanceID]; }

        /** Get the number of custom primitives.
        */
        uint32_t getCustomPrimitiveCount() const { return (uint32_t)mCustomPrimitiveDesc.size(); }

        /** Get the custom primitive index for a geometry.
            \param[in] geometryID Global geometry ID.
            \return The custom primitive index of the geometry that can be used with getCustomPrimitive().
        */
        uint32_t getCustomPrimitiveIndex(uint32_t geometryID) const;

        /** Get a custom primitive.
            \param[in] index Index of the custom primitive.
        */
        const CustomPrimitiveDesc& getCustomPrimitive(uint32_t index) const;

        /** Get a custom primitive AABB.
            \param[in] index Index of the custom primitive.
        */
        const AABB& getCustomPrimitiveAABB(uint32_t index) const;

        /** Add a custom primitive.
            Custom primitives are sequentially numbered in the scene. The function returns the index at
            which the primitive was inserted. Note that this index may change if primitives are removed.
            Adding/removing custom primitives is a slow operation as the acceleration structure is rebuilt.
            \param[in] userID User ID of primitive.
            \param[in] aabb AABB of the primitive.
            \return Index of the custom primitive that was added.
        */
        uint32_t addCustomPrimitive(uint32_t userID, const AABB& aabb);

        /** Remove a custom primitive.
            Custom primitives are sequentially numbered in the scene. The function removes the primitive at
            the given index. Note that the index of subsequent primtives will change.
            Adding/removing custom primitives is a slow operation as the acceleration structure is rebuilt.
            \param[in] index Index of custom primitive to remove.
        */
        void removeCustomPrimitive(uint32_t index) { removeCustomPrimitives(index, index + 1); }

        /** Remove a range [first,last) of custom primitives.
            Note that the last index is non-inclusive. If first == last no action is performed.
            \param[in] first Index of first custom primitive to remove.
            \param[in] last Index one past the last custom primitive to remove.
        */
        void removeCustomPrimitives(uint32_t first, uint32_t last);

        /** Update a custom primitive.
            \param[in] index Index of the custom primitive.
            \param[in] aabb AABB of the primitive.
        */
        void updateCustomPrimitive(uint32_t index, const AABB& aabb);

        /** Get primitive types that exist in the scene.
        */
        PrimitiveTypeFlags getPrimitiveTypes() const { return mPrimitiveTypes; }

        /** Get a list of all materials in the scene.
        */
        const std::vector<Material::SharedPtr>& getMaterials() const { return mMaterials; }

        /** Get the total number of materials in the scene.
        */
        uint32_t getMaterialCount() const { return (uint32_t)mMaterials.size(); }

        /** Get the number of materials of the given type.
        */
        uint32_t getMaterialCount(MaterialType type) const;

        /** Get a material.
        */
        const Material::SharedPtr& getMaterial(uint32_t materialID) const { return mMaterials[materialID]; }

        /** Get a material by name.
        */
        Material::SharedPtr getMaterialByName(const std::string& name) const;

        /** Get a list of all volumes in the scene.
        */
        const std::vector<Volume::SharedPtr>& getVolumes() const { return mVolumes; }

        /** Get a volume.
        */
        const Volume::SharedPtr& getVolume(uint32_t volumeID) const { return mVolumes[volumeID]; }

        /** Get a volume by name.
        */
        Volume::SharedPtr getVolumeByName(const std::string& name) const;

        /** Get the hit info requirements.
        */
        const HitInfo& getHitInfo() const { return mHitInfo; }

        /** Get the scene bounds in world space.
        */
        const AABB& getSceneBounds() const { return mSceneBB; }

        /** Get a mesh's bounds in object space.
        */
        const AABB& getMeshBounds(uint32_t meshID) const { return mMeshBBs[meshID]; }

        /** Get a curve's bounds in object space.
        */
        const AABB& getCurveBounds(uint32_t curveID) const { return mCurveBBs[curveID]; }

        /** Get a list of all lights in the scene.
        */
        const std::vector<Light::SharedPtr>& getLights() { return mLights; };

        /** Get the number of lights in the scene.
        */
        uint32_t getLightCount() const { return (uint32_t)mLights.size(); }

        /** Get a light.
        */
        const Light::SharedPtr& getLight(uint32_t lightID) const { return mLights[lightID]; }

        /** Get a light by name.
        */
        Light::SharedPtr getLightByName(const std::string& name) const;

        /** Get the light collection representing all the mesh lights in the scene.
            The light collection is created lazily on the first call. It needs a render context.
            to run the initialization shaders.
            \param[in] pContext Render context.
            \return Returns the light collection.
        */
        const LightCollection::SharedPtr& getLightCollection(RenderContext* pContext);

        /** Get the environment map or nullptr if it doesn't exist.
        */
        const EnvMap::SharedPtr& getEnvMap() const { return mpEnvMap; }

        /** Set how the scene's TLASes are updated when raytracing.
            TLASes are REBUILT by default.
        */
        void setTlasUpdateMode(UpdateMode mode) { mTlasUpdateMode = mode; }

        /** Get the scene's TLAS update mode when raytracing.
        */
        UpdateMode getTlasUpdateMode() { return mTlasUpdateMode; }

        /** Set how the scene's BLASes are updated when raytracing.
            BLASes are REFIT by default.
        */
        void setBlasUpdateMode(UpdateMode mode);

        /** Get the scene's BLAS update mode when raytracing.
        */
        UpdateMode getBlasUpdateMode() { return mBlasUpdateMode; }

        /** Update the scene. Call this once per frame to update the camera location, animations, etc.
            \param[in] pContext
            \param[in] currentTime The current time in seconds
        */
        UpdateFlags update(RenderContext* pContext, double currentTime);

        /** Get the changes that happened during the last update.
            The flags only change during an `update()` call, if something changed between calling `update()` and `getUpdates()`, the returned result will not reflect it
        */
        UpdateFlags getUpdates() const { return mUpdates; }

        /** Render the scene using the rasterizer.
            Note the rasterizer state bound to 'pState' is ignored.
            \param[in] pContext Render context.
            \param[in] pState Graphics state.
            \param[in] pVars Graphics vars.
            \param[in] cullMode Optional rasterizer cull mode. The default is to cull back-facing primitives.
        */
        void rasterize(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, RasterizerState::CullMode cullMode = RasterizerState::CullMode::Back);

        /** Render the scene using the rasterizer.
            This overload uses the supplied rasterizer states.
            \param[in] pContext Render context.
            \param[in] pState Graphics state.
            \param[in] pVars Graphics vars.
            \param[in] pRasterizerStateCW Rasterizer state for meshes with clockwise triangle winding.
            \param[in] pRasterizerStateCCW Rasterizer state for meshes with counter-clockwise triangle winding. Can be the same as for clockwise.
        */
        void rasterize(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, const RasterizerState::SharedPtr& pRasterizerStateCW, const RasterizerState::SharedPtr& pRasterizerStateCCW);

        /** Get the required raytracing maximum attribute size for this scene.
            Note: This depends on what types of geometry are used in the scene.
            \return Max attribute size in bytes.
        */
        uint32_t getRaytracingMaxAttributeSize() const;

        /** Render the scene using raytracing.
        */
        void raytrace(RenderContext* pContext, RtProgram* pProgram, const std::shared_ptr<RtProgramVars>& pVars, uint3 dispatchDims);

        /** Render the UI.
        */
        void renderUI(Gui::Widgets& widget);

        /** Bind a sampler to the materials.
        */
        void bindSamplerToMaterials(const Sampler::SharedPtr& pSampler);

        /** Get the scene's VAO.
            The default VAO uses 32-bit vertex indices. For meshes with 16-bit indices, use getVao16() instead.
        */
        const Vao::SharedPtr& getVao() const { return mpVao; }

        /** Get the scene's VAO for 16-bit vertex indices.
        */
        const Vao::SharedPtr& getVao16() const { return mpVao16Bit; }

        /** Get the scene's VAO for curves.
        */
        const Vao::SharedPtr& getCurveVao() const { return mpCurveVao; }

        /** Set an environment map.
            \param[in] pEnvMap Environment map. Can be nullptr.
        */
        void setEnvMap(EnvMap::SharedPtr pEnvMap);

        /** Load an environment from an image.
            \param[in] filename Texture filename.
        */
        void loadEnvMap(const std::string& filename);

        /** Handle mouse events.
        */
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Handle keyboard events.
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Get the filename that the scene was loaded from.
        */
        const std::string& getFilename() const { return mFilename; }

        /** Get the animation controller.
        */
        const AnimationController* getAnimationController() const { return mpAnimationController.get(); }

        /** Get the scene's animations.
        */
        std::vector<Animation::SharedPtr>& getAnimations() { return mpAnimationController->getAnimations(); }

        /** Returns true if scene has animation data.
        */
        bool hasAnimation() const { return mpAnimationController->hasAnimations(); }

        /** Enable/disable scene animation.
        */
        void setIsAnimated(bool isAnimated) { mpAnimationController->setEnabled(isAnimated); }

        /** Returns true if scene animation is enabled.
        */
        bool isAnimated() const { return mpAnimationController->isEnabled(); };

        /** Enable/disable global animation looping.
        */
        void setIsLooped(bool looped) { mpAnimationController->setIsLooped(looped); }

        /** Returns true if scene animations are looped globally.
        */
        bool isLooped() { return mpAnimationController->isLooped(); }

        /** Toggle all animations on or off.
        */
        void toggleAnimations(bool animate);

        /** Get the parameter block with all scene resources.
        */
        const ParameterBlock::SharedPtr& getParameterBlock() const { return mpSceneBlock; }

        /** Set the scene ray tracing resources into a shader var.
            The acceleration structure is created lazily, which requires the render context.
            \param[in] pContext Render context.
            \param[in] var Shader variable to set data into, usually the root var.
            \param[in] rayTypeCount Number of ray types in raygen program. Not needed for DXR 1.1.
        */
        void setRaytracingShaderData(RenderContext* pContext, const ShaderVar& var, uint32_t rayTypeCount = 1);

        /** Get the name of the mesh with the given ID.
        */
        std::string getMeshName(uint32_t meshID) const { assert(meshID < mMeshNames.size());  return mMeshNames[meshID]; }

        /** Return true if the given mesh ID is valid, false otherwise.
        */
        bool hasMesh(uint32_t meshID) const { return meshID < mMeshNames.size(); }

        /** Get a list of raytracing BLAS IDs for all meshes. The list is arranged by mesh ID.
        */
        std::vector<uint32_t> getMeshBlasIDs() const;

        /** Returns the scene graph parent node ID for a node.
            \return Node ID of the parent node, or kInvalidNode if top-level node.
        */
        uint32_t getParentNodeID(uint32_t nodeID) const;

        static void nullTracePass(RenderContext* pContext, const uint2& dim);

        std::string getScript(const std::string& sceneVar);

    private:
        /** Represents a group of meshes.
            The meshes are geometries in the same ray tracing bottom-level acceleration structure (BLAS).
        */
        struct MeshGroup
        {
            std::vector<uint32_t> meshList;     ///< List of meshId's that are part of the group.
            bool isStatic = false;              ///< True if group represents static non-instanced geometry.
            bool isDisplaced = false;           ///< True if group uses displacement mapping.
        };

        /** Scene graph node.
        */
        struct Node
        {
            Node() = default;
            Node(const std::string& n, uint32_t p, const glm::mat4& t, const glm::mat4& mb, const glm::mat4& l2b) : parent(p), name(n), transform(t), meshBind(mb), localToBindSpace(l2b) {};
            std::string name;
            uint32_t parent = kInvalidNode;
            float4x4 transform;         ///< The node's transformation matrix.
            float4x4 meshBind;          ///< For skinned meshes. Mesh world space transform at bind time.
            float4x4 localToBindSpace;  ///< For bones. Skeleton to bind space transformation. AKA the inverse-bind transform.
        };

        /** Full set of required data to create a scene object.
            This data is typically prepared by SceneBuilder before creating a Scene object.
        */
        struct SceneData
        {
            std::string filename;                                   ///< Filename of the asset file the scene was loaded from.
            RenderSettings renderSettings;                          ///< Render settings.
            std::vector<Camera::SharedPtr> cameras;                 ///< List of cameras.
            uint32_t selectedCamera = 0;                            ///< Index of selected camera.
            float cameraSpeed = 1.f;                                ///< Camera speed.
            std::vector<Light::SharedPtr> lights;                   ///< List of light sources.
            std::vector<Material::SharedPtr> materials;             ///< List of materials.
            std::vector<Volume::SharedPtr> volumes;                 ///< List of heterogeneous volumes.
            std::vector<Grid::SharedPtr> grids;                     ///< List of volume grids.
            EnvMap::SharedPtr pEnvMap;                              ///< Environment map.
            std::vector<Node> sceneGraph;                           ///< Scene graph nodes.
            std::vector<Animation::SharedPtr> animations;           ///< List of animations.
            Metadata metadata;                                      ///< Scene meadata.

            // Mesh data
            std::vector<MeshDesc> meshDesc;                         ///< List of mesh descriptors.
            std::vector<std::string> meshNames;                     ///< List of mesh names.
            std::vector<AABB> meshBBs;                              ///< List of mesh bounding boxes in object space.
            std::vector<MeshInstanceData> meshInstanceData;         ///< List of mesh instances.
            uint32_t displacedMeshInstanceCount;                    ///< Number of displaced mesh instances. All displaced mesh instances are at the end of the mesh instance list.
            std::vector<std::vector<uint32_t>> meshIdToInstanceIds; ///< Mapping of what instances belong to which mesh.
            std::vector<MeshGroup> meshGroups;                      ///< List of mesh groups. Each group maps to a BLAS for ray tracing.
            std::vector<CachedMesh> cachedMeshes;                   ///< Cached data for vertex-animated meshes.

            bool has16BitIndices = false;                           ///< True if 16-bit mesh indices are used.
            bool has32BitIndices = false;                           ///< True if 32-bit mesh indices are used.
            uint32_t meshDrawCount = 0;                             ///< Number of meshes to draw.

            std::vector<uint32_t> meshIndexData;                    ///< Vertex indices for all meshes in either 32-bit or 16-bit format packed tightly, decided per mesh.
            std::vector<PackedStaticVertexData> meshStaticData;     ///< Vertex attributes for all meshes in packed format.
            std::vector<DynamicVertexData> meshDynamicData;         ///< Additional vertex attributes for dynamic (skinned) meshes.

            // Curve data
            std::vector<CurveDesc> curveDesc;                       ///< List of curve descriptors.
            std::vector<AABB> curveBBs;                             ///< List of curve bounding boxes in object space. Each curve consists of many segments, each with its own AABB. The bounding boxes here are the unions of those.
            std::vector<CurveInstanceData> curveInstanceData;       ///< List of curve instances.

            std::vector<uint32_t> curveIndexData;                   ///< Vertex indices for all curves in 32-bit.
            std::vector<StaticCurveVertexData> curveStaticData;     ///< Vertex attributes for all curves.
            std::vector<CachedCurve> cachedCurves;                  ///< Vertex cache for dynamic (vertex animated) curves.

            // Custom primitive data
            std::vector<CustomPrimitiveDesc> customPrimitiveDesc;   ///< Custom primitive descriptors.
            std::vector<AABB> customPrimitiveAABBs;                 ///< List of AABBs for custom primitives in world space. Each custom primitive consists of one AABB.
        };

        friend class SceneBuilder;
        friend class SceneCache;
        friend class AnimationController;
        friend class AnimatedVertexCache;

        static constexpr uint32_t kStaticDataBufferIndex = 0;
        static constexpr uint32_t kDrawIdBufferIndex = kStaticDataBufferIndex + 1;
        static constexpr uint32_t kVertexBufferCount = kDrawIdBufferIndex + 1;

        static SharedPtr create(SceneData&& sceneData);

        void createMeshVao(uint32_t drawCount, const std::vector<uint32_t>& indexData, const std::vector<PackedStaticVertexData>& staticData, const std::vector<DynamicVertexData>& dynamicData);
        void createCurveVao(const std::vector<uint32_t>& indexData, const std::vector<StaticCurveVertexData>& staticData);

        /** Create scene parameter block and retrieve pointers to buffers.
        */
        void initResources();

        /** Uploads scene data to parameter block.
        */
        void uploadResources();

        /** Uploads a single material.
        */
        void uploadMaterial(uint32_t materialID);

        /** Uploads the currently selected camera.
        */
        void uploadSelectedCamera();

        /** Update the scene's global bounding box.
        */
        void updateBounds();

        /** Update mesh instances.
        */
        void updateMeshInstances(bool forceUpdate);

        /** Update curve instances.
        */
        void updateCurveInstances(bool forceUpdate);

        /** Update primitive type flags.
        */
        void updatePrimitiveTypes();

        /** Do any additional initialization required after scene data is set and draw lists are determined.
        */
        void finalize();

        /** Create the draw list for rasterization.
        */
        void createDrawList();

        /** Initialize geometry descs for each BLAS.
        */
        void initGeomDesc(RenderContext* pContext);

        /** Initialize pre-build information for each BLAS.
        */
        void preparePrebuildInfo(RenderContext* pContext);

        /** Compute BLAS groups.
        */
        void computeBlasGroups();

        /** Generate bottom level acceleration structures for all meshes.
        */
        void buildBlas(RenderContext* pContext);

        /** Generate data for creating a TLAS.
            #SCENE TODO: Add argument to build descs based off a draw list.
        */
        void fillInstanceDesc(std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry) const;

        /** Generate top level acceleration structure for the scene. Automatically determines whether to build or refit.
            \param[in] rayCount Number of ray types in the shader. Required to setup how instances index into the Shader Table.
        */
        void buildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry);

        /** Check whether scene has an index buffer.
        */
        bool hasIndexBuffer() const { return mpVao->getIndexBuffer() != nullptr; }

        /** Initialize all cameras in the scene through the animation controller using their corresponding scene graph nodes.
        */
        void initializeCameras();

        /** Prepare all UI-related objects that do not change over the course of execution.
        */
        void prepareUI();

        /** Update an animatable object.
        */
        bool updateAnimatable(Animatable& animatable, const AnimationController& controller, bool force = false);

        UpdateFlags updateSelectedCamera(bool forceUpdate);
        UpdateFlags updateLights(bool forceUpdate);
        UpdateFlags updateVolumes(bool forceUpdate);
        UpdateFlags updateEnvMap(bool forceUpdate);
        UpdateFlags updateMaterials(bool forceUpdate);
        UpdateFlags updateGeometry(bool forceUpdate);
        UpdateFlags updateProceduralPrimitives(bool forceUpdate);
        UpdateFlags updateRaytracingAABBData(bool forceUpdate);
        UpdateFlags updateDisplacement(bool forceUpdate);

        void updateGeometryStats();
        void updateMaterialStats();
        void updateRaytracingBLASStats();
        void updateRaytracingTLASStats();
        void updateLightStats();
        void updateVolumeStats();

        Scene(SceneData&& sceneData);

        // Scene Geometry

        struct DrawArgs
        {
            Buffer::SharedPtr pBuffer;      ///< Buffer holding the draw-indirect arguments.
            uint32_t count = 0;             ///< Number of draws.
            bool ccw = true;                ///< True if counterclockwise triangle winding.
            ResourceFormat ibFormat = ResourceFormat::Unknown;  ///< Index buffer format.
        };

        PrimitiveTypeFlags mPrimitiveTypes;                         ///< Flags indicating what primitive types exist in the scene.

        bool mHas16BitIndices = false;                              ///< True if any meshes use 16-bit indices.
        bool mHas32BitIndices = false;                              ///< True if any meshes use 32-bit indices.

        Vao::SharedPtr mpVao;                                       ///< Vertex array object for the global mesh vertex/index buffers.
        Vao::SharedPtr mpVao16Bit;                                  ///< VAO for drawing meshes with 16-bit vertex indices.
        Vao::SharedPtr mpCurveVao;                                  ///< Vertex array object for the global curve vertex/index buffers.
        std::vector<DrawArgs> mDrawArgs;                            ///< List of draw arguments for rasterizing the meshes in the scene.

        std::vector<MeshDesc> mMeshDesc;                            ///< Copy of mesh data GPU buffer (mpMeshesBuffer).
        std::vector<MeshInstanceData> mMeshInstanceData;            ///< Mesh instance data.
        uint32_t mDisplacedMeshInstanceCount;                       ///< Number of displaced mesh instances. All displaced mesh instances are at the end of the mesh instance list.
        std::vector<PackedMeshInstanceData> mPackedMeshInstanceData;///< Copy of packed mesh instance data GPU buffer (mpMeshInstancesBuffer).
        std::vector<MeshGroup> mMeshGroups;                         ///< Groups of meshes. Each group maps to a BLAS for ray tracing.
        std::vector<std::string> mMeshNames;                        ///< Mesh names, indxed by mesh ID
        std::vector<Node> mSceneGraph;                              ///< For each index i, the array element indicates the parent node. Indices are in relation to mLocalToWorldMatrices.

        // Displacement mapping.
        struct
        {
            bool needsUpdate = true;                                ///< True if displacement data has changed and a AABB update is required.
            struct DisplacementMeshData { uint32_t AABBOffset = 0; uint32_t AABBCount = 0; };
            std::vector<DisplacementMeshData> meshData;             ///< List of displacement mesh data (reference to AABBs).
            std::vector<DisplacementUpdateTask> updateTasks;        ///< List of displacement AABB update tasks.
            Buffer::SharedPtr pUpdateTasksBuffer;                   ///< GPU Buffer with list of displacement AABB update tasks.
            ComputePass::SharedPtr pUpdatePass;                     ///< Comput epass to update displacement AABB data.
            Buffer::SharedPtr pAABBBuffer;                          ///< GPU Buffer of raw displacement AABB data. Used for acceleration structure creation, and bound to the Scene for access in shaders.
        } mDisplacement;

        // Procedural primitives
        std::vector<CurveDesc> mCurveDesc;                          ///< Copy of curve data GPU buffer (mpCurvesBuffer).
        std::vector<CurveInstanceData> mCurveInstanceData;          ///< Curve instance data.
        std::vector<uint32_t> mCurveIndexData;                      ///< Vertex indices for all curves in 32-bit.
        std::vector<StaticCurveVertexData> mCurveStaticData;        ///< Vertex attributes for all curves.

        std::vector<CustomPrimitiveDesc> mCustomPrimitiveDesc;      ///< Copy of custom primitive data GPU buffer (mpCustomPrimitivesBuffer).
        std::vector<AABB> mCustomPrimitiveAABBs;                    ///< User-defined custom primitive AABBs.
        uint32_t mCustomPrimitiveAABBOffset = 0;                    ///< Offset of custom primitive AABBs in global AABB list.
        bool mCustomPrimitivesMoved = false;                        ///< Flag indicating that custom primitives were moved since last frame.
        bool mCustomPrimitivesChanged = false;                      ///< Flag indicating that custom primitives were added/removed since last frame.

        // The following array and buffer records the AABBs of all procedural primitives, including custom primitives, curves, etc.
        // There is an implicit type conversion from D3D12_RAYTRACING_AABB to AABB (defined in Utils.Math.AABB).
        // It is fine because both structs have the same data layout.
        std::vector<D3D12_RAYTRACING_AABB> mRtAABBRaw;              ///< Raw AABB data (min, max) for all procedural primitives.
        Buffer::SharedPtr mpRtAABBBuffer;                           ///< GPU Buffer of raw AABB data. Used for acceleration structure creation, and bound to the Scene for access in shaders.

        // Materials
        std::vector<Material::SharedPtr> mMaterials;                ///< Bound to parameter block.
        std::vector<uint32_t> mMaterialCountByType;                 ///< Number of materials of each type, indexed by MaterialType.
        std::vector<uint32_t> mSortedMaterialIndices;               ///< Indices of materials, sorted alphabetically by case-insensitive name.
        bool mSortMaterialsByName = false;                          ///< If true, display materials sorted by name, rather than by ID.
        bool mHasSpecGlossMaterials = false;                        ///< If true, scene uses materials with the SpecGloss shading model.

        // Lights
        std::vector<Light::SharedPtr> mLights;                      ///< All analytic lights. Note that not all may be active.
        std::vector<Volume::SharedPtr> mVolumes;                    ///< All loaded volumes.
        std::vector<Grid::SharedPtr> mGrids;                        ///< All loaded volume grids.
        std::unordered_map<Grid::SharedPtr, uint32_t> mGridIDs;     ///< Lookup table for grid IDs.
        LightCollection::SharedPtr mpLightCollection;               ///< Class for managing emissive geometry. This is created lazily upon first use.
        EnvMap::SharedPtr mpEnvMap;                                 ///< Environment map or nullptr if not loaded.
        bool mEnvMapChanged = false;                                ///< Flag indicating that the environment map has changed since last frame.
        uint32_t mActiveLightCount = 0;                             ///< Number of currently active analytic lights.

        // Scene metadata (CPU only)
        std::vector<AABB> mMeshBBs;                                 ///< Bounding boxes for meshes (not instances) in object space.
        std::vector<std::vector<uint32_t>> mMeshIdToInstanceIds;    ///< Mapping of what instances belong to which mesh. The instanceID are sorted in ascending order.
        std::vector<AABB> mCurveBBs;                                ///< Bounding boxes for curves (not instances) in object space.
        std::vector<std::vector<uint32_t>> mCurveIdToInstanceIds;   ///< Mapping of what instances belong to which curve.
        HitInfo mHitInfo;                                           ///< Geometry hit info requirements.
        AABB mSceneBB;                                              ///< Bounding boxes of the entire scene in world space.
        SceneStats mSceneStats;                                     ///< Scene statistics.
        Metadata mMetadata;                                         ///< Importer-provided metadata.
        RenderSettings mRenderSettings;                             ///< Render settings.
        RenderSettings mPrevRenderSettings;
        UpdateCallback mUpdateCallback;                             ///< Scene update callback.

        // Scene block resources
        Buffer::SharedPtr mpMeshesBuffer;
        Buffer::SharedPtr mpMeshInstancesBuffer;
        Buffer::SharedPtr mpCurvesBuffer;
        Buffer::SharedPtr mpCurveInstancesBuffer;
        Buffer::SharedPtr mpCustomPrimitivesBuffer;
        Buffer::SharedPtr mpMaterialsBuffer;
        Buffer::SharedPtr mpLightsBuffer;
        Buffer::SharedPtr mpVolumesBuffer;
        ParameterBlock::SharedPtr mpSceneBlock;

        // Camera
        CameraControllerType mCamCtrlType = CameraControllerType::FirstPerson;
        CameraController::SharedPtr mpCamCtrl;
        std::vector<Camera::SharedPtr> mCameras;
        uint32_t mSelectedCamera = 0;
        float mCameraSpeed = 1.0f;
        bool mCameraSwitched = false;

        Gui::DropdownList mCameraList;

        // Saved Camera Viewpoints
        struct Viewpoint
        {
            uint32_t index;
            float3 position;
            float3 target;
            float3 up;
        };
        std::vector<Viewpoint> mViewpoints;
        uint32_t mCurrentViewpoint = 0;

        // Rendering
        std::map<RasterizerState::CullMode, RasterizerState::SharedPtr> mFrontClockwiseRS;
        std::map<RasterizerState::CullMode, RasterizerState::SharedPtr> mFrontCounterClockwiseRS;
        UpdateFlags mUpdates = UpdateFlags::All;
        AnimationController::UniquePtr mpAnimationController;

        // Raytracing data
        UpdateMode mTlasUpdateMode = UpdateMode::Rebuild;   ///< How the TLAS should be updated when there are changes in the scene.
        UpdateMode mBlasUpdateMode = UpdateMode::Refit;     ///< How the BLAS should be updated when there are changes to meshes.

        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> mInstanceDescs; ///< Shared between TLAS builds to avoid reallocating CPU memory.

        struct TlasData
        {
            Buffer::SharedPtr pTlas;
            ShaderResourceView::SharedPtr pSrv;             ///< Shader Resource View for binding the TLAS.
            Buffer::SharedPtr pInstanceDescs;               ///< Buffer holding instance descs for the TLAS.
            UpdateMode updateMode = UpdateMode::Rebuild;    ///< Update mode this TLAS was created with.
        };

        std::unordered_map<uint32_t, TlasData> mTlasCache;  ///< Top Level Acceleration Structure for scene data cached per shader ray count.
                                                            ///< Number of ray types in program affects Shader Table indexing.
        Buffer::SharedPtr mpTlasScratch;                    ///< Scratch buffer used for TLAS builds. Can be shared as long as instance desc count is the same, which for now it is.
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO mTlasPrebuildInfo; ///< This can be reused as long as the number of instance descs doesn't change.

        /** Describes one BLAS.
        */
        struct BlasData
        {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS buildInputs;
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDescs;

            uint32_t blasGroupIndex = 0;                    ///< Index of the BLAS group that contains this BLAS.

            uint64_t resultByteSize = 0;                    ///< Maximum result data size for the BLAS build, including padding.
            uint64_t resultByteOffset = 0;                  ///< Offset into the BLAS result buffer.
            uint64_t scratchByteSize = 0;                   ///< Maximum scratch data size for the BLAS build, including padding.
            uint64_t scratchByteOffset = 0;                 ///< Offset into the BLAS scratch buffer.

            uint64_t blasByteSize = 0;                      ///< Size of the final BLAS post-compaction, including padding.
            uint64_t blasByteOffset = 0;                    ///< Offset into the final BLAS buffer.

            bool hasProceduralPrimitives = false;           ///< True if the BLAS contains procedural primitives. Otherwise it is triangles.
            bool hasSkinnedMesh = false;                    ///< Whether the BLAS contains a skinned mesh, which means the BLAS may need to be updated.
            bool hasAnimatedVertexCache = false;            ///< Whether the BLAS contains an animated vertex cache, which means the BLAS may need to be updated.
            bool useCompaction = false;                     ///< Whether the BLAS should be compacted after build.
            UpdateMode updateMode = UpdateMode::Refit;      ///< Update mode this BLAS was created with.
        };

        /** Describes a group of BLASes.
        */
        struct BlasGroup
        {
            std::vector<uint32_t> blasIndices;              ///< Indices of all BLASes in the group.

            uint64_t resultByteSize = 0;                    ///< Maximum result data size for all BLASes in the group, including padding.
            uint64_t scratchByteSize = 0;                   ///< Maximum scratch data size for all BLASes in the group, including padding.
            uint64_t finalByteSize = 0;                     ///< Size of the final BLASes in the group post-compaction, including padding.

            Buffer::SharedPtr pBlas;                        ///< Buffer containing all final BLASes in the group.
        };

        // BLAS Data is ordered as all mesh BLAS's first, followed by one BLAS containing all AABBs.
        std::vector<BlasData> mBlasData;                    ///< All data related to the scene's BLASes.
        std::vector<BlasGroup> mBlasGroups;                 ///< BLAS group data.
        Buffer::SharedPtr mpBlasScratch;                    ///< Scratch buffer used for BLAS builds.
        Buffer::SharedPtr mpBlasStaticWorldMatrices;        ///< Object-to-world transform matrices in row-major format. Only valid for static meshes.
        bool mRebuildBlas = true;                           ///< Flag to indicate BLASes need to be rebuilt.
        bool mHasSkinnedMesh = false;                       ///< Whether the scene has a skinned mesh at all.
        bool mHasAnimatedVertexCache = false;               ///< Whether the scene has an animated vertex cache at all.

        std::string mFilename;
        bool mFinalized = false;                            ///< True if scene is ready to be bound to the GPU.
    };

    enum_class_operators(Scene::UpdateFlags);
}
