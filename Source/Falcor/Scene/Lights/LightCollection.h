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
#include "MeshLightData.slang"
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/API/Sampler.h"
#include "Core/API/GpuFence.h"
#include "Core/State/GraphicsState.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Math/Vector.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include <memory>
#include <vector>

namespace Falcor
{
    class Scene;
    class RenderContext;
    struct ShaderVar;

    /** Class that holds a collection of mesh lights for a scene.

        Each mesh light is represented by a mesh instance with an emissive material.

        This class has utility functions for updating and pre-processing the mesh lights.
        The LightCollection can be used standalone, but more commonly it will be wrapped
        by an emissive light sampler.
    */
    class FALCOR_API LightCollection
    {
    public:
        using SharedPtr = std::shared_ptr<LightCollection>;
        using SharedConstPtr = std::shared_ptr<const LightCollection>;

        enum class UpdateFlags : uint32_t
        {
            None                = 0u,   ///< Nothing was changed.
            MatrixChanged       = 1u,   ///< Mesh instance transform changed.
        };

        struct UpdateStatus
        {
            std::vector<UpdateFlags> lightsUpdateInfo;
        };

        struct MeshLightStats
        {
            // Stats before pre-processing (input data).
            uint32_t meshLightCount = 0;                ///< Number of mesh lights.
            uint32_t triangleCount = 0;                 ///< Number of mesh light triangles (total).
            uint32_t meshesTextured = 0;                ///< Number of mesh lights with textured emissive.
            uint32_t trianglesTextured = 0;             ///< Number of mesh light triangles with textured emissive.

            // Stats after pre-processing (what's used during rendering).
            uint32_t trianglesCulled = 0;               ///< Number of triangles culled (due to zero radiance and/or area).
            uint32_t trianglesActive = 0;               ///< Number of active (non-culled) triangles.
            uint32_t trianglesActiveUniform = 0;        ///< Number of active triangles with const radiance.
            uint32_t trianglesActiveTextured = 0;       ///< Number of active triangles with textured radiance.
        };

        /** Represents one mesh light triangle vertex.
        */
        struct MeshLightVertex
        {
            float3 pos;     ///< World-space position.
            float2 uv;      ///< Texture coordinates in emissive texture (if textured).
        };

        /** Represents one mesh light triangle.
        */
        struct MeshLightTriangle
        {
            // TODO: Perf of indexed vs non-indexed on GPU. We avoid level of indirection, but have more bandwidth non-indexed.
            MeshLightVertex vtx[3];                             ///< Vertices. These are non-indexed for now.
            uint32_t        lightIdx = MeshLightData::kInvalidIndex; ///< Per-triangle index into mesh lights array.

            // Pre-computed quantities.
            float3          normal = float3(0);                 ///< Triangle's face normal in world space.
            float3          averageRadiance = float3(0);        ///< Average radiance emitted over triangle. For textured emissive the radiance varies over the surface.
            float           flux = 0.f;                         ///< Pre-integrated flux emitted by the triangle. Note that emitters are single-sided.
            float           area = 0.f;                         ///< Triangle area in world space units.

            /** Returns the center of the triangle in world space.
            */
            float3 getCenter() const
            {
                return (vtx[0].pos + vtx[1].pos + vtx[2].pos) / 3.0f;
            }
        };


        ~LightCollection() = default;

        /** Creates a light collection for the given scene.
            Note that update() must be called before the collection is ready to use.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
            \return A pointer to a new light collection object, or throws an exception if creation failed.
        */
        static SharedPtr create(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene);

        /** Updates the light collection to the current state of the scene.
            \param[in] pRenderContext The render context.
            \param[out] pUpdateStatus Stores information about which type of updates were performed for each mesh light. This is an optional output parameter.
            \return True if the lighting in the scene has changed since the last frame.
        */
        bool update(RenderContext* pRenderContext, UpdateStatus* pUpdateStatus = nullptr);

        /** Bind the light collection data to a given shader var
            \param[in] var The shader variable to set the data into.
        */
        void setShaderData(const ShaderVar& var) const;

        /** Returns the total number of active (non-culled) triangle lights.
        */
        uint32_t getActiveLightCount() const { return getStats().trianglesActive; }

        /** Returns the total number of triangle lights (may include culled triangles).
        */
        uint32_t getTotalLightCount() const { return mTriangleCount; }

        /** Returns stats.
        */
        const MeshLightStats& getStats() const { computeStats(); return mMeshLightStats; }

        /** Returns a CPU buffer with all emissive triangles in world space.
            Note that update() must have been called before for the data to be valid.
            Call prepareSyncCPUData() ahead of time to avoid stalling the GPU.
        */
        const std::vector<MeshLightTriangle>& getMeshLightTriangles() const { syncCPUData(); return mMeshLightTriangles; }

        /** Returns a CPU buffer with all mesh lights.
            Note that update() must have been called before for the data to be valid.
        */
        const std::vector<MeshLightData>& getMeshLights() const { return mMeshLights; }

        /** Prepare for syncing the CPU data.
            If the mesh light triangles will be accessed with getMeshLightTriangles()
            performance can be improved by calling this function ahead of time.
            This function schedules the copies so that it can be read back without delay later.
        */
        void prepareSyncCPUData(RenderContext* pRenderContext) const { copyDataToStagingBuffer(pRenderContext); }

        /** Get the total GPU memory usage in bytes.
        */
        uint64_t getMemoryUsageInBytes() const;

        // Internal update flags. This only public for FALCOR_ENUM_CLASS_OPERATORS() to work.
        enum class CPUOutOfDateFlags : uint32_t
        {
            None         = 0,
            TriangleData = 0x1,
            FluxData     = 0x2,

            All          = TriangleData | FluxData
        };

    protected:
        LightCollection(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene);

        void initIntegrator(const Scene& scene);
        void setupMeshLights(const Scene& scene);
        void build(RenderContext* pRenderContext, const Scene& scene);
        void prepareTriangleData(RenderContext* pRenderContext, const Scene& scene);
        void prepareMeshData(const Scene& scene);
        void integrateEmissive(RenderContext* pRenderContext, const Scene& scene);
        void computeStats() const;
        void buildTriangleList(RenderContext* pRenderContext, const Scene& scene);
        void updateActiveTriangleList();
        void updateTrianglePositions(RenderContext* pRenderContext, const Scene& scene, const std::vector<uint32_t>& updatedLights);

        void copyDataToStagingBuffer(RenderContext* pRenderContext) const;
        void syncCPUData() const;

        // Internal state
        std::weak_ptr<Scene>                    mpScene;                ///< Weak pointer to scene (scene owns LightCollection).

        std::vector<MeshLightData>              mMeshLights;            ///< List of all mesh lights.
        uint32_t                                mTriangleCount = 0;     ///< Total number of triangles in all mesh lights (= mMeshLightTriangles.size()). This may include culled triangles.

        mutable std::vector<MeshLightTriangle>  mMeshLightTriangles;    ///< List of all pre-processed mesh light triangles.
        mutable std::vector<uint32_t>           mActiveTriangleList;    ///< List of active (non-culled) emissive triangles.
        mutable std::vector<uint32_t>           mTriToActiveList;       ///< Mapping of all light triangles to index in mActiveTriangleList.

        mutable MeshLightStats                  mMeshLightStats;        ///< Stats before/after pre-processing of mesh lights. Do not access this directly, use getStats() which ensures the stats are up-to-date.
        mutable bool                            mStatsValid = false;    ///< True when stats are valid.

        // GPU resources for the mesh lights and emissive triangles.
        Buffer::SharedPtr                       mpTriangleData;         ///< Per-triangle geometry data for emissive triangles (mTriangleCount elements).
        Buffer::SharedPtr                       mpActiveTriangleList;   ///< List of active (non-culled) emissive triangle.
        Buffer::SharedPtr                       mpTriToActiveList;      ///< Mapping of all light triangles to index in mActiveTriangleList.
        Buffer::SharedPtr                       mpFluxData;             ///< Per-triangle flux data for emissive triangles (mTriangleCount elements).
        Buffer::SharedPtr                       mpMeshData;             ///< Per-mesh data for emissive meshes (mMeshLights.size() elements).
        Buffer::SharedPtr                       mpPerMeshInstanceOffset; ///< Per-mesh instance offset into emissive triangles array (Scene::getMeshInstanceCount() elements).

        mutable Buffer::SharedPtr               mpStagingBuffer;        ///< Staging buffer used for retrieving the vertex positions, texture coordinates and light IDs from the GPU.
        GpuFence::SharedPtr                     mpStagingFence;         ///< Fence used for waiting on the staging buffer being filled in.

        Sampler::SharedPtr                      mpSamplerState;         ///< Material sampler for emissive textures.

        // Shader programs.
        struct
        {
            GraphicsProgram::SharedPtr          pProgram;
            GraphicsVars::SharedPtr             pVars;
            GraphicsState::SharedPtr            pState;
            Sampler::SharedPtr                  pPointSampler;      ///< Point sampler for fetching individual texels in integrator. Must use same wrap mode etc. as material sampler.
            Buffer::SharedPtr                   pResultBuffer;      ///< The output of the integration pass is written here. Using raw buffer for fp32 compatibility.
        } mIntegrator;

        ComputePass::SharedPtr                  mpTriangleListBuilder;
        ComputePass::SharedPtr                  mpTrianglePositionUpdater;
        ComputePass::SharedPtr                  mpFinalizeIntegration;

        mutable CPUOutOfDateFlags               mCPUInvalidData = CPUOutOfDateFlags::None;  ///< Flags indicating which CPU data is valid.
        mutable bool                            mStagingBufferValid = true;                 ///< Flag to indicate if the contents of the staging buffer is up-to-date.
    };

    FALCOR_ENUM_CLASS_OPERATORS(LightCollection::CPUOutOfDateFlags);
    FALCOR_ENUM_CLASS_OPERATORS(LightCollection::UpdateFlags);
}
