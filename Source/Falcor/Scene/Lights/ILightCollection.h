/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Object.h"
#include "Utils/Math/VectorTypes.h"

#include <sigs/sigs.h>

#include <vector>
#include <functional>

namespace Falcor
{
    class Device;
    class Scene;
    class RenderContext;
    struct ShaderVar;

    /** Class that holds a collection of mesh lights for a scene.

        Each mesh light is represented by a mesh instance with an emissive material.

        This class has utility functions for updating and pre-processing the mesh lights.
        The LightCollection can be used standalone, but more commonly it will be wrapped
        by an emissive light sampler.
    */
    class FALCOR_API ILightCollection : public Object
    {
    public:
        enum class UpdateFlags : uint32_t
        {
            None                = 0u,   ///< Nothing was changed.
            MatrixChanged       = 1u,   ///< Mesh instance transform changed.
            LayoutChanged       = 2u,   ///< MeshLightData layouts have changed.
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

        using UpdateFlagsSignal = sigs::Signal<void(UpdateFlags)>;

        virtual ~ILightCollection() = default;

        virtual const ref<Device>& getDevice() const = 0;

        /** Updates the light collection to the current state of the scene.
            \param[in] pRenderContext The render context.
            \param[out] pUpdateStatus Stores information about which type of updates were performed for each mesh light. This is an optional output parameter.
            \return True if the lighting in the scene has changed since the last frame.
        */
        virtual bool update(RenderContext* renderContext, UpdateStatus* pUpdateStatus = nullptr) = 0;

        /** Bind the light collection data to a given shader var
            \param[in] var The shader variable to set the data into.
        */
        virtual void bindShaderData(const ShaderVar& var) const = 0;

        /** Returns the total number of active (non-culled) triangle lights.
        */
        uint32_t getActiveLightCount(RenderContext* pRenderContext) const { return getStats(pRenderContext).trianglesActive; }

        /** Returns the total number of triangle lights (may include culled triangles).
        */
        virtual uint32_t getTotalLightCount() const = 0;

        /** Returns stats.
        */
        virtual const MeshLightStats& getStats(RenderContext* pRenderContext) const = 0;

        /** Returns a CPU buffer with all emissive triangles in world space.
            Note that update() must have been called before for the data to be valid.
            Call prepareSyncCPUData() ahead of time to avoid stalling the GPU.
        */
        virtual const std::vector<MeshLightTriangle>& getMeshLightTriangles(RenderContext* pRenderContext) const = 0;

        /** Returns a CPU buffer with all mesh lights.
            Note that update() must have been called before for the data to be valid.
        */
        virtual const std::vector<MeshLightData>& getMeshLights() const = 0;

        /** Prepare for syncing the CPU data.
            If the mesh light triangles will be accessed with getMeshLightTriangles()
            performance can be improved by calling this function ahead of time.
            This function schedules the copies so that it can be read back without delay later.
        */
        virtual void prepareSyncCPUData(RenderContext* pRenderContext) const = 0;

        /** Get the total GPU memory usage in bytes.
        */
        virtual uint64_t getMemoryUsageInBytes() const = 0;

        /** Gets a signal interface that is signaled when the LightCollection is updated.
         */
        virtual UpdateFlagsSignal::Interface getUpdateFlagsSignal() = 0;
    };

    FALCOR_ENUM_CLASS_OPERATORS(ILightCollection::UpdateFlags);
}
