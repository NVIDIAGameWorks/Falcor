/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/API/Texture.h"
#include "Core/API/RtAccelerationStructure.h"
#include <memory>
#include <type_traits>
#include <vector>

namespace Falcor
{
class FALCOR_API CustomAccelerationStructure
{
public:
    enum class BuildMode : uint
    {
        None = 0u,
        FastBuild = 1u,
        FastTrace = 2u,
    };

    enum class UpdateMode : uint
    {
        None = 0u,
        TLASOnly = 1u,
        BLASOnly = 2u,
        All = 3u
    };

    CustomAccelerationStructure(
        ref<Device> pDevice,
        const uint64_t aabbCount,
        const uint64_t aabbGpuAddress,
        const BuildMode buildMode = BuildMode::None,
        const UpdateMode updateMode = UpdateMode::None
    );
    CustomAccelerationStructure(
        ref<Device> pDevice,
        const std::vector<uint64_t>& aabbCount,
        const std::vector<uint64_t>& aabbGpuAddress,
        const BuildMode buildMode = BuildMode::None,
        const UpdateMode updateMode = UpdateMode::None
    );

    ~CustomAccelerationStructure();

    void update(RenderContext* pRenderContext);
    void update(RenderContext* pRenderContext, const uint64_t aabbCount);
    void update(RenderContext* pRenderContext, const std::vector<uint64_t>& aabbCount);

    void bindTlas(const ShaderVar& rootVar, std::string shaderName = "gCustomAccel");

    void setMinBLASUpdateCount(const size_t count) { mMinUpdateAABBCount = count; }

    /** Clears the AABBs. Clears the min.x value to NaN by default (reduces traversal times)
     */
    void clearAABBBuffers(
        RenderContext* pRenderContext,
        const ref<Buffer> pAABBBuffer,
        bool clearToNaN = true,
        ref<Buffer> pCounterBuffer = nullptr
    );
    void clearAABBBuffers(
        RenderContext* pRenderContext,
        const std::vector<ref<Buffer>>& pAABBBuffers,
        bool clearToNaN = true,
        ref<Buffer> pCounterBuffer = nullptr
    );

    /* Inserts a barrier and transitions buffer to the correct state
     */
    void AABBBufferBarrier(RenderContext* pRenderContext, const ref<Buffer> pAABBBuffer);
    void AABBBufferBarrier(RenderContext* pRenderContext, const std::vector<ref<Buffer>>& pAABBBuffers);

private:
    /*  * Creates the acceleration structure.
     * Has to be called at least once to create the AS. buildAccelerationStructure(...) needs to be called after that to build/update the AS
     * Calling it with an existing AS rebuilds it
     */
    void createAccelerationStructure(const std::vector<uint64_t>& aabbCount, const std::vector<uint64_t>& aabbGpuAddress);

    /** Clears and resets all Data
     */
    void clearData();

    /** Creates the TLAS
     */
    void createTopLevelAS();

    /** Creates the BLAS
     */
    void createBottomLevelAS(const std::vector<uint64_t> aabbCount, const std::vector<uint64_t> aabbGpuAddress);

    /* * Builds the acceleration structure. Is needed every time one of the aabb buffers changes
     */
    void buildAccelerationStructure(RenderContext* pRenderContext, const std::vector<uint64_t>& aabbCount, bool updateAABBCount);

    /** Build or Update the TLAS
     */
    void buildTopLevelAS(RenderContext* pRenderContext);

    /** Build the BLAS
     */
    void buildBottomLevelAS(RenderContext* pRenderContext, const std::vector<uint64_t>& aabbCount, bool updateAABBCount);

    // Acceleration Structure helpers Structs
    struct BLASData
    {
        RtAccelerationStructurePrebuildInfo prebuildInfo;
        RtAccelerationStructureBuildInputs buildInputs;
        RtGeometryDesc geomDescs;

        uint64_t blasByteSize = 0;    ///< Maximum result data size for the BLAS build, including padding.
        uint64_t scratchByteSize = 0; ///< Maximum scratch data size for the BLAS build, including padding.
    };

    struct TLASData
    {
        RtAccelerationStructureBuildInputs buildInputs;

        ref<Buffer> pTlas;
        ref<RtAccelerationStructure> pTlasObject; //<API Object
        ref<Buffer> pInstanceDescs;               ///< Buffer holding instance descs for the TLAS.
    };

    ref<Device> mpDevice; // Pointer to the Device

    BuildMode mBuildMode = BuildMode::None;
    UpdateMode mUpdateMode = UpdateMode::None;
    bool mAccelerationStructureWasBuild = false; // To check if update can be performed
    size_t mNumberBlas = 0;
    size_t mBlasScratchMaxSize = 0;
    size_t mMinUpdateAABBCount = 0; // Custom update count. AABB is only update if count is bigger than the min update count

    std::vector<BLASData> mBlasData;
    std::vector<ref<Buffer>> mBlas;
    std::vector<ref<RtAccelerationStructure>> mBlasObjects; // API AS object
    std::vector<RtInstanceDesc> mInstanceDesc;
    ref<Buffer> mBlasScratch;
    RtAccelerationStructurePrebuildInfo mTlasPrebuildInfo = {};
    ref<Buffer> mTlasScratch;
    TLASData mTlas;

    // Custom Clear Pass for AABBs (need NaN in minPoint.x)
    ref<ComputePass> mpClearAABBsPass;
};

} // namespace Falcor
