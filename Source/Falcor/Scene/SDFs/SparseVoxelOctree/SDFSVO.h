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

#include "Scene/SDFs/SDFGrid.h"
#include "Core/API/Buffer.h"
#include "Core/API/Texture.h"
#include "RenderGraph/BasePasses/ComputePass.h"

namespace Falcor
{
    /** SDF Sparse Voxel Octree. Can only be utilized on the GPU.
    */
    class FALCOR_API SDFSVO : public SDFGrid
    {
    public:
        using SharedPtr = std::shared_ptr<SDFSVO>;

        /** Create a new, empty SDFSVO.
            \return SDFSVO object, or nullptr if errors occurred.
        */
        static SharedPtr create();

        uint32_t getSVOIndexBitCount() const { return mSVOIndexBitCount; }

        virtual size_t getSize() const override;
        virtual uint32_t getMaxPrimitiveIDBits() const override;

        virtual Type getType() const override { return Type::SparseVoxelOctree; }

        virtual void createResources(RenderContext* pRenderContext, bool deleteScratchData = true) override;

        virtual const Buffer::SharedPtr& getAABBBuffer() const override { return spSDFSVOGridUnitAABBBuffer; }
        virtual uint32_t getAABBCount() const override { return 1; }

        virtual void setShaderData(const ShaderVar& var) const override;

    protected:
        virtual void setValuesInternal(const std::vector<float>& cornerValues) override;

    private:
        SDFSVO() = default;

        // CPU data.
        std::vector<int8_t> mValues;

        // Specs.
        uint32_t mLevelCount = 0;
        uint32_t mSVOElementCount = 0;
        uint32_t mVirtualGridWidth = 0;
        uint32_t mSVOIndexBitCount = 0;

        // GPU Data.
        Buffer::SharedPtr mpSVOBuffer;

        // Resources shared among all SDFSVOs.
        static Buffer::SharedPtr spSDFSVOGridUnitAABBBuffer;

        // Compute passes used to build the SVO.
        ComputePass::SharedPtr mpCountSurfaceVoxelsPass;
        ComputePass::SharedPtr mpBuildFinestLevelFromDistanceTexturePass;
        ComputePass::SharedPtr mpBuildLevelFromDistanceTexturePass;
        ComputePass::SharedPtr mpSortLocationCodesPass;
        ComputePass::SharedPtr mpWriteSVOOffsetsPass;
        ComputePass::SharedPtr mpBuildOctreePass;

        // Scratch data used for building.
        Texture::SharedPtr mpSDFGridTexture;
        Buffer::SharedPtr mpSurfaceVoxelCounter;
        Buffer::SharedPtr mpSurfaceVoxelCounterStagingBuffer;
        Buffer::SharedPtr mpVoxelCountPerLevelBuffer;
        Buffer::SharedPtr mpVoxelCountPerLevelStagingBuffer;
        Buffer::SharedPtr mpHashTableBuffer;
        Buffer::SharedPtr mpLocationCodesBuffer;
        GpuFence::SharedPtr mpReadbackFence;
    };
}
