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
#include "Utils/Algorithm/PrefixSum.h"

namespace Falcor
{
    /** A single SDF Sparse Brick Set. Can only be utilized on the GPU.
    */
    class FALCOR_API SDFSBS : public SDFGrid
    {
    public:
        using SharedPtr = std::shared_ptr<SDFSBS>;

        /** Create a new, empty SDF sparse brick set.
            \param[in] brickWidth The width of a brick in voxels.
            \param[in] compressed Selects if bricks should be compressed using lossy BC4 compression. brickWidth + 1 must be a multiple of 4 to enable compression.
            \return SDFSBS object, or nullptr if errors occurred.
        */
        static SharedPtr create(uint32_t brickWidth = 7, bool compressed = false);

        virtual UpdateFlags update(RenderContext* pRenderContext) override;

        uint32_t getVirtualBrickCoordsBitCount() const { return mVirtualBrickCoordsBitCount; }
        uint32_t getBrickLocalVoxelCoordsBrickCount() const { return mBrickLocalVoxelCoordsBitCount; }
        bool isCompressed() const { return mCompressed; }

        virtual size_t getSize() const override;
        virtual uint32_t getMaxPrimitiveIDBits() const override;
        virtual Type getType() const override { return Type::SparseBrickSet; }

        virtual void createResources(RenderContext* pRenderContext, bool deleteScratchData = true) override;

        virtual const Buffer::SharedPtr& getAABBBuffer() const override { return mpBrickAABBsBuffer; }
        virtual uint32_t getAABBCount() const override { return mBrickCount; }

        virtual void setShaderData(const ShaderVar& var) const override;

    protected:
        UpdateFlags createResourcesFromPrimitives(RenderContext* pRenderContext, bool deleteScratchData);
        void createResourcesFromValues(RenderContext* pRenderContext, bool deleteScratchData);

        void allocatePrimitiveBits();

        virtual void setValuesInternal(const std::vector<float>& cornerValues) override;

    private:
        SDFSBS(uint32_t brickWidth, bool compressed);

        // CPU data.
        std::vector<int8_t> mValues;

        // Specs.
        uint32_t mVirtualBricksPerAxis = 0;
        uint32_t mVoxelCount = 0;
        uint32_t mBrickCount = 0;
        uint2 mBricksPerAxis = uint2(0);
        uint2 mBrickTextureDimensions = uint2(0);
        uint32_t mVirtualBrickCoordsBitCount = 0;
        uint32_t mBrickLocalVoxelCoordsBitCount = 0;
        uint32_t mBrickWidth = 0;
        bool mCompressed = false;

        // GPU data.
        Buffer::SharedPtr mpBrickAABBsBuffer;           ///< A compact buffer containing AABBs for each brick.
        Texture::SharedPtr mpIndirectionTexture;        ///< An indirection texture to map from virtual brick coords to actual brick ID.
        Texture::SharedPtr mpBrickTexture;              ///< A texture of SDF bricks with data at corners.

        // Sampler, shared among all SDFSBS instances.
        static Sampler::SharedPtr spSDFSBSSampler;

        // Compute passes used to build the SBS from values.
        ComputePass::SharedPtr mpAssignBrickValidityPass;
        ComputePass::SharedPtr mpResetBrickValidityPass;
        ComputePass::SharedPtr mpCopyIndirectionBufferPass;
        ComputePass::SharedPtr mpCreateBricksFromValuesPass;

        // Compute passes used to build the SBS from values.
        ComputePass::SharedPtr mpCreateRootChunksFromPrimitives;
        ComputePass::SharedPtr mpSubdivideChunksUsingPrimitives;
        ComputePass::SharedPtr mpCompactifyChunks;
        ComputePass::SharedPtr mpCoarselyPruneEmptyBricks;
        ComputePass::SharedPtr mpFinelyPruneEmptyBricks;
        ComputePass::SharedPtr mpCreateBricksFromChunks;

        // Compute passes used to build the SBS from both values and primitives.
        PrefixSum::SharedPtr mpPrefixSumPass;

        // Scratch data used for building from values.
        Texture::SharedPtr mpBrickScratchTexture;
        Buffer::SharedPtr mpIndirectionBuffer;
        Buffer::SharedPtr mpValidityBuffer;
        Texture::SharedPtr mpSDFGridTexture;

        // Scratch data used for building from primitives.
        Buffer::SharedPtr mpChunkIndirectionBuffer;
        Buffer::SharedPtr mpChunkCoordsBuffer;
        Buffer::SharedPtr mpSubChunkValidityBuffer;
        Buffer::SharedPtr mpSubChunkCoordsBuffer;
        Buffer::SharedPtr mpSubdivisionArgBuffer;
        GpuFence::SharedPtr mpReadbackFence;
    };
}
