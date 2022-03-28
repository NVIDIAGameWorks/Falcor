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
#include "stdafx.h"
#include "SDFSBS.h"
#include "Scene/SDFs/SDFVoxelTypes.slang"
#include "Utils/Math/MathHelpers.h"

namespace Falcor
{
    Sampler::SharedPtr SDFSBS::spSDFSBSSampler;

    namespace
    {
        const std::string kAssignBrickValidityShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSAssignBrickValidityFromValuesPass.cs.slang";
        const std::string kResetBrickValidityShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSResetBrickValidity.cs.slang";
        const std::string kCopyIndirectionBufferShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCopyIndirectionBuffer.cs.slang";
        const std::string kCreateBricksFromValuesShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCreateBricksFromValues.cs.slang";

        const std::string kCreateChunksFromPrimitivesShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCreateChunksFromPrimitives.cs.slang";
        const std::string kCompactifyChunksShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCompactifyChunks.cs.slang";
        const std::string kPruneEmptyBricksShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSPruneEmptyBricks.cs.slang";
        const std::string kCreateBricksFromChunksShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCreateBricksFromChunks.cs.slang";

        const bool kEnableCoarseBrickPruning = true;
        const bool kEnableFineBrickPruning = true;
    }

    SDFSBS::SharedPtr SDFSBS::create(uint32_t brickWidth, bool compressed)
    {
        checkArgument(!compressed || (brickWidth + 1) % 4 == 0, "'brickWidth' ({}) must be a multiple of 4 minus 1 for compressed SDFSBSs", brickWidth);


        if (!spSDFSBSSampler)
        {
            Sampler::Desc samplerDesc;
            samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
            samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
            spSDFSBSSampler = Sampler::create(samplerDesc);
        }

        return SharedPtr(new SDFSBS(brickWidth, compressed));
    }

    size_t SDFSBS::getSize() const
    {
        return  (mpBrickAABBsBuffer ? mpBrickAABBsBuffer->getSize() : 0) +
                (mpIndirectionTexture ? mpIndirectionTexture->getTextureSizeInBytes() : 0) +
                (mpBrickTexture ? mpBrickTexture->getTextureSizeInBytes() : 0);
    }

    uint32_t SDFSBS::getMaxPrimitiveIDBits() const
    {
        return bitScanReverse(mBrickCount - 1) + 1;
    }

    SDFGrid::UpdateFlags SDFSBS::update(RenderContext* pRenderContext)
    {
        // If the SDF grid isn't dirty or isn't constructed from primitives then no update is performed.
        if (!mPrimitivesDirty || mPrimitives.empty()) return UpdateFlags::None;

        return createResourcesFromPrimitives(pRenderContext, false);
    }

    void SDFSBS::createResources(RenderContext* pRenderContext, bool deleteScratchData)
    {
        if (!pRenderContext) pRenderContext = gpDevice->getRenderContext();

        // Calculate the maximum number of bricks that could be created.
        mVirtualBricksPerAxis = std::max(mVirtualBricksPerAxis, (uint32_t)std::ceilf(float(mGridWidth) / mBrickWidth));

        if (!mPrimitives.empty() && mValues.empty())
        {
            createResourcesFromPrimitives(pRenderContext, deleteScratchData);
        }
        else if (mPrimitives.empty() && !mValues.empty())
        {
            createResourcesFromValues(pRenderContext, deleteScratchData);
        }
        else
        {
            throw RuntimeError("SDFSBS::setValues() or SDFSBS::setPrimitives() must be called prior to calling SDFSBS::construct()");
        }

        allocatePrimitiveBits();
    }

    void SDFSBS::setShaderData(const ShaderVar& var) const
    {
        if (!mpBrickAABBsBuffer || !mpIndirectionTexture || !mpBrickTexture)
        {
            throw RuntimeError("SDFSBS::setShaderData() can't be called before calling SDFSBS::createResources()!");
        }

        var["aabbs"] = mpBrickAABBsBuffer;
        var["indirectionBuffer"] = mpIndirectionTexture;
        var["bricks"] = mpBrickTexture;
        var["sampler"] = spSDFSBSSampler;

        var["virtualGridWidth"] = mGridWidth;
        var["virtualBricksPerAxis"] = mVirtualBricksPerAxis;
        var["bricksPerAxis"] = mBricksPerAxis;
        var["brickTextureDimensions"] = mBrickTextureDimensions;
        var["brickWidth"] = mBrickWidth;
        var["normalizationFactor"] = 0.5f * glm::root_three<float>() / mGridWidth;
    }

    SDFGrid::UpdateFlags SDFSBS::createResourcesFromPrimitives(RenderContext* pRenderContext, bool deleteScratchData)
    {
        // Assume AABBs will change.
        UpdateFlags updateFlags = UpdateFlags::AABBsChanged;

        // Chunk width must be equal to 4 for now.
        static const uint32_t kChunkWidth = 4;

        // Calculate the number of subdivision so that the final grid width most closely matches the requested grid width without losing quality.
        uint32_t subdivisionCount = (uint32_t)std::ceilf(std::log2f((float)mGridWidth / mBrickWidth) / std::log2f((float)kChunkWidth));
        mVirtualBricksPerAxis = (uint32_t)std::powf((float)kChunkWidth, (float)subdivisionCount);
        mGridWidth = mBrickWidth * mVirtualBricksPerAxis;

        // Initialize build passes.
        {
            if (!mpCreateRootChunksFromPrimitives)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kCreateChunksFromPrimitivesShaderName).csEntry("rootEntryPoint").setShaderModel("6_5");

                Program::DefineList defines;
                defines.add("CHUNK_WIDTH", std::to_string(kChunkWidth));
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpCreateRootChunksFromPrimitives = ComputePass::create(desc, defines);
            }

            if (!mpSubdivideChunksUsingPrimitives)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kCreateChunksFromPrimitivesShaderName).csEntry("subdivideEntryPoint").setShaderModel("6_5");

                Program::DefineList defines;
                defines.add("CHUNK_WIDTH", std::to_string(kChunkWidth));
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpSubdivideChunksUsingPrimitives = ComputePass::create(desc, defines);
            }

            if (!mpCompactifyChunks)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kCompactifyChunksShaderName).csEntry("main").setShaderModel("6_5");
                mpCompactifyChunks = ComputePass::create(desc);
            }

            if (!mpCoarselyPruneEmptyBricks)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kPruneEmptyBricksShaderName).csEntry("coarsePrune").setShaderModel("6_5");

                Program::DefineList defines;
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpCoarselyPruneEmptyBricks = ComputePass::create(desc, defines);
            }

            if (!mpFinelyPruneEmptyBricks)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kPruneEmptyBricksShaderName).csEntry("finePrune").setShaderModel("6_5");

                Program::DefineList defines;
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpFinelyPruneEmptyBricks = ComputePass::create(desc, defines);
            }

            if (!mpCreateBricksFromChunks)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kCreateBricksFromChunksShaderName).csEntry("main").setShaderModel("6_5");

                Program::DefineList defines;
                defines.add("CHUNK_WIDTH", std::to_string(kChunkWidth));
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));
                defines.add("COMPRESS_BRICKS", mCompressed ? "1" : "0");

                mpCreateBricksFromChunks = ComputePass::create(desc, defines);
            }
        }

        uint32_t primitiveCount = (uint32_t)mPrimitives.size();

        // Create or update primitives buffer.
        {
            mpCreateRootChunksFromPrimitives["gPrimitives"] = mpPrimitivesBuffer;
            mpSubdivideChunksUsingPrimitives["gPrimitives"] = mpPrimitivesBuffer;
            mpCoarselyPruneEmptyBricks["gPrimitives"] = mpPrimitivesBuffer;
            mpFinelyPruneEmptyBricks["gPrimitives"] = mpPrimitivesBuffer;
            mpCreateBricksFromChunks["gPrimitives"] = mpPrimitivesBuffer;
        }

        uint32_t currentGridWidth = kChunkWidth;
        uint32_t currentSubChunkCount = currentGridWidth * currentGridWidth * currentGridWidth;
        if (!mpSubChunkValidityBuffer || mpSubChunkValidityBuffer->getSize() < currentSubChunkCount * sizeof(uint32_t))
        {
            mpSubChunkValidityBuffer = Buffer::create(currentSubChunkCount * sizeof(uint32_t));
        }
        else
        {
            pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
        }

        if (!mpSubChunkCoordsBuffer || mpSubChunkCoordsBuffer->getElementCount() < currentSubChunkCount * sizeof(uint3))
        {
            mpSubChunkCoordsBuffer = Buffer::create(currentSubChunkCount * sizeof(uint3));
        }

        // Create root chunk(s).
        {
            auto cb = mpCreateRootChunksFromPrimitives["CB"];
            cb["gPrimitiveCount"] = primitiveCount;
            cb["gGridWidth"] = currentGridWidth;
            cb["gGroupCount"] = 1;

            mpCreateRootChunksFromPrimitives["gSubChunkValidity"] = mpSubChunkValidityBuffer;
            mpCreateRootChunksFromPrimitives["gSubChunkCoords"] = mpSubChunkCoordsBuffer;
            mpCreateRootChunksFromPrimitives->execute(pRenderContext, uint3(currentGridWidth));
        }

        // Create prefix sum pass.
        if (!mpPrefixSumPass)
        {
            mpPrefixSumPass = PrefixSum::create();
        }

        // Create Indirect Args buffer to hold number of groups to be executed for next subdivision.
        if (!mpSubdivisionArgBuffer)
        {
            static const DispatchArguments baseIndirectArgs = { 0, 1, 1 };
            mpSubdivisionArgBuffer = Buffer::create(sizeof(DispatchArguments), ResourceBindFlags::IndirectArg, Buffer::CpuAccess::None, &baseIndirectArgs);
        }

        // Subdivisions.
        {
            auto cb = mpSubdivideChunksUsingPrimitives["CB"];
            cb["gPrimitiveCount"] = primitiveCount;

            // The root pass performed one subdivision, subtract one from subdivisionCount to get the number of remaining subdivisions.
            uint32_t remainingSubdivisionCount = subdivisionCount - 1;
            for (uint32_t s = 0; s < remainingSubdivisionCount; s++)
            {
                // Update chunk buffers for this subdivision.
                if (!mpChunkIndirectionBuffer || mpChunkIndirectionBuffer->getSize() < mpSubChunkValidityBuffer->getSize())
                {
                    mpChunkIndirectionBuffer = Buffer::create(mpSubChunkValidityBuffer->getSize());
                }

                uint32_t currentChunkCount;

                // Execute prefix sum over validity buffer to set up indirection buffer and acquire total chunk count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, currentSubChunkCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, currentSubChunkCount, &currentChunkCount, mpSubdivisionArgBuffer);
                }

                if (!mpChunkCoordsBuffer || mpChunkCoordsBuffer->getSize() < currentChunkCount * sizeof(uint3))
                {
                    mpChunkCoordsBuffer = Buffer::create(currentChunkCount * sizeof(uint3));
                }

                // Compactify the chunk coords, removing invalid chunks.
                {
                    mpCompactifyChunks["gChunkIndirection"] = mpChunkIndirectionBuffer;
                    mpCompactifyChunks["gChunkValidity"] = mpSubChunkValidityBuffer;
                    mpCompactifyChunks["gChunkCoords"] = mpSubChunkCoordsBuffer;
                    mpCompactifyChunks["gCompactedChunkCoords"] = mpChunkCoordsBuffer;
                    mpCompactifyChunks["CB"]["gChunkCount"] = currentSubChunkCount;

                    mpCompactifyChunks->execute(pRenderContext, currentSubChunkCount, 1);
                }

                // Calculate current sub chunk count as a subdivision of the previous valid chunks.
                currentSubChunkCount = currentChunkCount * (kChunkWidth * kChunkWidth * kChunkWidth);
                currentGridWidth *= kChunkWidth;

                // Clear or realloc sub chunk buffers.
                if (!mpSubChunkValidityBuffer || mpSubChunkValidityBuffer->getSize() < currentSubChunkCount * sizeof(uint32_t))
                {
                    mpSubChunkValidityBuffer = Buffer::create(currentSubChunkCount * sizeof(uint32_t));
                    mpSubChunkValidityBuffer->setName("SDFSBS::subChunkValidityBuffer");
                }
                else
                {
                    pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
                }

                if (!mpSubChunkCoordsBuffer || mpSubChunkCoordsBuffer->getElementCount() < currentSubChunkCount * sizeof(uint3))
                {
                    mpSubChunkCoordsBuffer = Buffer::create(currentSubChunkCount * sizeof(uint3));
                    mpSubChunkCoordsBuffer->setName("SDFSBS::subChunkCoordsBuffer");
                }

                cb["gGridWidth"] = currentGridWidth;
                cb["gGroupCount"] = currentChunkCount;

                mpSubdivideChunksUsingPrimitives["gChunkCoords"] = mpChunkCoordsBuffer;
                mpSubdivideChunksUsingPrimitives["gSubChunkValidity"] = mpSubChunkValidityBuffer;
                mpSubdivideChunksUsingPrimitives["gSubChunkCoords"] = mpSubChunkCoordsBuffer;

                mpSubdivideChunksUsingPrimitives->executeIndirect(pRenderContext, mpSubdivisionArgBuffer.get());
            }
        }

        // Create bricks from final chunks.
        {
            // Update chunk buffer for brick creation.
            if (!mpChunkIndirectionBuffer || mpChunkIndirectionBuffer->getSize() < mpSubChunkValidityBuffer->getSize())
            {
                mpChunkIndirectionBuffer = Buffer::create(mpSubChunkValidityBuffer->getSize());
            }

            // Execute prefix sum over validity buffer to set up indirection buffer and acquire total chunk count.
            {
                pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, currentSubChunkCount * sizeof(uint32_t));
                mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, currentSubChunkCount, &mBrickCount, mpSubdivisionArgBuffer);
            }

            if (!mpChunkCoordsBuffer || mpChunkCoordsBuffer->getSize() < mBrickCount * sizeof(uint3))
            {
                mpChunkCoordsBuffer = Buffer::create(mBrickCount * sizeof(uint3));
                mpChunkCoordsBuffer->setName("SDFSBS::chunkCoordsBuffer");
            }

            // Compactify the chunk coords, removing invalid chunks.
            {
                mpCompactifyChunks["gChunkIndirection"] = mpChunkIndirectionBuffer;
                mpCompactifyChunks["gChunkValidity"] = mpSubChunkValidityBuffer;
                mpCompactifyChunks["gChunkCoords"] = mpSubChunkCoordsBuffer;
                mpCompactifyChunks["gCompactedChunkCoords"] = mpChunkCoordsBuffer;
                mpCompactifyChunks["CB"]["gChunkCount"] = currentSubChunkCount;

                mpCompactifyChunks->execute(pRenderContext, currentSubChunkCount, 1);
            }

            // Coarsely prune empty bricks.
            if (kEnableCoarseBrickPruning)
            {
                uint32_t preCoarsePruningBrickCount = mBrickCount;

                {
                    pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
                    pRenderContext->copyBufferRegion(mpSubChunkCoordsBuffer.get(), 0, mpChunkCoordsBuffer.get(), 0, preCoarsePruningBrickCount * sizeof(uint3));

                    auto cb = mpCoarselyPruneEmptyBricks["CB"];
                    cb["gPrimitiveCount"] = primitiveCount;
                    cb["gGridWidth"] = mGridWidth;
                    cb["gBrickCount"] = preCoarsePruningBrickCount;

                    mpCoarselyPruneEmptyBricks["gChunkCoords"] = mpSubChunkCoordsBuffer;
                    mpCoarselyPruneEmptyBricks["gChunkValidity"] = mpSubChunkValidityBuffer;
                    mpCoarselyPruneEmptyBricks->execute(pRenderContext, preCoarsePruningBrickCount, 1);
                }

                // Execute another prefix sum over validity buffer to acquire coarsely pruned brick count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, preCoarsePruningBrickCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, preCoarsePruningBrickCount, &mBrickCount, mpSubdivisionArgBuffer);
                }

                // Compactify the brick coords, removing coarsely pruned bricks.
                {
                    mpCompactifyChunks["gChunkIndirection"] = mpChunkIndirectionBuffer;
                    mpCompactifyChunks["gChunkValidity"] = mpSubChunkValidityBuffer;
                    mpCompactifyChunks["gChunkCoords"] = mpSubChunkCoordsBuffer;
                    mpCompactifyChunks["gCompactedChunkCoords"] = mpChunkCoordsBuffer;
                    mpCompactifyChunks["CB"]["gChunkCount"] = preCoarsePruningBrickCount;

                    mpCompactifyChunks->execute(pRenderContext, preCoarsePruningBrickCount, 1);
                }
            }

            // Finely prune empty bricks.
            if (kEnableFineBrickPruning)
            {
                uint32_t preFinePruningBrickCount = mBrickCount;

                {
                    pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
                    pRenderContext->copyBufferRegion(mpSubChunkCoordsBuffer.get(), 0, mpChunkCoordsBuffer.get(), 0, preFinePruningBrickCount * sizeof(uint3));

                    auto cb = mpFinelyPruneEmptyBricks["CB"];
                    cb["gPrimitiveCount"] = primitiveCount;
                    cb["gGridWidth"] = mGridWidth;
                    cb["gBrickCount"] = preFinePruningBrickCount;

                    mpFinelyPruneEmptyBricks["gChunkCoords"] = mpSubChunkCoordsBuffer;
                    mpFinelyPruneEmptyBricks["gChunkValidity"] = mpSubChunkValidityBuffer;
                    mpFinelyPruneEmptyBricks->executeIndirect(pRenderContext, mpSubdivisionArgBuffer.get());
                }

                // Execute another prefix sum over validity buffer to acquire finely pruned brick count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, preFinePruningBrickCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, preFinePruningBrickCount, &mBrickCount);
                }

                // Compactify the brick coords, removing finely pruned bricks.
                {
                    mpCompactifyChunks["gChunkIndirection"] = mpChunkIndirectionBuffer;
                    mpCompactifyChunks["gChunkValidity"] = mpSubChunkValidityBuffer;
                    mpCompactifyChunks["gChunkCoords"] = mpSubChunkCoordsBuffer;
                    mpCompactifyChunks["gCompactedChunkCoords"] = mpChunkCoordsBuffer;
                    mpCompactifyChunks["CB"]["gChunkCount"] = preFinePruningBrickCount;

                    mpCompactifyChunks->execute(pRenderContext, preFinePruningBrickCount, 1);
                }
            }

            // Allocate AABB buffer, indirection buffer and brick texture.
            if (!mpBrickAABBsBuffer || mpBrickAABBsBuffer->getElementCount() < mBrickCount)
            {
                mpBrickAABBsBuffer = Buffer::createStructured(sizeof(AABB), mBrickCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
                updateFlags |= UpdateFlags::BuffersReallocated;
            }

            if (!mpIndirectionTexture || mpIndirectionTexture->getWidth() < mVirtualBricksPerAxis)
            {
                mpIndirectionTexture = Texture::create3D(mVirtualBricksPerAxis, mVirtualBricksPerAxis, mVirtualBricksPerAxis, ResourceFormat::R32Uint, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                updateFlags |= UpdateFlags::BuffersReallocated;
            }

            pRenderContext->clearUAV(mpIndirectionTexture->getUAV().get(), uint4(std::numeric_limits<uint32_t>::max()));

            uint32_t brickWidthInValues = mBrickWidth + 1;
            uint32_t bricksAlongX = (uint32_t)std::ceilf(std::sqrtf((float)mBrickCount / brickWidthInValues));
            uint32_t bricksAlongY = (uint32_t)std::ceilf((float)mBrickCount / bricksAlongX);

            // Create brick texture.
            if (!mpBrickTexture || mBricksPerAxis.x < bricksAlongX || mBricksPerAxis.y < bricksAlongY)
            {
                mBricksPerAxis = uint2(bricksAlongX, bricksAlongY);
                updateFlags |= UpdateFlags::BuffersReallocated;

                uint32_t textureWidth = brickWidthInValues * brickWidthInValues * bricksAlongX;
                uint32_t textureHeight = brickWidthInValues * bricksAlongY;

                if (mCompressed)
                {
                    mpBrickTexture = Texture::create2D(textureWidth, textureHeight, ResourceFormat::BC4Snorm, 1, 1);

                    // Compression scheme may change the actual width and height to something else.
                    mBrickTextureDimensions = uint2(mpBrickTexture->getWidth(), mpBrickTexture->getHeight());

                    mpBrickScratchTexture = Texture::create2D(mBrickTextureDimensions.x / 4, mBrickTextureDimensions.y / 4, ResourceFormat::RG32Int, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess);
                }
                else
                {
                    mpBrickTexture = Texture::create2D(textureWidth, textureHeight, ResourceFormat::R8Snorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

                    mBrickTextureDimensions = uint2(textureWidth, textureHeight);
                }
            }

            auto cb = mpCreateBricksFromChunks["CB"];
            cb["gPrimitiveCount"] = primitiveCount;
            cb["gGridWidth"] = mGridWidth;
            cb["gBrickCount"] = mBrickCount;
            cb["gBricksPerAxis"] = mBricksPerAxis;

            mpCreateBricksFromChunks["gChunkCoords"] = mpChunkCoordsBuffer;

            mpCreateBricksFromChunks["gBrickAABBs"] = mpBrickAABBsBuffer;
            mpCreateBricksFromChunks["gIndirectionBuffer"] = mpIndirectionTexture;
            mpCreateBricksFromChunks["gBricks"] = mpBrickTexture;
            mpCreateBricksFromChunks->execute(pRenderContext, mBrickCount, 1);

            // Copy the uncompressed brick texture to the compressed brick texture.
            if (mCompressed) pRenderContext->copyResource(mpBrickTexture.get(), mpBrickScratchTexture.get());
        }

        if (deleteScratchData)
        {
            mpCreateRootChunksFromPrimitives.reset();
            mpSubdivideChunksUsingPrimitives.reset();
            mpCompactifyChunks.reset();
            mpCoarselyPruneEmptyBricks.reset();
            mpFinelyPruneEmptyBricks.reset();
            mpCreateBricksFromChunks.reset();

            mpChunkIndirectionBuffer.reset();
            mpChunkCoordsBuffer.reset();
            mpSubChunkValidityBuffer.reset();
            mpSubChunkCoordsBuffer.reset();
            mpSubdivisionArgBuffer.reset();
        }

        mPrimitivesDirty = false;
        return updateFlags;
    }

    void SDFSBS::createResourcesFromValues(RenderContext* pRenderContext, bool deleteScratchData)
    {
        // Create source grid texture to read from.
        if (mpSDFGridTexture && mpSDFGridTexture->getWidth() == mGridWidth + 1)
        {
            pRenderContext->updateTextureData(mpSDFGridTexture.get(), mValues.data());
        }
        else
        {
            mpSDFGridTexture = Texture::create3D(mGridWidth + 1, mGridWidth + 1, mGridWidth + 1, ResourceFormat::R8Snorm, 1, mValues.data());
        }

        // Calculate the maximum number of bricks that could be created.
        mVirtualBricksPerAxis = std::max(mVirtualBricksPerAxis, (uint32_t)std::ceilf(float(mGridWidth) / mBrickWidth));
        uint32_t virtualBrickCount = mVirtualBricksPerAxis * mVirtualBricksPerAxis * mVirtualBricksPerAxis;

        // Assign brick validity to the brick validity buffer. If any voxel in a brick contains surface, the brick is valid.
        {
            if (!mpAssignBrickValidityPass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kAssignBrickValidityShaderName).csEntry("main").setShaderModel("6_5");

                // For brick widths smaller than 8 brick validation will be performed using group shared memory.
                // For larger brick widths, brick validation will be performed using a global atomic.

                Program::DefineList defines;
                defines.add("GROUP_BRICK_CREATION", mBrickWidth <= 8u ? "1" : "0");
                defines.add("GROUP_WIDTH", std::to_string(std::min(mBrickWidth, 8u)));

                mpAssignBrickValidityPass = ComputePass::create(desc, defines);
            }

            if (!mpIndirectionTexture || mpIndirectionTexture->getWidth() < mVirtualBricksPerAxis)
            {
                mpIndirectionTexture = Texture::create3D(mVirtualBricksPerAxis, mVirtualBricksPerAxis, mVirtualBricksPerAxis, ResourceFormat::R32Uint, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }

            if (!mpValidityBuffer || mpValidityBuffer->getElementCount() < virtualBrickCount)
            {
                mpValidityBuffer = Buffer::createStructured(sizeof(uint32_t), virtualBrickCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            }
            else
            {
                pRenderContext->clearUAV(mpValidityBuffer->getUAV().get(), uint4(0));
            }

            mpAssignBrickValidityPass["CB"]["gVirtualGridWidth"] = mGridWidth;
            mpAssignBrickValidityPass["CB"]["gVirtualBricksPerAxis"] = mVirtualBricksPerAxis;
            mpAssignBrickValidityPass["CB"]["gBrickWidthInVoxels"] = mBrickWidth;
            mpAssignBrickValidityPass["gSDFGrid"] = mpSDFGridTexture;
            mpAssignBrickValidityPass["gBrickValidity"] = mpValidityBuffer;
            mpAssignBrickValidityPass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);
        }

        // Execute a prefix sum over the validity buffer to create an indirection buffer and find the total number of bricks.
        {
            if (!mpPrefixSumPass)
            {
                mpPrefixSumPass = PrefixSum::create();
            }

            if (!mpIndirectionBuffer || mpIndirectionBuffer->getElementCount() < virtualBrickCount)
            {
                mpIndirectionBuffer = Buffer::createStructured(sizeof(uint32_t), virtualBrickCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            }

            pRenderContext->copyResource(mpIndirectionBuffer.get(), mpValidityBuffer.get());
            mpPrefixSumPass->execute(pRenderContext, mpIndirectionBuffer, virtualBrickCount, &mBrickCount);
        }

        // Set invalid bricks in the indirection buffer to an invalid value.
        {
            if (!mpResetBrickValidityPass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kResetBrickValidityShaderName).csEntry("main").setShaderModel("6_5");
                mpResetBrickValidityPass = ComputePass::create(desc);
            }

            mpResetBrickValidityPass["CB"]["gVirtualBrickCount"] = virtualBrickCount;
            mpResetBrickValidityPass["gBrickValidity"] = mpValidityBuffer;
            mpResetBrickValidityPass["gIndirectionBuffer"] = mpIndirectionBuffer;
            mpResetBrickValidityPass->execute(pRenderContext, virtualBrickCount, 1);
        }

        // Copy indirection buffer to indirection texture.
        {
            if (!mpCopyIndirectionBufferPass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kCopyIndirectionBufferShaderName).csEntry("main").setShaderModel("6_5");
                mpCopyIndirectionBufferPass = ComputePass::create(desc);
            }

            mpCopyIndirectionBufferPass["CB"]["gVirtualBricksPerAxis"] = mVirtualBricksPerAxis;
            mpCopyIndirectionBufferPass["gIndirectionBuffer"] = mpIndirectionBuffer;
            mpCopyIndirectionBufferPass["gIndirectionTexture"] = mpIndirectionTexture;
            mpCopyIndirectionBufferPass->execute(pRenderContext, uint3(mVirtualBricksPerAxis));
        }

        // Create bricks and brick AABBs.
        {
            if (!mpCreateBricksFromValuesPass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kCreateBricksFromValuesShaderName).csEntry("main").setShaderModel("6_5");
                mpCreateBricksFromValuesPass = ComputePass::create(desc, { {"COMPRESS_BRICKS", mCompressed ? "1" : "0"} });
            }

            // TextureWidth = kBrickWidthInValues * kBrickWidthInValues * BricksAlongX
            // TextureHeight = kBrickWidthInValues * BricksAlongY
            // TotalBrickCount = BricksAlongX * BricksAlongY
            // Set TextureWidth = TextureHeight and solve for BricksAlongX.
            // This gives: BricksAlongX = ceil(sqrt(TotalNumBricks / kBrickWidthInValues).
            // And: BricksAlongY = ceil(TotalNumBricks / BricksAlongX).
            // This should give TextureWidth ~= TextureHeight.

            uint32_t brickWidthInValues = mBrickWidth + 1;
            uint32_t bricksAlongX = (uint32_t)std::ceilf(std::sqrtf((float)mBrickCount / brickWidthInValues));
            uint32_t bricksAlongY = (uint32_t)std::ceilf((float)mBrickCount / bricksAlongX);

            // Create brick texture.
            if (!mpBrickTexture || mBricksPerAxis.x < bricksAlongX || mBricksPerAxis.y < bricksAlongY)
            {
                mBricksPerAxis = uint2(bricksAlongX, bricksAlongY);

                uint32_t textureWidth = brickWidthInValues * brickWidthInValues * bricksAlongX;
                uint32_t textureHeight = brickWidthInValues * bricksAlongY;

                if (mCompressed)
                {
                    mpBrickTexture = Texture::create2D(textureWidth, textureHeight, ResourceFormat::BC4Snorm, 1, 1);

                    // Compression scheme may change the actual width and height to something else.
                    mBrickTextureDimensions = uint2(mpBrickTexture->getWidth(), mpBrickTexture->getHeight());

                    mpBrickScratchTexture = Texture::create2D(mBrickTextureDimensions.x / 4, mBrickTextureDimensions.y / 4, ResourceFormat::RG32Int, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess);
                }
                else
                {
                    mpBrickTexture = Texture::create2D(textureWidth, textureHeight, ResourceFormat::R8Snorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

                    mBrickTextureDimensions = uint2(textureWidth, textureHeight);
                }
            }

            if (!mpBrickAABBsBuffer || mpBrickAABBsBuffer->getElementCount() < mBrickCount)
            {
                mpBrickAABBsBuffer = Buffer::createStructured(sizeof(AABB), mBrickCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            }

            mpCreateBricksFromValuesPass["CB"]["gVirtualGridWidth"] = mGridWidth;
            mpCreateBricksFromValuesPass["CB"]["gVirtualBrickCount"] = virtualBrickCount;
            mpCreateBricksFromValuesPass["CB"]["gVirtualBricksPerAxis"] = mVirtualBricksPerAxis;
            mpCreateBricksFromValuesPass["CB"]["gBrickCount"] = mBrickCount;
            mpCreateBricksFromValuesPass["CB"]["gBrickWidthInVoxels"] = mBrickWidth;
            mpCreateBricksFromValuesPass["CB"]["gBricksPerAxis"] = mBricksPerAxis;
            mpCreateBricksFromValuesPass["gSDFGrid"] = mpSDFGridTexture;
            mpCreateBricksFromValuesPass["gIndirectionBuffer"] = mpIndirectionBuffer;
            mpCreateBricksFromValuesPass["gBrickAABBs"] = mpBrickAABBsBuffer;
            mpCreateBricksFromValuesPass["gBricks"] = mCompressed ? mpBrickScratchTexture : mpBrickTexture;
            mpCreateBricksFromValuesPass->execute(pRenderContext, virtualBrickCount, 1);
        }

        // Copy the uncompressed brick texture to the compressed brick texture.
        if (mCompressed) pRenderContext->copyResource(mpBrickTexture.get(), mpBrickScratchTexture.get());

        if (deleteScratchData)
        {
            mpAssignBrickValidityPass.reset();
            mpPrefixSumPass.reset();
            mpCopyIndirectionBufferPass.reset();
            mpCreateBricksFromValuesPass.reset();

            mpBrickScratchTexture.reset();
            mpValidityBuffer.reset();
            mpIndirectionBuffer.reset();
            mpSDFGridTexture.reset();
        }
    }

    void SDFSBS::allocatePrimitiveBits()
    {
        // Calculate bits required to encode brick coords and brick local voxel coords.
        mVirtualBrickCoordsBitCount = bitScanReverse(mVirtualBricksPerAxis * mVirtualBricksPerAxis * mVirtualBricksPerAxis - 1) + 1;
        mBrickLocalVoxelCoordsBitCount = bitScanReverse((mBrickWidth * mBrickWidth * mBrickWidth) - 1) + 1;
    }

    void SDFSBS::setValuesInternal(const std::vector<float>& cornerValues)
    {
        uint32_t gridWidthInValues = mGridWidth + 1;
        uint32_t valueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        mValues.resize(valueCount);

        float normalizationMultipler = 2.0f * mGridWidth / glm::root_three<float>();
        for (uint32_t v = 0; v < valueCount; v++)
        {
            float normalizedValue = glm::clamp(cornerValues[v] * normalizationMultipler, -1.0f, 1.0f);

            float integerScale = normalizedValue * float(INT8_MAX);
            mValues[v] = integerScale >= 0.0f ? int8_t(integerScale + 0.5f) : int8_t(integerScale - 0.5f);
        }
    }

    SDFSBS::SDFSBS(uint32_t brickWidth, bool compressed) :
        mBrickWidth(brickWidth),
        mCompressed(compressed)
    {
    }
}
