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
#include "SDFSBS.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/API/IndirectCommands.h"
#include "Utils/Math/MathHelpers.h"
#include "Utils/Math/MathConstants.slangh"
#include "Utils/SharedCache.h"
#include "Scene/SDFs/SDFVoxelTypes.slang"

namespace Falcor
{
    namespace
    {
        const std::string kAssignBrickValidityShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSAssignBrickValidityFromSDFieldPass.cs.slang";
        const std::string kResetBrickValidityShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSResetBrickValidity.cs.slang";
        const std::string kCopyIndirectionBufferShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCopyIndirectionBuffer.cs.slang";
        const std::string kCreateBricksFromSDFieldShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCreateBricksFromSDField.cs.slang";

        const std::string kCreateChunksFromPrimitivesShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCreateChunksFromPrimitives.cs.slang";
        const std::string kCompactifyChunksShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCompactifyChunks.cs.slang";
        const std::string kPruneEmptyBricksShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSPruneEmptyBricks.cs.slang";
        const std::string kCreateBricksFromChunksShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSCreateBricksFromChunks.cs.slang";

        const std::string kComputeIntervalSDFieldFromGridShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSComputeIntervalSDFieldFromGrid.cs.slang";
        const std::string kExpandSDFieldShaderName = "Scene/SDFs/SparseBrickSet/SDFSBSExpandSDFieldData.cs.slang";

        const bool kEnableCoarseBrickPruning = true;
        const bool kEnableFineBrickPruning = true;

        // Chunk width must be equal to 4 for now.
        const uint32_t kChunkWidth = 4;
    }

    struct SDFSBS::SharedData
    {
        ref<Sampler> pSampler;

        SharedData(ref<Device> pDevice)
        {
            Sampler::Desc samplerDesc;
            samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
            samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
            pSampler = pDevice->createSampler(samplerDesc);
        }
    };

    static SharedCache<SDFSBS::SharedData, Device*> sSharedCache;

    SDFSBS::SDFSBS(ref<Device> pDevice, uint32_t brickWidth, bool compressed, uint32_t defaultGridWidth)
        : SDFGrid(pDevice)
        , mDefaultGridWidth(defaultGridWidth)
        , mBrickWidth(brickWidth)
        , mCompressed(compressed)
    {
        FALCOR_CHECK(!compressed || (brickWidth + 1) % 4 == 0, "'brickWidth' ({}) must be a multiple of 4 minus 1 for compressed SDFSBSs", brickWidth);


        mpSharedData = sSharedCache.acquire(mpDevice.get(), [this]() { return std::make_shared<SharedData>(mpDevice); });
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
        // No update is performed if the SDF grid isn't dirty or isn't constructed from primitives and should not be created as an empty grid.
        bool isEmpty = mPrimitives.empty() && !mpSDFGridTexture && !mWasEmpty;
        if ((!mPrimitivesDirty || (mPrimitives.empty() && !mHasGridRepresentation)) && !isEmpty) return UpdateFlags::None;

        // Update grid texture, if user loads an sdf-file.
        if (!mSDField.empty())
        {
            createSDFGridTexture(pRenderContext, mSDField);
            mSDField.clear();
        }
        return createResourcesFromPrimitivesAndSDField(pRenderContext, false);
    }

    void SDFSBS::createResources(RenderContext* pRenderContext, bool deleteScratchData)
    {
        FALCOR_ASSERT(pRenderContext);

        // Update grid texture, if user loads an sdf-file.
        if (!mSDField.empty())
        {
            createSDFGridTexture(pRenderContext, mSDField);
            mSDField.clear();
        }

        if (!mPrimitives.empty())
        {
            createResourcesFromPrimitivesAndSDField(pRenderContext, deleteScratchData);
        }
        else if (mPrimitives.empty() && mpSDFGridTexture != nullptr)
        {
            createResourcesFromSDField(pRenderContext, deleteScratchData);
        }
        else
        {
            // Use default value for the grid width if it was not initialized.
            if (mGridWidth == 0)
            {
                mGridWidth = mDefaultGridWidth;
                mInitializedWithPrimitives = true; // Tell the editor that it can save this as primitives.
            }

            // If the SBS has no primitives nor values, then create one empty brick for the renderer to be happy.
            createResourcesFromPrimitivesAndSDField(pRenderContext, deleteScratchData);
        }

        allocatePrimitiveBits();
    }

    void SDFSBS::bindShaderData(const ShaderVar& var) const
    {
        if (!mpBrickAABBsBuffer || !mpIndirectionTexture || !mpBrickTexture)
        {
            FALCOR_THROW("SDFSBS::bindShaderData() can't be called before calling SDFSBS::createResources()!");
        }

        var["aabbs"] = mpBrickAABBsBuffer;
        var["indirectionBuffer"] = mpIndirectionTexture;
        var["bricks"] = mpBrickTexture;
        var["sampler"] = mpSharedData->pSampler;

        var["virtualGridWidth"] = mGridWidth;
        var["virtualBricksPerAxis"] = mVirtualBricksPerAxis;
        var["bricksPerAxis"] = mBricksPerAxis;
        var["brickTextureDimensions"] = mBrickTextureDimensions;
        var["brickWidth"] = mBrickWidth;
        var["normalizationFactor"] = 0.5f * float(M_SQRT3) / mGridWidth;
    }

    void SDFSBS::createResourcesFromSDField(RenderContext* pRenderContext, bool deleteScratchData)
    {
        FALCOR_ASSERT(mpSDFGridTexture && mpSDFGridTexture->getWidth() == mGridWidth + 1);

        // Calculate the maximum number of bricks that could be created.
        mVirtualBricksPerAxis = std::max(mVirtualBricksPerAxis, (uint32_t)std::ceil(float(mGridWidth) / mBrickWidth));
        uint32_t virtualBrickCount = mVirtualBricksPerAxis * mVirtualBricksPerAxis * mVirtualBricksPerAxis;

        // Assign brick validity to the brick validity buffer. If any voxel in a brick contains surface, the brick is valid.
        {
            if (!mpAssignBrickValidityPass)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kAssignBrickValidityShaderName).csEntry("main");

                // For brick widths smaller than 8 brick validation will be performed using group shared memory.
                // For larger brick widths, brick validation will be performed using a global atomic.

                DefineList defines;
                defines.add("GROUP_BRICK_CREATION", mBrickWidth <= 8u ? "1" : "0");
                defines.add("GROUP_WIDTH", std::to_string(std::min(mBrickWidth, 8u)));

                mpAssignBrickValidityPass = ComputePass::create(mpDevice, desc, defines);
            }

            if (!mpIndirectionTexture || mpIndirectionTexture->getWidth() < mVirtualBricksPerAxis)
            {
                mpIndirectionTexture = mpDevice->createTexture3D(mVirtualBricksPerAxis, mVirtualBricksPerAxis, mVirtualBricksPerAxis, ResourceFormat::R32Uint, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpIndirectionTexture->setName("SDFSBS::IndirectionTextureValues");
            }

            if (!mpValidityBuffer || mpValidityBuffer->getElementCount() < virtualBrickCount)
            {
                mpValidityBuffer = mpDevice->createStructuredBuffer(sizeof(uint32_t), virtualBrickCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }
            else
            {
                pRenderContext->clearUAV(mpValidityBuffer->getUAV().get(), uint4(0));
            }

            auto paramBlock = mpAssignBrickValidityPass->getRootVar()["gParamBlock"];
            paramBlock["virtualGridWidth"] = mGridWidth;
            paramBlock["virtualBricksPerAxis"] = mVirtualBricksPerAxis;
            paramBlock["brickWidthInVoxels"] = mBrickWidth;
            paramBlock["sdfGrid"] = mpSDFGridTexture;
            paramBlock["brickValidity"] = mpValidityBuffer;
            mpAssignBrickValidityPass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);
        }

        // Execute a prefix sum over the validity buffer to create an indirection buffer and find the total number of bricks.
        {
            if (!mpPrefixSumPass)
            {
                mpPrefixSumPass = std::make_unique<PrefixSum>(mpDevice);
            }

            if (!mpIndirectionBuffer || mpIndirectionBuffer->getElementCount() < virtualBrickCount)
            {
                mpIndirectionBuffer = mpDevice->createStructuredBuffer(sizeof(uint32_t), virtualBrickCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            }

            if (!mpCountBuffer)
            {
                mpCountBuffer = mpDevice->createBuffer(4, ResourceBindFlags::None, MemoryType::DeviceLocal);
            }

            pRenderContext->copyResource(mpIndirectionBuffer.get(), mpValidityBuffer.get());
            mpPrefixSumPass->execute(pRenderContext, mpIndirectionBuffer, virtualBrickCount, nullptr, mpCountBuffer);
            mBrickCount = fetchCount(pRenderContext, mpCountBuffer);
        }

        // Set invalid bricks in the indirection buffer to an invalid value.
        {
            if (!mpResetBrickValidityPass)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kResetBrickValidityShaderName).csEntry("main");
                mpResetBrickValidityPass = ComputePass::create(mpDevice, desc);
            }

            auto paramBlock = mpResetBrickValidityPass->getRootVar()["gParamBlock"];
            paramBlock["virtualBrickCount"] = virtualBrickCount;
            paramBlock["brickValidity"] = mpValidityBuffer;
            paramBlock["indirectionBuffer"] = mpIndirectionBuffer;
            mpResetBrickValidityPass->execute(pRenderContext, virtualBrickCount, 1);
        }

        // Copy indirection buffer to indirection texture.
        {
            if (!mpCopyIndirectionBufferPass)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kCopyIndirectionBufferShaderName).csEntry("main");
                mpCopyIndirectionBufferPass = ComputePass::create(mpDevice, desc);
            }

            auto paramBlock = mpCopyIndirectionBufferPass->getRootVar()["gParamBlock"];
            paramBlock["virtualBricksPerAxis"] = mVirtualBricksPerAxis;
            paramBlock["indirectionBuffer"] = mpIndirectionBuffer;
            paramBlock["indirectionTexture"] = mpIndirectionTexture;
            mpCopyIndirectionBufferPass->execute(pRenderContext, uint3(mVirtualBricksPerAxis));
        }

        // Create bricks and brick AABBs.
        {
            if (!mpCreateBricksFromSDFieldPass)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kCreateBricksFromSDFieldShaderName).csEntry("main");
                mpCreateBricksFromSDFieldPass = ComputePass::create(mpDevice, desc, { {"COMPRESS_BRICKS", mCompressed ? "1" : "0"} });
            }

            // TextureWidth = kBrickWidthInValues * kBrickWidthInValues * BricksAlongX
            // TextureHeight = kBrickWidthInValues * BricksAlongY
            // TotalBrickCount = BricksAlongX * BricksAlongY
            // Set TextureWidth = TextureHeight and solve for BricksAlongX.
            // This gives: BricksAlongX = ceil(sqrt(TotalNumBricks / kBrickWidthInValues).
            // And: BricksAlongY = ceil(TotalNumBricks / BricksAlongX).
            // This should give TextureWidth ~= TextureHeight.

            uint32_t brickWidthInValues = mBrickWidth + 1;
            uint32_t bricksAlongX = (uint32_t)std::ceil(std::sqrt((float)mBrickCount / brickWidthInValues));
            uint32_t bricksAlongY = (uint32_t)std::ceil((float)mBrickCount / bricksAlongX);

            // Create brick texture.
            if (!mpBrickTexture || mBricksPerAxis.x < bricksAlongX || mBricksPerAxis.y < bricksAlongY)
            {
                mBricksPerAxis = uint2(bricksAlongX, bricksAlongY);

                uint32_t textureWidth = brickWidthInValues * brickWidthInValues * bricksAlongX;
                uint32_t textureHeight = brickWidthInValues * bricksAlongY;

                if (mCompressed)
                {
                    mpBrickTexture = mpDevice->createTexture2D(textureWidth, textureHeight, ResourceFormat::BC4Snorm, 1, 1);

                    // Compression scheme may change the actual width and height to something else.
                    mBrickTextureDimensions = uint2(mpBrickTexture->getWidth(), mpBrickTexture->getHeight());

                    mpBrickScratchTexture = mpDevice->createTexture2D(mBrickTextureDimensions.x / 4, mBrickTextureDimensions.y / 4, ResourceFormat::RG32Int, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess);
                }
                else
                {
                    mpBrickTexture = mpDevice->createTexture2D(textureWidth, textureHeight, ResourceFormat::R8Snorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

                    mBrickTextureDimensions = uint2(textureWidth, textureHeight);
                }
            }

            if (!mpBrickAABBsBuffer || mpBrickAABBsBuffer->getElementCount() < mBrickCount)
            {
                mpBrickAABBsBuffer = mpDevice->createStructuredBuffer(sizeof(AABB), mBrickCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr, false);
            }

            auto paramBlock = mpCreateBricksFromSDFieldPass->getRootVar()["gParamBlock"];
            paramBlock["virtualGridWidth"] = mGridWidth;
            paramBlock["virtualBrickCount"] = virtualBrickCount;
            paramBlock["virtualBricksPerAxis"] = mVirtualBricksPerAxis;
            paramBlock["brickCount"] = mBrickCount;
            paramBlock["brickWidthInVoxels"] = mBrickWidth;
            paramBlock["bricksPerAxis"] = mBricksPerAxis;
            paramBlock["sdfGrid"] = mpSDFGridTexture;
            paramBlock["indirectionBuffer"] = mpIndirectionBuffer;
            paramBlock["brickAABBs"] = mpBrickAABBsBuffer;
            paramBlock["bricks"] = mCompressed ? mpBrickScratchTexture : mpBrickTexture;
            mpCreateBricksFromSDFieldPass->execute(pRenderContext, virtualBrickCount, 1);
        }

        // Copy the uncompressed brick texture to the compressed brick texture.
        if (mCompressed) pRenderContext->copyResource(mpBrickTexture.get(), mpBrickScratchTexture.get());

        if (deleteScratchData)
        {
            mpAssignBrickValidityPass.reset();
            mpPrefixSumPass.reset();
            mpCopyIndirectionBufferPass.reset();
            mpCreateBricksFromSDFieldPass.reset();

            mpBrickScratchTexture.reset();
            mpValidityBuffer.reset();
            mpIndirectionBuffer.reset();
            mpCountBuffer.reset();
        }

        mWasEmpty = false;
    }

    SDFGrid::UpdateFlags SDFSBS::createResourcesFromPrimitivesAndSDField(RenderContext* pRenderContext, bool deleteScratchData)
    {
        // Assume AABBs will change.
        UpdateFlags updateFlags = UpdateFlags::AABBsChanged;
        uint32_t oldGridWidthInValues = mGridWidth + 1;

        // Calculate new width that encapsulate both the values and primitives to form a grid of bricks.
        uint32_t subdivisionCount = (uint32_t)std::ceil(std::log2((float)mGridWidth / mBrickWidth) / std::log2((float)kChunkWidth));
        mVirtualBricksPerAxis = (uint32_t)std::pow((float)kChunkWidth, (float)subdivisionCount);
        mGridWidth = mBrickWidth * mVirtualBricksPerAxis;

        uint32_t gridWidthInValues = mGridWidth + 1;

        if (oldGridWidthInValues != gridWidthInValues)
        {
            logInfo("Updated grid width of SDF Grid from {} to {}.", oldGridWidthInValues-1, mGridWidth);
        }

        // Check if the primitives should be combined with the values.
        bool includeValues = mpSDFGridTexture != nullptr;
        if (!includeValues && mBakePrimitives)
        {
            // Create a grid to hold the values if the grid is missing a value representation but request to bake the primitives.
            mpSDFGridTexture = mpDevice->createTexture3D(gridWidthInValues, gridWidthInValues, gridWidthInValues, ResourceFormat::R8Snorm, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mpSDFGridTexture->setName("SDFSBS::SDFGridTexture");
            pRenderContext->clearUAV(mpSDFGridTexture->getUAV().get(), float4(std::numeric_limits<float>::max()));
            includeValues = true;
            oldGridWidthInValues = gridWidthInValues;
            mSDFieldUpdated = true;
            mCurrentBakedPrimitiveCount = 0;
            mHasGridRepresentation = true;
        }

        bool createEmptyBrick = false;
        // Check if the SDF grid is empty, if it is, create one brick with no surface to make the renderer happy.
        if ((mPrimitives.empty() && !includeValues && !mWasEmpty) || mBuildEmptyGrid)
        {
            mBrickCount = 1;
            if (!mpChunkCoordsBuffer || mpChunkCoordsBuffer->getSize() < mBrickCount * sizeof(uint3))
            {
                uint3 initialData(0);
                mpChunkCoordsBuffer = mpDevice->createBuffer(mBrickCount * sizeof(uint3), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, (const void*)&initialData);
                mpChunkCoordsBuffer->setName("SDFSBS::chunkCoordsBuffer");
            }
            createEmptyBrick = true;
            mBuildEmptyGrid = false;
        }

        if (includeValues)
        {
            // Only expand sdf grid texture if the size has changed.
            if (oldGridWidthInValues != gridWidthInValues && mpSDFGridTexture->getWidth() == oldGridWidthInValues)
            {
                expandSDFGridTexture(pRenderContext, deleteScratchData, oldGridWidthInValues, gridWidthInValues);
            }

            if (mSDFieldUpdated)
            {
                createIntervalSDFieldTextures(pRenderContext, deleteScratchData, kChunkWidth, subdivisionCount);
                mSDFieldUpdated = false;
            }
        }

        // Initialize build passes.
        {
            if (!mpCreateRootChunksFromPrimitives)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kCreateChunksFromPrimitivesShaderName).csEntry("rootEntryPoint");

                DefineList defines;
                defines.add("CHUNK_WIDTH", std::to_string(kChunkWidth));
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpCreateRootChunksFromPrimitives = ComputePass::create(mpDevice, desc, defines);
            }

            if (!mpSubdivideChunksUsingPrimitives)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kCreateChunksFromPrimitivesShaderName).csEntry("subdivideEntryPoint");

                DefineList defines;
                defines.add("CHUNK_WIDTH", std::to_string(kChunkWidth));
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpSubdivideChunksUsingPrimitives = ComputePass::create(mpDevice, desc, defines);
            }

            if (!mpCoarselyPruneEmptyBricks)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kPruneEmptyBricksShaderName).csEntry("coarsePrune");

                DefineList defines;
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpCoarselyPruneEmptyBricks = ComputePass::create(mpDevice, desc, defines);
            }

            if (!mpFinelyPruneEmptyBricks)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kPruneEmptyBricksShaderName).csEntry("finePrune");

                DefineList defines;
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));

                mpFinelyPruneEmptyBricks = ComputePass::create(mpDevice, desc, defines);
            }

            if (!mpCreateBricksFromChunks)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kCreateBricksFromChunksShaderName).csEntry("main");

                DefineList defines;
                defines.add("CHUNK_WIDTH", std::to_string(kChunkWidth));
                defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));
                defines.add("COMPRESS_BRICKS", mCompressed ? "1" : "0");

                mpCreateBricksFromChunks = ComputePass::create(mpDevice, desc, defines);
            }
        }

        uint32_t primitiveCount = (uint32_t)mPrimitives.size();

        // Create or update primitives buffer.
        {
            mpCreateRootChunksFromPrimitives->getRootVar()["gParamBlock"]["primitives"] = mpPrimitivesBuffer;
            mpSubdivideChunksUsingPrimitives->getRootVar()["gParamBlock"]["primitives"] = mpPrimitivesBuffer;
            mpCoarselyPruneEmptyBricks->getRootVar()["gParamBlock"]["primitives"] = mpPrimitivesBuffer;
            mpFinelyPruneEmptyBricks->getRootVar()["gParamBlock"]["primitives"] = mpPrimitivesBuffer;
            mpCreateBricksFromChunks->getRootVar()["gParamBlock"]["primitives"] = mpPrimitivesBuffer;
        }

        if (includeValues)
        {
            mpCreateRootChunksFromPrimitives->addDefine("_BUILD_WITH_SD_FIELD");
            mpSubdivideChunksUsingPrimitives->addDefine("_BUILD_WITH_SD_FIELD");
            mpCoarselyPruneEmptyBricks->addDefine("_BUILD_WITH_SD_FIELD");
            mpFinelyPruneEmptyBricks->addDefine("_BUILD_WITH_SD_FIELD");
            mpCreateBricksFromChunks->addDefine("_BUILD_WITH_SD_FIELD");
        }

        uint32_t currentGridWidth = kChunkWidth;
        uint32_t currentSubChunkCount = currentGridWidth * currentGridWidth * currentGridWidth;
        if (!mpSubChunkValidityBuffer || mpSubChunkValidityBuffer->getSize() < currentSubChunkCount * sizeof(uint32_t))
        {
            mpSubChunkValidityBuffer = mpDevice->createBuffer(currentSubChunkCount * sizeof(uint32_t));
            mpSubChunkValidityBuffer->setName("SDFSBS::SubChunkValidityBuffer");
        }
        else
        {
            pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
        }

        if (!mpSubChunkCoordsBuffer || mpSubChunkCoordsBuffer->getElementCount() < currentSubChunkCount * sizeof(uint3))
        {
            mpSubChunkCoordsBuffer = mpDevice->createBuffer(currentSubChunkCount * sizeof(uint3));
            mpSubChunkCoordsBuffer->setName("SDFSBS::SubChunkCoordsBuffer");
        }

        // Create root chunk(s).
        {
            auto paramBlock = mpCreateRootChunksFromPrimitives->getRootVar()["gParamBlock"];
            paramBlock["primitiveCount"] = primitiveCount - mCurrentBakedPrimitiveCount;
            paramBlock["currentGridWidth"] = currentGridWidth;
            paramBlock["groupCount"] = 1u;
            paramBlock["intervalValues"] = includeValues ? mIntervalSDFieldMaps[subdivisionCount-1] : nullptr;
            paramBlock["subChunkValidity"] = mpSubChunkValidityBuffer;
            paramBlock["subChunkCoords"] = mpSubChunkCoordsBuffer;
            mpCreateRootChunksFromPrimitives->execute(pRenderContext, uint3(currentGridWidth));
        }

        // Create prefix sum pass.
        if (!mpPrefixSumPass)
        {
            mpPrefixSumPass = std::make_unique<PrefixSum>(mpDevice);
        }

        // Create Indirect Args buffer to hold number of groups to be executed for next subdivision.
        if (!mpSubdivisionArgBuffer)
        {
            static const DispatchArguments baseIndirectArgs = { 0, 1, 1 };
            mpSubdivisionArgBuffer = mpDevice->createBuffer(sizeof(DispatchArguments), ResourceBindFlags::IndirectArg, MemoryType::DeviceLocal, &baseIndirectArgs);
        }

        // Subdivisions.
        if (!createEmptyBrick)
        {
            auto subdivideParamBlock = mpSubdivideChunksUsingPrimitives->getRootVar()["gParamBlock"];
            subdivideParamBlock["primitiveCount"] = primitiveCount - mCurrentBakedPrimitiveCount;

            // The root pass performed one subdivision, subtract one from subdivisionCount to get the number of remaining subdivisions.
            uint32_t remainingSubdivisionCount = subdivisionCount - 1;
            for (uint32_t s = 0; s < remainingSubdivisionCount; s++)
            {
                // Update chunk buffers for this subdivision.
                if (!mpChunkIndirectionBuffer || mpChunkIndirectionBuffer->getSize() < mpSubChunkValidityBuffer->getSize())
                {
                    mpChunkIndirectionBuffer = mpDevice->createBuffer(mpSubChunkValidityBuffer->getSize());
                    mpChunkIndirectionBuffer->setName("SDFSBS::ChunkIndirectionBuffer");
                }

                uint32_t currentChunkCount = 0;

                // Execute prefix sum over validity buffer to set up indirection buffer and acquire total chunk count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, currentSubChunkCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, currentSubChunkCount, nullptr, mpSubdivisionArgBuffer);
                    currentChunkCount = fetchCount(pRenderContext, mpSubdivisionArgBuffer);
                }

                if (currentChunkCount != 0)
                {
                    if (!mpChunkCoordsBuffer || mpChunkCoordsBuffer->getSize() < currentChunkCount * sizeof(uint3))
                    {
                        mpChunkCoordsBuffer = mpDevice->createBuffer(currentChunkCount * sizeof(uint3));
                        mpChunkCoordsBuffer->setName("SDFSBS::ChunkCoordsBuffer");
                    }

                    // Compactify the chunk coords, removing invalid chunks.
                    compactifyChunks(pRenderContext, currentSubChunkCount);

                    // Calculate current sub chunk count as a subdivision of the previous valid chunks.
                    currentSubChunkCount = currentChunkCount * (kChunkWidth * kChunkWidth * kChunkWidth);
                    currentGridWidth *= kChunkWidth;

                    // Clear or realloc sub chunk buffers.
                    if (!mpSubChunkValidityBuffer || mpSubChunkValidityBuffer->getSize() < currentSubChunkCount * sizeof(uint32_t))
                    {
                        mpSubChunkValidityBuffer = mpDevice->createBuffer(currentSubChunkCount * sizeof(uint32_t));
                        mpSubChunkValidityBuffer->setName("SDFSBS::subChunkValidityBuffer");
                    }
                    else
                    {
                        pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
                    }

                    if (!mpSubChunkCoordsBuffer || mpSubChunkCoordsBuffer->getElementCount() < currentSubChunkCount * sizeof(uint3))
                    {
                        mpSubChunkCoordsBuffer = mpDevice->createBuffer(currentSubChunkCount * sizeof(uint3));
                        mpSubChunkCoordsBuffer->setName("SDFSBS::subChunkCoordsBuffer");
                    }

                    subdivideParamBlock["currentGridWidth"] = currentGridWidth;
                    subdivideParamBlock["groupCount"] = currentChunkCount;
                    subdivideParamBlock["intervalValues"] = includeValues ? mIntervalSDFieldMaps[subdivisionCount - 2 - s] : nullptr;
                    subdivideParamBlock["chunkCoords"] = mpChunkCoordsBuffer;
                    subdivideParamBlock["subChunkValidity"] = mpSubChunkValidityBuffer;
                    subdivideParamBlock["subChunkCoords"] = mpSubChunkCoordsBuffer;

                    mpSubdivideChunksUsingPrimitives->executeIndirect(pRenderContext, mpSubdivisionArgBuffer.get());
                }
                else
                {
                    mBakePrimitives = false;
                    mPrimitivesDirty = false;
                    return updateFlags;
                }
            }
        }

        // Create bricks from final chunks.
        {
            if (!createEmptyBrick)
            {
                // Update chunk buffer for brick creation.
                if (!mpChunkIndirectionBuffer || mpChunkIndirectionBuffer->getSize() < mpSubChunkValidityBuffer->getSize())
                {
                    mpChunkIndirectionBuffer = mpDevice->createBuffer(mpSubChunkValidityBuffer->getSize());
                    mpChunkIndirectionBuffer->setName("SDFSBS::ChunkIndirectionBuffer");
                }

                // Execute prefix sum over validity buffer to set up indirection buffer and acquire total chunk count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, currentSubChunkCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, currentSubChunkCount, nullptr, mpSubdivisionArgBuffer);
                    mBrickCount = fetchCount(pRenderContext, mpSubdivisionArgBuffer);
                }

                if (!mpChunkCoordsBuffer || mpChunkCoordsBuffer->getSize() < mBrickCount * sizeof(uint3))
                {
                    mpChunkCoordsBuffer = mpDevice->createBuffer(mBrickCount * sizeof(uint3));
                    mpChunkCoordsBuffer->setName("SDFSBS::chunkCoordsBuffer");
                }

                // Compactify the chunk coords, removing invalid chunks.
                compactifyChunks(pRenderContext, currentSubChunkCount);
            }

            // Coarsely prune empty bricks.
            if (kEnableCoarseBrickPruning && !createEmptyBrick && mBrickCount != 0)
            {
                uint32_t preCoarsePruningBrickCount = mBrickCount;

                {
                    pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
                    pRenderContext->copyBufferRegion(mpSubChunkCoordsBuffer.get(), 0, mpChunkCoordsBuffer.get(), 0, preCoarsePruningBrickCount * sizeof(uint3));

                    auto paramBlock = mpCoarselyPruneEmptyBricks->getRootVar()["gParamBlock"];
                    paramBlock["primitiveCount"] = primitiveCount - mCurrentBakedPrimitiveCount;
                    paramBlock["gridWidth"] = mGridWidth;
                    paramBlock["brickCount"] = preCoarsePruningBrickCount;
                    paramBlock["sdfGrid"] = includeValues ? mpSDFGridTexture : nullptr;
                    paramBlock["chunkCoords"] = mpSubChunkCoordsBuffer;
                    paramBlock["chunkValidity"] = mpSubChunkValidityBuffer;
                    mpCoarselyPruneEmptyBricks->execute(pRenderContext, preCoarsePruningBrickCount, 1);
                }

                // Execute another prefix sum over validity buffer to acquire coarsely pruned brick count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, preCoarsePruningBrickCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, preCoarsePruningBrickCount, nullptr, mpSubdivisionArgBuffer);
                    mBrickCount = fetchCount(pRenderContext, mpSubdivisionArgBuffer);
                }

                // Compactify the brick coords, removing coarsely pruned bricks.
                compactifyChunks(pRenderContext, preCoarsePruningBrickCount);
            }

            // Finely prune empty bricks.
            if (kEnableFineBrickPruning && !createEmptyBrick && mBrickCount != 0)
            {
                uint32_t preFinePruningBrickCount = mBrickCount;

                {
                    pRenderContext->clearUAV(mpSubChunkValidityBuffer->getUAV().get(), uint4(0));
                    pRenderContext->copyBufferRegion(mpSubChunkCoordsBuffer.get(), 0, mpChunkCoordsBuffer.get(), 0, preFinePruningBrickCount * sizeof(uint3));

                    auto paramBlock = mpFinelyPruneEmptyBricks->getRootVar()["gParamBlock"];
                    paramBlock["primitiveCount"] = primitiveCount - mCurrentBakedPrimitiveCount;
                    paramBlock["gridWidth"] = mGridWidth;
                    paramBlock["brickCount"] = preFinePruningBrickCount;
                    paramBlock["sdfGrid"] = includeValues ? mpSDFGridTexture : nullptr;
                    paramBlock["chunkCoords"] = mpSubChunkCoordsBuffer;
                    paramBlock["chunkValidity"] = mpSubChunkValidityBuffer;
                    mpFinelyPruneEmptyBricks->executeIndirect(pRenderContext, mpSubdivisionArgBuffer.get());
                }

                // Execute another prefix sum over validity buffer to acquire finely pruned brick count.
                {
                    pRenderContext->copyBufferRegion(mpChunkIndirectionBuffer.get(), 0, mpSubChunkValidityBuffer.get(), 0, preFinePruningBrickCount * sizeof(uint32_t));
                    mpPrefixSumPass->execute(pRenderContext, mpChunkIndirectionBuffer, preFinePruningBrickCount, nullptr, mpSubdivisionArgBuffer);
                    mBrickCount = fetchCount(pRenderContext, mpSubdivisionArgBuffer); // Reusing arg buffer for fetching brick count.
                }

                // Compactify the brick coords, removing finely pruned bricks.
                compactifyChunks(pRenderContext, preFinePruningBrickCount);
            }

            if (mBrickCount != 0)
            {
                // Allocate AABB buffer, indirection buffer and brick texture.
                if (!mpBrickAABBsBuffer || mpBrickAABBsBuffer->getElementCount() < mBrickCount)
                {
                    mpBrickAABBsBuffer = mpDevice->createStructuredBuffer(sizeof(AABB), mBrickCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr, false);
                    mpBrickAABBsBuffer->setName("SDFSBS::BrickAABBsBuffer");
                    updateFlags |= UpdateFlags::BuffersReallocated;
                }

                if (!mpIndirectionTexture || mpIndirectionTexture->getWidth() < mVirtualBricksPerAxis)
                {
                    mpIndirectionTexture = mpDevice->createTexture3D(mVirtualBricksPerAxis, mVirtualBricksPerAxis, mVirtualBricksPerAxis, ResourceFormat::R32Uint, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                    mpIndirectionTexture->setName("SDFSBS::IndirectionTextureValuesPrims");
                    updateFlags |= UpdateFlags::BuffersReallocated;
                }

                pRenderContext->clearUAV(mpIndirectionTexture->getUAV().get(), uint4(std::numeric_limits<uint32_t>::max()));

                uint32_t brickWidthInValues = mBrickWidth + 1;
                uint32_t bricksAlongX = (uint32_t)std::ceil(std::sqrt((float)mBrickCount / brickWidthInValues));
                uint32_t bricksAlongY = (uint32_t)std::ceil((float)mBrickCount / bricksAlongX);

                // Create brick texture.
                if (((!mpBrickScratchTexture && mCompressed) || !mpBrickTexture) || mBricksPerAxis.x < bricksAlongX || mBricksPerAxis.y < bricksAlongY)
                {
                    mBricksPerAxis = uint2(bricksAlongX, bricksAlongY);
                    updateFlags |= UpdateFlags::BuffersReallocated;

                    uint32_t textureWidth = brickWidthInValues * brickWidthInValues * bricksAlongX;
                    uint32_t textureHeight = brickWidthInValues * bricksAlongY;

                    if (mCompressed)
                    {
                        mpBrickTexture = mpDevice->createTexture2D(textureWidth, textureHeight, ResourceFormat::BC4Snorm, 1, 1);
                        mpBrickTexture->setName("SDFSBS::BrickTexture");

                        // Compression scheme may change the actual width and height to something else.
                        mBrickTextureDimensions = uint2(mpBrickTexture->getWidth(), mpBrickTexture->getHeight());

                        mpBrickScratchTexture = mpDevice->createTexture2D(mBrickTextureDimensions.x / 4, mBrickTextureDimensions.y / 4, ResourceFormat::RG32Int, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess);
                    }
                    else
                    {
                        mpBrickTexture = mpDevice->createTexture2D(textureWidth, textureHeight, ResourceFormat::R8Snorm, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
                        mpBrickTexture->setName("SDFSBS::BrickTexture");

                        mBrickTextureDimensions = uint2(textureWidth, textureHeight);
                    }
                }

                // Bake primitives into the SDF grid.
                if (includeValues && mBakePrimitives && mCurrentBakedPrimitiveCount != mBakedPrimitiveCount)
                {
                    if (!mpSDFGridTextureModified || mpSDFGridTextureModified->getWidth() < gridWidthInValues)
                    {
                        mpSDFGridTextureModified = mpDevice->createTexture3D(gridWidthInValues, gridWidthInValues, gridWidthInValues, ResourceFormat::R8Snorm, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                        mpSDFGridTextureModified->setName("SDFSBS::SDFGridTextureModified");
                    }

                    createEvaluatePrimitivesPass(true, true);

                    auto var = mpEvaluatePrimitivesPass->getRootVar();
                    var["CB"]["gGridWidth"] = mGridWidth;
                    var["CB"]["gPrimitiveCount"] = mBakedPrimitiveCount - mCurrentBakedPrimitiveCount;
                    var["gPrimitives"] = mpPrimitivesBuffer;
                    var["gOldValues"] = mpSDFGridTexture;
                    var["gValues"] = mpSDFGridTextureModified;
                    mpEvaluatePrimitivesPass->execute(pRenderContext, uint3(mGridWidth + 1));

                    pRenderContext->copyResource(mpSDFGridTexture.get(), mpSDFGridTextureModified.get());

                    createIntervalSDFieldTextures(pRenderContext, deleteScratchData, kChunkWidth, subdivisionCount);

                    mPrimitivesExcludedFromBuffer = mBakedPrimitiveCount;
                    updatePrimitivesBuffer();
                }

                auto cparamBlock = mpCreateBricksFromChunks->getRootVar()["gParamBlock"];
                cparamBlock["primitiveCount"] = primitiveCount - mBakedPrimitiveCount;
                cparamBlock["gridWidth"] = mGridWidth;
                cparamBlock["brickCount"] = mBrickCount;
                cparamBlock["bricksPerAxis"] = mBricksPerAxis;
                cparamBlock["chunkCoords"] = mpChunkCoordsBuffer;
                cparamBlock["brickAABBs"] = mpBrickAABBsBuffer;
                cparamBlock["indirectionBuffer"] = mpIndirectionTexture;
                cparamBlock["bricks"] = mCompressed ? mpBrickScratchTexture : mpBrickTexture;;
                cparamBlock["sdfGrid"] = includeValues ? mpSDFGridTexture : nullptr;
                mpCreateBricksFromChunks->execute(pRenderContext, mBrickCount, 1);

                // Copy the uncompressed brick texture to the compressed brick texture.
                if (mCompressed) pRenderContext->copyResource(mpBrickTexture.get(), mpBrickScratchTexture.get());
            }
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

            mpSDFGridTextureModified.reset();
            mpCountStagingBuffer.reset();
        }

        mCurrentBakedPrimitiveCount = mBakedPrimitiveCount;
        mWasEmpty = createEmptyBrick;
        if (mBrickCount == 0)
        {
            mBuildEmptyGrid = true;
            mInitializedWithPrimitives = true; // Tell the editor that it can save this as primitives.
            createResourcesFromPrimitivesAndSDField(pRenderContext, deleteScratchData);
        }

        mBakePrimitives = false;
        mPrimitivesDirty = false;
        return updateFlags;
    }

    void SDFSBS::expandSDFGridTexture(RenderContext* pRenderContext, bool deleteScratchData, uint32_t oldGridWidthInValues, uint32_t gridWidthInValues)
    {
        FALCOR_CHECK(oldGridWidthInValues < gridWidthInValues, "Can only expand SDF grid texture if the old width is smaller than the new width.");

        mResolutionScalingFactor = (float)gridWidthInValues / oldGridWidthInValues;

        if (!mpExpandSDFieldPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kExpandSDFieldShaderName).csEntry("main");

            DefineList defines;
            defines.add("GROUP_WIDTH", "8");
            mpExpandSDFieldPass = ComputePass::create(mpDevice, desc, defines);
        }

        // Create source grid texture to read from with the values that should be expanded.
        if (mpSDFGridTexture && mpSDFGridTexture->getWidth() == oldGridWidthInValues)
        {
            if (!mpOldSDFGridTexture || mpOldSDFGridTexture->getWidth() != oldGridWidthInValues)
            {
                mpOldSDFGridTexture = mpDevice->createTexture3D(oldGridWidthInValues, oldGridWidthInValues, oldGridWidthInValues, ResourceFormat::R8Snorm, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpOldSDFGridTexture->setName("SDFSBS::OldSDFGridTexture");
            }

            pRenderContext->copyResource(mpOldSDFGridTexture.get(), mpSDFGridTexture.get());
        }

        // Create destination grid texture to write to.
        if (!mpSDFGridTexture || mpSDFGridTexture->getWidth() < gridWidthInValues)
        {
            mpSDFGridTexture = mpDevice->createTexture3D(gridWidthInValues, gridWidthInValues, gridWidthInValues, ResourceFormat::R8Snorm, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mpSDFGridTexture->setName("SDFSBS::SDFGridTexture");
        }

        // Expand mValues to fit the new grid width, the result is placed in mpSDFGridTexture.
        {
            uint32_t offest = (gridWidthInValues - oldGridWidthInValues) / 2;

            pRenderContext->clearUAV(mpSDFGridTexture->getUAV().get(), float4(std::numeric_limits<float>::max()));

            auto sdFieldData = mpExpandSDFieldPass->getRootVar()["gSDFieldData"];
            sdFieldData["gridWidthInValues"] = oldGridWidthInValues;
            sdFieldData["offset"] = offest;
            sdFieldData["expansionFactor"] = (float)(oldGridWidthInValues - 1) / (gridWidthInValues - 1);
            sdFieldData["oldSDField"] = mpOldSDFGridTexture;
            sdFieldData["newSDField"] = mpSDFGridTexture;
            mpExpandSDFieldPass->execute(pRenderContext, oldGridWidthInValues, oldGridWidthInValues, oldGridWidthInValues);
        }

        if (deleteScratchData)
        {
            mpOldSDFGridTexture.reset();
            mpExpandSDFieldPass.reset();
        }
    }

    void SDFSBS::createIntervalSDFieldTextures(RenderContext* pRenderContext, bool deleteScratchData, uint32_t chunkWidth, uint32_t subdivisionCount)
    {
        if (!mpComputeRootIntervalSDFieldFromGridPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kComputeIntervalSDFieldFromGridShaderName).csEntry("rootGather");

            // For brick widths smaller than 8 interval computation will be performed using group shared memory.
            // For larger brick widths, interval computation will be performed by letting each thread (brick) iterate over all its children.

            DefineList defines;
            defines.add("GROUP_BRICK_CREATION", mBrickWidth <= 8u ? "1" : "0");
            defines.add("GROUP_WIDTH", std::to_string(std::min(mBrickWidth, 8u)));
            defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));
            defines.add("CHUNK_WIDTH", std::to_string(chunkWidth));
            mpComputeRootIntervalSDFieldFromGridPass = ComputePass::create(mpDevice, desc, defines);
        }

        if (!mpComputeIntervalSDFieldFromGridPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kComputeIntervalSDFieldFromGridShaderName).csEntry("chunkGather");

            DefineList defines;
            defines.add("GROUP_BRICK_CREATION", "0");
            defines.add("GROUP_WIDTH", std::to_string(8u));
            defines.add("BRICK_WIDTH", std::to_string(mBrickWidth));
            defines.add("CHUNK_WIDTH", std::to_string(chunkWidth));
            mpComputeIntervalSDFieldFromGridPass = ComputePass::create(mpDevice, desc, defines);
        }

        // Holds the interval data for all levels of division.
        // First is for bricks, and the rest are desimated from the bricks by the division factor of chunkWidth.
        mIntervalSDFieldMaps.resize((size_t)(subdivisionCount));
        uint32_t width = mGridWidth / mBrickWidth;
        for (uint32_t i = 0; i < subdivisionCount; i++)
        {
            auto& pTexture = mIntervalSDFieldMaps[i];
            if (!pTexture || pTexture->getWidth() < width)
            {
                pTexture = mpDevice->createTexture3D(width, width, width, ResourceFormat::RG8Snorm, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                pTexture->setName("SDFSBS::IntervalValuesMaps[" + std::to_string(i) + "]");
            }

            if (i == 0)
            {
                uint32_t threadsWidth = mBrickWidth <= 8u ? mGridWidth : mBrickWidth;
                auto paramBlock = mpComputeRootIntervalSDFieldFromGridPass->getRootVar()["gParamBlock"];
                paramBlock["virtualGridWidth"] = threadsWidth;
                paramBlock["sdfGrid"] = mpSDFGridTexture;
                paramBlock["intervalDistances"] = pTexture;
                mpComputeRootIntervalSDFieldFromGridPass->execute(pRenderContext, threadsWidth, threadsWidth, threadsWidth);
            }
            else
            {
                auto paramBlock = mpComputeIntervalSDFieldFromGridPass->getRootVar()["gParamBlock"];
                paramBlock["virtualGridWidth"] = width;
                paramBlock["sdfGrid"] = mpSDFGridTexture;
                paramBlock["intervalDistances"] = pTexture;
                paramBlock["parentIntervalDistances"] = mIntervalSDFieldMaps[i - 1];
                mpComputeIntervalSDFieldFromGridPass->execute(pRenderContext, width, width, width);
            }

            width /= chunkWidth;
        }

        if (deleteScratchData)
        {
            mIntervalSDFieldMaps.clear();
            mpComputeRootIntervalSDFieldFromGridPass.reset();
            mpComputeIntervalSDFieldFromGridPass.reset();
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
        mSDField.resize(valueCount);

        // The grid is in the size [-1, 1] thus the longest distance that can be stored is sqrt(3) (the length from corner to corner)
        float normalizationFactor = 2.0f * mGridWidth / float(M_SQRT3);
        for (uint32_t v = 0; v < valueCount; v++)
        {
            float normalizedValue = std::clamp(cornerValues[v] * normalizationFactor, -1.0f, 1.0f);
            float integerScale = normalizedValue * float(INT8_MAX);
            mSDField[v] = integerScale >= 0.0f ? int8_t(integerScale + 0.5f) : int8_t(integerScale - 0.5f);
        }
    }

    void SDFSBS::createSDFGridTexture(RenderContext* pRenderContext, const std::vector<int8_t>& sdField)
    {
        FALCOR_CHECK(!sdField.empty(), "Cannot create SDF grid texture from empty values vector");

        if (mpSDFGridTexture && mpSDFGridTexture->getWidth() == mGridWidth + 1)
        {
            pRenderContext->updateTextureData(mpSDFGridTexture.get(), sdField.data());
        }
        else
        {
            mpSDFGridTexture = mpDevice->createTexture3D(mGridWidth + 1, mGridWidth + 1, mGridWidth + 1, ResourceFormat::R8Snorm, 1, sdField.data());
        }

        mSDFieldUpdated = true;
        mCurrentBakedPrimitiveCount = 0;
        mBakedPrimitiveCount = 0;
        mHasGridRepresentation = true;
    }

    uint32_t SDFSBS::fetchCount(RenderContext* pRenderContext, const ref<Buffer>& pBuffer)
    {
        if (!mpCountStagingBuffer)
        {
            mpCountStagingBuffer = mpDevice->createBuffer(4, ResourceBindFlags::None, MemoryType::ReadBack);
        }

        // Copy result to staging buffer.
        pRenderContext->copyBufferRegion(mpCountStagingBuffer.get(), 0, pBuffer.get(), 0, 4);
        pRenderContext->submit(true);

        // Read back final results.
        uint32_t finalResults = *reinterpret_cast<uint32_t*>(mpCountStagingBuffer->map());
        mpCountStagingBuffer->unmap();
        return finalResults;
    }

    void SDFSBS::compactifyChunks(RenderContext* pRenderContext, uint32_t chunkCount)
    {
        if (!mpCompactifyChunks)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kCompactifyChunksShaderName).csEntry("main");
            mpCompactifyChunks = ComputePass::create(mpDevice, desc);
        }

        auto paramBlock = mpCompactifyChunks->getRootVar()["gParamBlock"];
        paramBlock["chunkIndirection"] = mpChunkIndirectionBuffer;
        paramBlock["chunkValidity"] = mpSubChunkValidityBuffer;
        paramBlock["chunkCoords"] = mpSubChunkCoordsBuffer;
        paramBlock["compactedChunkCoords"] = mpChunkCoordsBuffer;
        paramBlock["chunkCount"] = chunkCount;
        mpCompactifyChunks->execute(pRenderContext, chunkCount, 1);
    }
}
