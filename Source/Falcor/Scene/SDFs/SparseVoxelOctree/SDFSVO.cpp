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
#include "SDFSVO.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/Math/MathHelpers.h"
#include "Scene/SDFs/SDFVoxelTypes.slang"

namespace Falcor
{
    Buffer::SharedPtr SDFSVO::spSDFSVOGridUnitAABBBuffer;

    namespace
    {
        const std::string kSDFCountSurfaceVoxelsShaderName = "Scene/SDFs/SDFSurfaceVoxelCounter.cs.slang";
        const std::string kSDFSVOBuildLevelFromTextureShaderName = "Scene/SDFs/SparseVoxelOctree/SDFSVOBuildLevelFromTexture.cs.slang";
        const std::string kSDFSVOBuildOctreeFromLevelsShaderName = "Scene/SDFs/SparseVoxelOctree/SDFSVOBuildOctreeFromLevels.cs.slang";
        const std::string kSDFSVOLocationCodeSorterShaderName = "Scene/SDFs/SparseVoxelOctree/SDFSVOLocationCodeSorter.cs.slang";
        const std::string kSDFSVOWriteSVOOffsetsShaderName = "Scene/SDFs/SparseVoxelOctree/SDFSVOWriteSVOOffsets.cs.slang";
        const std::string kSDFSVOBuildOctreeShaderName = "Scene/SDFs/SparseVoxelOctree/SDFSVOBuildOctree.cs.slang";

        uint32_t ceilPow2(uint32_t v)
        {
            if (v == 0) return 1;
            v -= 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return ++v;
        }
    }

    SDFSVO::SharedPtr SDFSVO::create()
    {
#if !FALCOR_NVAPI_AVAILABLE
        throw RuntimeError("SDFSVO requires NVAPI. See installation instructions in README.");
#endif

        if (!spSDFSVOGridUnitAABBBuffer)
        {
            RtAABB unitAABB { float3(-0.5f), float3(0.5f) };
            spSDFSVOGridUnitAABBBuffer = Buffer::create(sizeof(RtAABB), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, &unitAABB);
        }

        return SharedPtr(new SDFSVO());
    }

    size_t SDFSVO::getSize() const
    {
        return (spSDFSVOGridUnitAABBBuffer ? spSDFSVOGridUnitAABBBuffer->getSize() : 0) + (mpSVOBuffer ? mpSVOBuffer->getSize() : 0);
    }

    uint32_t SDFSVO::getMaxPrimitiveIDBits() const
    {
        return bitScanReverse(mSVOElementCount - 1) + 1;
    }

    void SDFSVO::createResources(RenderContext* pRenderContext, bool deleteScratchData)
    {
        if (!mPrimitives.empty())
        {
            throw RuntimeError("An SDFSVO instance cannot be created from primitives!");
        }

        // Create source grid texture to read from.
        if (mpSDFGridTexture && mpSDFGridTexture->getWidth() == mGridWidth + 1)
        {
            pRenderContext->updateTextureData(mpSDFGridTexture.get(), mValues.data());
        }
        else
        {
            mpSDFGridTexture = Texture::create3D(mGridWidth + 1, mGridWidth + 1, mGridWidth + 1, ResourceFormat::R8Snorm, 1, mValues.data());
        }

        if (!mpCountSurfaceVoxelsPass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kSDFCountSurfaceVoxelsShaderName).csEntry("main").setShaderModel("6_5");
            mpCountSurfaceVoxelsPass = ComputePass::create(desc);
        }

        if (!mpSurfaceVoxelCounter)
        {
            mpReadbackFence = GpuFence::create();

            static const uint32_t zero = 0;
            mpSurfaceVoxelCounter = Buffer::create(sizeof(uint32_t), Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
            mpSurfaceVoxelCounterStagingBuffer = Buffer::create(sizeof(uint32_t), Resource::BindFlags::None, Buffer::CpuAccess::Read);
        }
        else
        {
            pRenderContext->clearUAV(mpSurfaceVoxelCounter->getUAV().get(), uint4(0));
        }

        uint32_t finestLevelVoxelCount = 0;

        // Count the number of surface containing voxels in the texture.
        {
            mpCountSurfaceVoxelsPass["CB"]["gGridWidth"] = mGridWidth;
            mpCountSurfaceVoxelsPass["gSDFGrid"] = mpSDFGridTexture;
            mpCountSurfaceVoxelsPass["gTotalVoxelCount"] = mpSurfaceVoxelCounter;
            mpCountSurfaceVoxelsPass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);

            // Copy surface containing voxels count to staging buffer.
            pRenderContext->copyResource(mpSurfaceVoxelCounterStagingBuffer.get(), mpSurfaceVoxelCounter.get());
            pRenderContext->flush(false);
            mpReadbackFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

            // Copy surface containing voxels count from staging buffer to CPU.
            mpReadbackFence->syncCpu();
            const uint32_t* pSurfaceContainingVoxels = reinterpret_cast<const uint32_t*>(mpSurfaceVoxelCounterStagingBuffer->map(Buffer::MapType::Read));
            std::memcpy(&finestLevelVoxelCount, pSurfaceContainingVoxels, sizeof(uint32_t));
            mpSurfaceVoxelCounterStagingBuffer->unmap();
        }

        uint32_t worstCaseTotalVoxels = 0;

        // Calculate worst case total voxel count across all levels.
        for (uint32_t l = 0; l < mLevelCount; l++)
        {
            uint32_t levelWidth = 1 << l;
            uint32_t levelVoxelMax = levelWidth * levelWidth * levelWidth;
            worstCaseTotalVoxels += glm::min(finestLevelVoxelCount, levelVoxelMax);
        }

        // Create the hash table that will store all voxels during the building process.
        uint32_t hashTableCapacity = mpHashTableBuffer ? uint32_t(mpHashTableBuffer->getSize() / sizeof(SDFSVOHashTableVoxel)) : 0;
        if (hashTableCapacity < worstCaseTotalVoxels)
        {
            hashTableCapacity = ceilPow2(worstCaseTotalVoxels);
            mpHashTableBuffer = Buffer::create(hashTableCapacity * sizeof(SDFSVOHashTableVoxel));
            mpLocationCodesBuffer = Buffer::create(hashTableCapacity * sizeof(uint64_t));
        }
        else
        {
            pRenderContext->clearUAV(mpHashTableBuffer->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(mpLocationCodesBuffer->getUAV().get(), uint4(0));
        }

        // Create the building pass for the bottom level.
        if (!mpBuildFinestLevelFromDistanceTexturePass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kSDFSVOBuildLevelFromTextureShaderName).csEntry("main").setShaderModel("6_5");
            mpBuildFinestLevelFromDistanceTexturePass = ComputePass::create(desc, Program::DefineList({ {"FINEST_LEVEL_PASS", "1"} }));
        }

        // Create voxels for the bottom level.
        {
            auto cbVar = mpBuildFinestLevelFromDistanceTexturePass["CB"];
            cbVar["gLevel"] = (mLevelCount - 1);
            cbVar["gNumLevels"] = mLevelCount;
            cbVar["gLevelWidth"] = mGridWidth;
            mpBuildFinestLevelFromDistanceTexturePass["gSDFGrid"] = mpSDFGridTexture;
            mpBuildFinestLevelFromDistanceTexturePass["gLocationCodes"] = mpLocationCodesBuffer;
            auto hashTableVar = mpBuildFinestLevelFromDistanceTexturePass["gVoxelHashTable"];
            hashTableVar["buffer"] = mpHashTableBuffer;
            hashTableVar["capacity"] = hashTableCapacity;
            mpBuildFinestLevelFromDistanceTexturePass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);
        }

        // Create the building pass for the other levels.
        if (!mpBuildLevelFromDistanceTexturePass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kSDFSVOBuildLevelFromTextureShaderName).csEntry("main").setShaderModel("6_5");
            mpBuildLevelFromDistanceTexturePass = ComputePass::create(desc, Program::DefineList({ {"FINEST_LEVEL_PASS", "0"} }));
        }

        // Allocate a buffer that will hold the voxel count for each level except the bottom level (as we already obtained that previously).
        std::vector<uint32_t> voxelCountsPerLevel(mLevelCount, 0);
        voxelCountsPerLevel.back() = finestLevelVoxelCount;
        {
            mpVoxelCountPerLevelBuffer = Buffer::create(sizeof(uint32_t) * (mLevelCount - 1), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, voxelCountsPerLevel.data());
            mpVoxelCountPerLevelStagingBuffer = Buffer::create(sizeof(uint32_t) * (mLevelCount - 1), Resource::BindFlags::None, Buffer::CpuAccess::Read);
        }

        // Create voxels for all the other levels, a voxel is only created if a child voxel has been created for that voxel.
        {
            // Set common shader vars.
            auto cbVar = mpBuildLevelFromDistanceTexturePass["CB"];
            cbVar["gNumLevels"] = mLevelCount;
            mpBuildLevelFromDistanceTexturePass["gSDFGrid"] = mpSDFGridTexture;
            mpBuildLevelFromDistanceTexturePass["gLocationCodes"] = mpLocationCodesBuffer;
            auto hashTableVar = mpBuildLevelFromDistanceTexturePass["gVoxelHashTable"];
            hashTableVar["buffer"] = mpHashTableBuffer;
            hashTableVar["capacity"] = hashTableCapacity;
            mpBuildLevelFromDistanceTexturePass["gVoxelCounts"] = mpVoxelCountPerLevelBuffer;

            for (int32_t l = mLevelCount - 2; l >= 0; l--)
            {
                uint32_t levelWidth = 1 << l;

                cbVar["gLevel"] = l;
                cbVar["gLevelWidth"] = levelWidth;
                mpBuildLevelFromDistanceTexturePass->execute(pRenderContext, levelWidth, levelWidth, levelWidth);
            }
        }

        // Copy child count to staging buffer.
        pRenderContext->copyResource(mpVoxelCountPerLevelStagingBuffer.get(), mpVoxelCountPerLevelBuffer.get());
        pRenderContext->flush(false);
        mpReadbackFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        // Sort the location codes.
        {
            const uint32_t kLocationCodeSorterMaxGroupSize = 128;
            const uint32_t kSDFSVOLocationCodeLocalBMS = 0;
            const uint32_t kSDFSVOLocationCodeLocalDisperse = 1;
            const uint32_t kSDFSVOLocationCodeBigFlip = 2;
            const uint32_t kSDFSVOLocationCodeBigDisperse = 3;

            uint32_t groupSize = glm::min(kLocationCodeSorterMaxGroupSize, hashTableCapacity >> 1);

            if (!mpSortLocationCodesPass)
            {
                Program::DefineList definesList;
                definesList.add("GROUP_SIZE", std::to_string(groupSize));
                definesList.add("BUFFER_SIZE", std::to_string(hashTableCapacity));
                definesList.add("LOCAL_BMS", std::to_string(kSDFSVOLocationCodeLocalBMS));
                definesList.add("LOCAL_DISPERSE", std::to_string(kSDFSVOLocationCodeLocalDisperse));
                definesList.add("BIG_FLIP", std::to_string(kSDFSVOLocationCodeBigFlip));
                definesList.add("BIG_DISPERSE", std::to_string(kSDFSVOLocationCodeBigDisperse));

                Program::Desc desc;
                desc.addShaderLibrary(kSDFSVOLocationCodeSorterShaderName).csEntry("main").setShaderModel("6_5");
                mpSortLocationCodesPass = ComputePass::create(desc, definesList);
            }
            else
            {
                mpSortLocationCodesPass->addDefine("GROUP_SIZE", std::to_string(groupSize));
                mpSortLocationCodesPass->addDefine("BUFFER_SIZE", std::to_string(hashTableCapacity));
            }

            auto cbVar = mpSortLocationCodesPass["CB"];
            mpSortLocationCodesPass["gBuffer"] = mpLocationCodesBuffer;
            uint32_t dispatchSize = hashTableCapacity >> 1;

            // Dispatch a local bitonic merge sort (BMS) that will sort as many elements as possible in group memory.
            pRenderContext->uavBarrier(mpLocationCodesBuffer.get());
            uint32_t doubleGroupSize = groupSize << 1;
            cbVar["gComparisonHeight"] = doubleGroupSize;
            cbVar["gAlgorithm"] = kSDFSVOLocationCodeLocalBMS;
            mpSortLocationCodesPass->execute(pRenderContext, dispatchSize, 1);

            for (uint32_t comparisonHeight = doubleGroupSize << 1; comparisonHeight <= hashTableCapacity; comparisonHeight <<= 1)
            {
                // All flips after the first local BMS step are big e.g., they cannot fit inside a single dispatch.
                cbVar["gComparisonHeight"] = comparisonHeight;
                cbVar["gAlgorithm"] = kSDFSVOLocationCodeBigFlip;
                mpSortLocationCodesPass->execute(pRenderContext, dispatchSize, 1);

                for (uint32_t disperseComparisonHeight = comparisonHeight >> 1; disperseComparisonHeight > 1; disperseComparisonHeight >>= 1)
                {
                    cbVar["gComparisonHeight"] = disperseComparisonHeight;

                    if (disperseComparisonHeight <= doubleGroupSize)
                    {
                        // We can fit all the next disperse steps into a single dispatch, dispatch it then stop disperses.
                        cbVar["gAlgorithm"] = kSDFSVOLocationCodeLocalDisperse;
                        mpSortLocationCodesPass->execute(pRenderContext, dispatchSize, 1);
                        break;
                    }

                    // Dispatch big disperse step.
                    cbVar["gAlgorithm"] = kSDFSVOLocationCodeBigDisperse;
                    mpSortLocationCodesPass->execute(pRenderContext, dispatchSize, 1);
                }
            }
        }

        // Copy child count from staging buffer to CPU.
        mpReadbackFence->syncCpu();
        const uint32_t* pVoxelCountPerLevel = reinterpret_cast<const uint32_t*>(mpVoxelCountPerLevelStagingBuffer->map(Buffer::MapType::Read));
        std::memcpy(voxelCountsPerLevel.data(), pVoxelCountPerLevel, sizeof(uint32_t) * (mLevelCount - 1));
        mpVoxelCountPerLevelStagingBuffer->unmap();

        // Calculate the total number of voxels in the octree.
        mSVOElementCount = 0;
        for (uint32_t l = 0; l < mLevelCount; l++)
        {
            mSVOElementCount += voxelCountsPerLevel[l];
        }


        // Write the sorted location code addresses into the hash table.
        {
            if (!mpWriteSVOOffsetsPass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kSDFSVOWriteSVOOffsetsShaderName).csEntry("main").setShaderModel("6_5");
                mpWriteSVOOffsetsPass = ComputePass::create(desc);
            }

            auto cbVar = mpWriteSVOOffsetsPass["CB"];
            cbVar["gLocationCodeStartOffset"] = hashTableCapacity - mSVOElementCount;
            cbVar["gVoxelCount"] = mSVOElementCount;
            auto hashTableVar = mpWriteSVOOffsetsPass["gVoxelHashTable"];
            hashTableVar["buffer"] = mpHashTableBuffer;
            hashTableVar["capacity"] = hashTableCapacity;
            mpWriteSVOOffsetsPass["gLocationCodes"] = mpLocationCodesBuffer;

            mpWriteSVOOffsetsPass->execute(pRenderContext, mSVOElementCount, 1);
        }

        // Build the octree from the sorted location codes and hash table.
        {
            //// Create or reallocate a buffer for the SVO if required.
            uint32_t requiredSVOSize = mSVOElementCount * sizeof(SDFSVOVoxel);
            if (!mpSVOBuffer || mpSVOBuffer->getSize() < requiredSVOSize)
            {
                mpSVOBuffer = Buffer::create(requiredSVOSize);
            }

            if (!mpBuildOctreePass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kSDFSVOBuildOctreeShaderName).csEntry("main").setShaderModel("6_5");
                mpBuildOctreePass = ComputePass::create(desc);
            }

            // Build the SVO from the levels hash table.
            {
                auto cbVar = mpBuildOctreePass["CB"];
                cbVar["gLocationCodeStartOffset"] = hashTableCapacity - mSVOElementCount;
                cbVar["gVoxelCount"] = mSVOElementCount;
                auto hashTableVar = mpBuildOctreePass["gVoxelHashTable"];
                hashTableVar["buffer"] = mpHashTableBuffer;
                hashTableVar["capacity"] = hashTableCapacity;
                mpBuildOctreePass["gLocationCodes"] = mpLocationCodesBuffer;
                mpBuildOctreePass["gSVO"] = mpSVOBuffer;

                mpBuildOctreePass->execute(pRenderContext, mSVOElementCount, 1);
            }
        }

        if (deleteScratchData)
        {
            mpCountSurfaceVoxelsPass.reset();
            mpBuildFinestLevelFromDistanceTexturePass.reset();
            mpBuildLevelFromDistanceTexturePass.reset();
            mpSortLocationCodesPass.reset();
            mpWriteSVOOffsetsPass.reset();
            mpBuildOctreePass.reset();
            mpSDFGridTexture.reset();
            mpSurfaceVoxelCounter.reset();
            mpSurfaceVoxelCounterStagingBuffer.reset();
            mpHashTableBuffer.reset();
            mpLocationCodesBuffer.reset();
            mpReadbackFence.reset();
        }
    }

    void SDFSVO::setShaderData(const ShaderVar& var) const
    {
        if (!mpSVOBuffer) throw RuntimeError("SDFSVO::setShaderData() can't be called before calling SDFSVO::createResources()!");

        var["svo"] = mpSVOBuffer;
        var["levelCount"] = mLevelCount;
    }

    void SDFSVO::setValuesInternal(const std::vector<float>& cornerValues)
    {
        mLevelCount = bitScanReverse(mGridWidth) + 1;

        uint32_t gridWidthInValues = mGridWidth + 1;
        uint32_t valueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        mValues.resize(valueCount);

        float normalizationMultipler = mGridWidth / (0.5f * glm::root_three<float>());
        for (uint32_t v = 0; v < valueCount; v++)
        {
            float normalizedValue = glm::clamp(cornerValues[v] * normalizationMultipler, -1.0f, 1.0f);

            float integerScale = normalizedValue * float(INT8_MAX);
            mValues[v] = integerScale >= 0.0f ? int8_t(integerScale + 0.5f) : int8_t(integerScale - 0.5f);
        }
    }
}
