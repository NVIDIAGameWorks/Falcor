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
#include "SDFSVS.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/Math/MathHelpers.h"
#include "Scene/SDFs/SDFVoxelTypes.slang"

namespace Falcor
{
    namespace
    {
        const std::string kSDFCountSurfaceVoxelsShaderName = "Scene/SDFs/SDFSurfaceVoxelCounter.cs.slang";
        const std::string kSDFSVSVoxelizerShaderName = "Scene/SDFs/SparseVoxelSet/SDFSVSVoxelizer.cs.slang";
    }

    SDFSVS::SharedPtr SDFSVS::create()
    {
        return SharedPtr(new SDFSVS());
    }

    size_t SDFSVS::getSize() const
    {
        return (mpVoxelBuffer ? mpVoxelBuffer->getSize() : 0) + (mpVoxelAABBBuffer ? mpVoxelAABBBuffer->getSize() : 0);
    }

    uint32_t SDFSVS::getMaxPrimitiveIDBits() const
    {
        return bitScanReverse(uint32_t(mGridWidth * mGridWidth * mGridWidth - 1)) + 1;
    }

    void SDFSVS::createResources(RenderContext* pRenderContext, bool deleteScratchData)
    {
        if (!mPrimitives.empty())
        {
            throw RuntimeError("An SDFSVS instance cannot be created from primitives!");
        }

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
            static uint32_t zero = 0;
            mpSurfaceVoxelCounter = Buffer::create(sizeof(uint32_t), Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
            mpSurfaceVoxelCounterStagingBuffer = Buffer::create(sizeof(uint32_t), Resource::BindFlags::None, Buffer::CpuAccess::Read);
        }
        else
        {
            pRenderContext->clearUAV(mpSurfaceVoxelCounter->getUAV().get(), uint4(0));
        }

        if (!mpReadbackFence)
        {
            mpReadbackFence = GpuFence::create();
        }

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
            std::memcpy(&mVoxelCount, pSurfaceContainingVoxels, sizeof(uint32_t));
            mpSurfaceVoxelCounterStagingBuffer->unmap();
        }


        // Create Buffers
        {
            if (!mpVoxelAABBBuffer || mpVoxelAABBBuffer->getElementCount() < mVoxelCount)
            {
                mpVoxelAABBBuffer = Buffer::createStructured(sizeof(AABB), mVoxelCount);
            }

            if (!mpVoxelBuffer || mpVoxelBuffer->getElementCount() < mVoxelCount)
            {
                mpVoxelBuffer = Buffer::createStructured(sizeof(SDFSVSVoxel), mVoxelCount);
            }
        }

        // Create the Sparse Voxel Set.
        {
            if (!mpSDFSVSVoxelizerPass)
            {
                Program::Desc desc;
                desc.addShaderLibrary(kSDFSVSVoxelizerShaderName).csEntry("main").setShaderModel("6_5");
                mpSDFSVSVoxelizerPass = ComputePass::create(desc);
            }

            pRenderContext->clearUAVCounter(mpVoxelBuffer, 0);

            mpSDFSVSVoxelizerPass["CB"]["gVirtualGridLevel"] = bitScanReverse(mGridWidth) + 1;
            mpSDFSVSVoxelizerPass["CB"]["gVirtualGridWidth"] = mGridWidth;
            mpSDFSVSVoxelizerPass["gSDFGrid"] = mpSDFGridTexture;

            mpSDFSVSVoxelizerPass["gVoxelAABBs"] = mpVoxelAABBBuffer;
            mpSDFSVSVoxelizerPass["gVoxels"] = mpVoxelBuffer;

            mpSDFSVSVoxelizerPass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);
        }

        if (deleteScratchData)
        {
            mpReadbackFence.reset();
            mpCountSurfaceVoxelsPass.reset();
            mpSurfaceVoxelCounter.reset();
            mpSurfaceVoxelCounterStagingBuffer.reset();
            mpSDFGridTexture.reset();
        }
    }

    void SDFSVS::setShaderData(const ShaderVar& var) const
    {
        if (!mpVoxelBuffer || !mpVoxelAABBBuffer)
        {
            throw RuntimeError("SDFSVS::setShaderData() can't be called before calling SDFSVS::createResources()!");
        }

        var["virtualGridLevel"] = bitScanReverse(mGridWidth) + 1;
        var["virtualGridWidth"] = mGridWidth;
        var["normalizationFactor"] = 0.5f * glm::root_three<float>() / mGridWidth;

        var["aabbs"] = mpVoxelAABBBuffer;
        var["voxels"] = mpVoxelBuffer;
    }

    void SDFSVS::setValuesInternal(const std::vector<float>& cornerValues)
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
}
