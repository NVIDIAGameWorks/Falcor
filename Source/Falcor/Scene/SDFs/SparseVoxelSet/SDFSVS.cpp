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
#include "SDFSVS.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/Math/MathHelpers.h"
#include "Utils/Math/MathConstants.slangh"
#include "Scene/SDFs/SDFVoxelTypes.slang"

namespace Falcor
{
    namespace
    {
        const std::string kSDFCountSurfaceVoxelsShaderName = "Scene/SDFs/SDFSurfaceVoxelCounter.cs.slang";
        const std::string kSDFSVSVoxelizerShaderName = "Scene/SDFs/SparseVoxelSet/SDFSVSVoxelizer.cs.slang";
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
            FALCOR_THROW("An SDFSVS instance cannot be created from primitives!");
        }

        if (mpSDFGridTexture && mpSDFGridTexture->getWidth() == mGridWidth + 1)
        {
            pRenderContext->updateTextureData(mpSDFGridTexture.get(), mValues.data());
        }
        else
        {
            mpSDFGridTexture = mpDevice->createTexture3D(mGridWidth + 1, mGridWidth + 1, mGridWidth + 1, ResourceFormat::R8Snorm, 1, mValues.data());
        }

        if (!mpCountSurfaceVoxelsPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kSDFCountSurfaceVoxelsShaderName).csEntry("main");
            mpCountSurfaceVoxelsPass = ComputePass::create(mpDevice, desc);
        }

        if (!mpSurfaceVoxelCounter)
        {
            static uint32_t zero = 0;
            mpSurfaceVoxelCounter = mpDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, &zero);
        }
        else
        {
            pRenderContext->clearUAV(mpSurfaceVoxelCounter->getUAV().get(), uint4(0));
        }

        // Count the number of surface containing voxels in the texture.
        {
            auto var = mpCountSurfaceVoxelsPass->getRootVar();
            var["CB"]["gGridWidth"] = mGridWidth;
            var["gSDFGrid"] = mpSDFGridTexture;
            var["gTotalVoxelCount"] = mpSurfaceVoxelCounter;
            mpCountSurfaceVoxelsPass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);
            mVoxelCount = mpSurfaceVoxelCounter->getElement<uint32_t>(0);
        }


        // Create Buffers
        {
            if (!mpVoxelAABBBuffer || mpVoxelAABBBuffer->getElementCount() < mVoxelCount)
            {
                mpVoxelAABBBuffer = mpDevice->createStructuredBuffer(sizeof(AABB), mVoxelCount);
            }

            if (!mpVoxelBuffer || mpVoxelBuffer->getElementCount() < mVoxelCount)
            {
                mpVoxelBuffer = mpDevice->createStructuredBuffer(sizeof(SDFSVSVoxel), mVoxelCount);
            }
        }

        // Create the Sparse Voxel Set.
        {
            if (!mpSDFSVSVoxelizerPass)
            {
                ProgramDesc desc;
                desc.addShaderLibrary(kSDFSVSVoxelizerShaderName).csEntry("main");
                mpSDFSVSVoxelizerPass = ComputePass::create(mpDevice, desc);
            }

            pRenderContext->clearUAVCounter(mpVoxelBuffer, 0);

            auto var = mpSDFSVSVoxelizerPass->getRootVar();
            var["CB"]["gVirtualGridLevel"] = bitScanReverse(mGridWidth) + 1;
            var["CB"]["gVirtualGridWidth"] = mGridWidth;
            var["gSDFGrid"] = mpSDFGridTexture;

            var["gVoxelAABBs"] = mpVoxelAABBBuffer;
            var["gVoxels"] = mpVoxelBuffer;

            mpSDFSVSVoxelizerPass->execute(pRenderContext, mGridWidth, mGridWidth, mGridWidth);
        }

        if (deleteScratchData)
        {
            mpCountSurfaceVoxelsPass.reset();
            mpSurfaceVoxelCounter.reset();
            mpSDFGridTexture.reset();
        }
    }

    void SDFSVS::bindShaderData(const ShaderVar& var) const
    {
        if (!mpVoxelBuffer || !mpVoxelAABBBuffer)
        {
            FALCOR_THROW("SDFSVS::bindShaderData() can't be called before calling SDFSVS::createResources()!");
        }

        var["virtualGridWidth"] = mGridWidth;
        var["oneDivVirtualGridWidth"] = 1.0f / mGridWidth;
        var["normalizationFactor"] = 0.5f * float(M_SQRT3) / mGridWidth;

        var["aabbs"] = mpVoxelAABBBuffer;
        var["voxels"] = mpVoxelBuffer;
    }

    void SDFSVS::setValuesInternal(const std::vector<float>& cornerValues)
    {
        uint32_t gridWidthInValues = mGridWidth + 1;
        uint32_t valueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        mValues.resize(valueCount);

        float normalizationMultipler = 2.0f * mGridWidth / float(M_SQRT3);
        for (uint32_t v = 0; v < valueCount; v++)
        {
            float normalizedValue = std::clamp(cornerValues[v] * normalizationMultipler, -1.0f, 1.0f);

            float integerScale = normalizedValue * float(INT8_MAX);
            mValues[v] = integerScale >= 0.0f ? int8_t(integerScale + 0.5f) : int8_t(integerScale - 0.5f);
        }
    }
}
