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
#include "NDSDFGrid.h"
#include "Core/API/RenderContext.h"

namespace Falcor
{
    Sampler::SharedPtr NDSDFGrid::spNDSDFGridSampler;
    Buffer::SharedPtr NDSDFGrid::spNDSDFGridUnitAABBBuffer;

    NDSDFGrid::SharedPtr NDSDFGrid::create(float normalizationFactor)
    {
        if (!spNDSDFGridSampler)
        {
            Sampler::Desc sdfGridSamplerDesc;
            sdfGridSamplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
            sdfGridSamplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
            spNDSDFGridSampler = Sampler::create(sdfGridSamplerDesc);
        }

        if (!spNDSDFGridUnitAABBBuffer)
        {
            RtAABB unitAABB { float3(-0.5f), float3(0.5f) };
            spNDSDFGridUnitAABBBuffer = Buffer::create(sizeof(RtAABB), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, &unitAABB);
        }

        return SharedPtr(new NDSDFGrid(normalizationFactor));
    }

    size_t NDSDFGrid::getSize() const
    {
        size_t totalSize = spNDSDFGridUnitAABBBuffer->getSize();

        for (const Texture::SharedPtr& pNormalizedVolumeTexture : mNDSDFTextures)
        {
            totalSize += pNormalizedVolumeTexture->getTextureSizeInBytes();
        }

        return totalSize;
    }

    uint32_t NDSDFGrid::getMaxPrimitiveIDBits() const
    {
        return bitScanReverse(uint32_t(mValues.size() - 1)) + 1;
    }

    void NDSDFGrid::createResources(RenderContext* pRenderContext, bool deleteScratchData)
    {
        if (!mPrimitives.empty())
        {
            throw RuntimeError("An NDSDFGrid instance cannot be created from primitives!");
        }

        uint32_t lodCount = (uint32_t)mValues.size();

        if (mNDSDFTextures.empty() || mNDSDFTextures.size() != lodCount)
        {
            mNDSDFTextures.clear();
            mNDSDFTextures.resize(lodCount);
        }

        for (uint32_t lod = 0; lod < lodCount; lod++)
        {
            uint32_t lodWidth = 1 + (mCoarsestLODGridWidth << lod);

            Texture::SharedPtr& pNDSDFTexture = mNDSDFTextures[lod];
            if (pNDSDFTexture && pNDSDFTexture->getWidth() == lodWidth)
            {
                pRenderContext->updateTextureData(pNDSDFTexture.get(), mValues[lod].data());
            }
            else
            {
                pNDSDFTexture = Texture::create3D(lodWidth, lodWidth, lodWidth, ResourceFormat::R8Snorm, 1, mValues[lod].data());
            }
        }
    }

    void NDSDFGrid::setShaderData(const ShaderVar& var) const
    {
        if (mNDSDFTextures.empty())
        {
            throw RuntimeError("NDSDFGrid::setShaderData() can't be called before calling NDSDFGrid::createResources()!");
        }

        var["sampler"] = spNDSDFGridSampler;
        var["lodCount"] = uint32_t(mNDSDFTextures.size());
        var["coarsestLODAsLevel"] = bitScanReverse(mCoarsestLODGridWidth);
        var["coarsestLODGridWidth"] = mCoarsestLODGridWidth;
        var["coarsestLODNormalizationFactor"] = mCoarsestLODNormalizationFactor;
        var["narrowBandThickness"] = mNarrowBandThickness;

        auto texturesVar = var["textures"];
        for (uint32_t lod = 0; lod < mNDSDFTextures.size(); lod++)
        {
            texturesVar[lod] = mNDSDFTextures[lod];
        }
    }

    void NDSDFGrid::setValuesInternal(const std::vector<float>& cornerValues)
    {
        const uint32_t kCoarsestAllowedGridWidth = 8;

        if (kCoarsestAllowedGridWidth > mGridWidth)
        {
            throw RuntimeError("NDSDFGrid::setValues() grid width must be larger than {}.", kCoarsestAllowedGridWidth);
        }

        uint32_t lodCount = bitScanReverse(mGridWidth / kCoarsestAllowedGridWidth) + 1;
        mCoarsestLODGridWidth = mGridWidth >> (lodCount - 1);
        mCoarsestLODNormalizationFactor = calculateNormalizationFactor(mCoarsestLODGridWidth);

        mValues.resize(lodCount);
        uint32_t gridWidthInValues = mGridWidth + 1;

        // Format all corner values to a normalized snorm8 format, where a distance of 1 represents "0.5 * narrowBandThickness" voxels of the current LOD.
        for (uint32_t lod = 0; lod < lodCount; lod++)
        {
            uint32_t lodWidthInVoxels = mCoarsestLODGridWidth << lod;
            uint32_t lodWidthInValues = 1 + lodWidthInVoxels;
            float normalizationFactor = mCoarsestLODNormalizationFactor / float(1 << lod);

            std::vector<int8_t>& lodFormattedValues = mValues[lod];
            lodFormattedValues.resize(lodWidthInValues * lodWidthInValues * lodWidthInValues);

            uint32_t lodReadStride = 1 << (lodCount - lod - 1);
            for (uint32_t z = 0; z < lodWidthInValues; z++)
            {
                for (uint32_t y = 0; y < lodWidthInValues; y++)
                {
                    for (uint32_t x = 0; x < lodWidthInValues; x++)
                    {
                        uint32_t writeLocation = x + lodWidthInValues * (y + lodWidthInValues * z);
                        uint32_t readLocation = lodReadStride * (x + gridWidthInValues * (y + gridWidthInValues * z));

                        float normalizedValue = glm::clamp(cornerValues[readLocation] / normalizationFactor, -1.0f, 1.0f);

                        float integerScale = normalizedValue * float(INT8_MAX);
                        lodFormattedValues[writeLocation] = integerScale >= 0.0f ? int8_t(integerScale + 0.5f) : int8_t(integerScale - 0.5f);
                    }
                }
            }
        }
    }

    float NDSDFGrid::calculateNormalizationFactor(uint32_t gridWidth) const
    {
        return 0.5f * glm::root_three<float>() * mNarrowBandThickness / gridWidth;
    }

    NDSDFGrid::NDSDFGrid(float narrowBandThickness) : mNarrowBandThickness(std::max(narrowBandThickness, 1.0f))
    {
    }
}
