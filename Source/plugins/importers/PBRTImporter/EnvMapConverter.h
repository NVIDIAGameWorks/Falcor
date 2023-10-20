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

#include "Core/API/Formats.h"
#include "Core/API/Sampler.h"
#include "Core/API/RenderContext.h"
#include "Core/Pass/ComputePass.h"

namespace Falcor::pbrt
{

/**
 * Helper class to convert env map from equal-area octahedral mapping to lat-long mapping.
 */
class EnvMapConverter
{
public:
    EnvMapConverter(ref<Device> pDevice) : mpDevice(pDevice)
    {
        mpComputePass = ComputePass::create(mpDevice, "plugins/importers/PBRTImporter/EnvMapConverter.cs.slang");

        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpSampler = mpDevice->createSampler(desc);
    }

    /**
     * Convert texture from equal-area octahedral mapping to lat-long mapping.
     * The output texture will have resolution [2 * width, height] of the input texture and RGBAF32 format.
     * @param[in] pRenderContext Render context.
     * @param[in] pSrcTexture Source texture with envmap in equal-area octahedral mapping.
     * @return Texture with envmap in lat-long mapping.
     */
    ref<Texture> convertEqualAreaOctToLatLong(RenderContext* pRenderContext, const ref<Texture>& pSrcTexture) const
    {
        FALCOR_ASSERT(pSrcTexture);
        FALCOR_ASSERT(pSrcTexture->getWidth() == pSrcTexture->getHeight());

        uint2 dstDim{pSrcTexture->getWidth() * 2, pSrcTexture->getHeight()};

        ref<Texture> pDstTexture = mpDevice->createTexture2D(
            dstDim.x,
            dstDim.y,
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        auto vars = mpComputePass->getRootVar()["gEnvMapConverter"];
        vars["src"] = pSrcTexture;
        vars["srcSampler"] = mpSampler;
        vars["dst"] = pDstTexture;
        vars["dstDim"] = dstDim;
        mpComputePass->execute(pRenderContext, uint3(dstDim, 1));

        return pDstTexture;
    }

private:
    ref<Device> mpDevice;
    ref<ComputePass> mpComputePass;
    ref<Sampler> mpSampler;
};

} // namespace Falcor::pbrt
