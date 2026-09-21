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
#include "TextureAlphaMerge.h"
#include "Core/Error.h"
#include "Core/API/Device.h"
#include "Core/Program/Program.h"
#include "Core/Pass/FullScreenPass.h"

namespace Falcor
{

TextureAlphaMerge::TextureAlphaMerge(Device* pDevice) : mpDevice(pDevice)
{
    FALCOR_ASSERT(pDevice);

    // Init the merge data.
    ProgramDesc d;
    d.addShaderLibrary("Plugins/importers/PBRTImporter/TextureAlphaMerge.3d.slang").vsEntry("vsMain").psEntry("psMain");
    mpPass = FullScreenPass::create(ref<Device>(pDevice), d);
    mpFbo = Fbo::create(ref<Device>(pDevice));
    FALCOR_ASSERT(mpPass && mpFbo);

    Sampler::Desc desc;
    desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    desc.setReductionMode(TextureReductionMode::Standard);
    desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    mpLinearSampler = pDevice->createSampler(desc);
}

ref<Texture> TextureAlphaMerge::merge(ref<Texture> pRgbTexture, ref<Texture> pAlphaTexture)
{
    int alphaIndex = getFormatChannelCount(pAlphaTexture->getFormat()) - 1;

    auto rgbSRV = pRgbTexture->getSRV();
    auto alphaSRV = pAlphaTexture->getSRV();

    // Clamp rectangles to the dimensions of the source/dest views.
    const uint32_t rgbMipLevel = rgbSRV->getViewInfo().mostDetailedMip;
    const uint32_t alphaMipLevel = alphaSRV->getViewInfo().mostDetailedMip;
    const uint2 rgbSize(pRgbTexture->getWidth(rgbMipLevel), pRgbTexture->getHeight(rgbMipLevel));
    const uint2 alphaSize(pAlphaTexture->getWidth(alphaMipLevel), pAlphaTexture->getHeight(alphaMipLevel));

    const uint2 resultSize = max(rgbSize, alphaSize);
    const GraphicsState::Viewport dstViewport(0.0f, 0.0f, (float)resultSize.x, (float)resultSize.y, 0.0f, 1.0f);

    ResourceFormat resultFormat = pRgbTexture->getFormat();
    if (!doesFormatHaveAlpha(resultFormat))
    {
        switch (resultFormat)
        {
        case ResourceFormat::BGRX8Unorm:
            resultFormat = ResourceFormat::BGRA8Unorm;
            break;
        case ResourceFormat::BGRX8UnormSrgb:
            resultFormat = ResourceFormat::BGRA8UnormSrgb;
            break;
        /// We just convert everything else into the most commonly used format
        default:
            resultFormat = ResourceFormat::BGRA8UnormSrgb;
        }
    }

    ResourceBindFlags bf = pRgbTexture->getBindFlags() | ResourceBindFlags::RenderTarget;
    ref<Texture> pResult = mpDevice->createTexture2D(resultSize.x, resultSize.y, resultFormat, 1, Resource::kMaxPossible, nullptr, bf);
    auto resultRTV = pResult->getRTV();
    mpFbo->attachColorTarget(
        pResult, 0, resultRTV->getViewInfo().mostDetailedMip, resultRTV->getViewInfo().firstArraySlice, resultRTV->getViewInfo().arraySize
    );

    auto config = mpPass->getRootVar()["gConfig"];
    config["rgbTexture"] = pRgbTexture;
    config["alphaTexture"] = pAlphaTexture;
    config["alphaIndex"] = alphaIndex;
    config["sampler"] = mpLinearSampler;
    mpPass->getState()->setViewport(0, dstViewport);
    mpPass->execute(mpDevice->getRenderContext(), mpFbo, false);

    // // Release the resources we bound
    mpFbo->attachColorTarget(nullptr, 0);

    pResult->generateMips(mpDevice->getRenderContext());

    return pResult;
}

} // namespace Falcor
