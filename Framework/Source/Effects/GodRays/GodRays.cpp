/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "GodRays.h"
#include "Graphics/Program/ProgramVars.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"

namespace Falcor
{
    GodRays::UniquePtr GodRays::create(float threshold, float mediumDensity, float mediumDecay, float mediumWeight, int32_t numSamples)
    {
        return GodRays::UniquePtr(new GodRays(threshold, mediumDensity, mediumDecay, mediumWeight, numSamples));
    }

    GodRays::GodRays(float threshold, float mediumDensity, float mediumDecay, float mediumWeight, int32_t numSamples)
        : mMediumDensity(mediumDensity), mMediumDecay(mediumDecay), mMediumWeight(mediumWeight), mNumSamples(numSamples)
    {
        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Effects/GodRays.ps.slang");
        mSrcTexLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gColor");
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);

        mpVars["GodRaySettings"]["gMedia.density"] = mediumDensity;
        mpVars["GodRaySettings"]["gMedia.decay"] = mediumDecay;
        mpVars["GodRaySettings"]["gMedia.weight"] = mediumWeight;
        mpVars["GodRaySettings"]["numSamples"] = numSamples;
        mpVars["GodRaySettings"]["lightIndex"] = 0;

        BlendState::Desc desc;
        desc.setRtBlend(0, true);
        desc.setRtParams(0,
            BlendState::BlendOp::Add, BlendState::BlendOp::Add,
            BlendState::BlendFunc::One, BlendState::BlendFunc::One,
            BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha);

        mpAdditiveBlend = BlendState::create(desc);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);
        mpVars->setSampler("gSampler", mpSampler);

        mpFilter = PassFilter::create(PassFilter::Type::HighPass, threshold);
        mpFilterResultFbo = Fbo::create();

    }

    void GodRays::updateLowResTexture(const Texture::SharedPtr& pTexture)
    {
        // Create FBO if not created already, or properties have changed since last use
        bool createFbo = mpLowResTexture == nullptr;

        float aspectRatio = (float)pTexture->getWidth() / (float)pTexture->getHeight();
        uint32_t lowResHeight = max(pTexture->getHeight() / 4, 512u);
        uint32_t lowResWidth = max(pTexture->getWidth() / 4, (uint32_t)(512.0f * aspectRatio));

        if (createFbo == false)
        {
            createFbo = (lowResWidth != mpLowResTexture->getWidth()) ||
                (lowResHeight != mpLowResTexture->getHeight()) ||
                (pTexture->getFormat() != mpLowResTexture->getFormat());
        }

        if (createFbo)
        {
            mpLowResTexture = Texture::create2D(
                lowResWidth,
                lowResHeight,
                pTexture->getFormat(),
                1,
                1,
                nullptr,
                Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);
        }
    }

    void GodRays::execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo)
    {
        // if number of possible lights change
        //mpBuf = TypedBuffer<uint>::create(, Resource::BindFlags::ShaderResource);
        //mpVars->setTypedBuffer("lightIndices", mpBuf);

        // experimenting with a down sampled image for GodRays
        updateLowResTexture(pFbo->getColorTexture(0));
        pRenderContext->blit(pFbo->getColorTexture(0)->getSRV(), mpLowResTexture->getRTV());

        // Run high-pass filter and attach it to an FBO for blurring
        Texture::SharedPtr pHighPassResult = mpFilter->execute(pRenderContext, mpLowResTexture);
        mpFilterResultFbo->attachColorTarget(pHighPassResult, 0);

        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, pHighPassResult->getSRV());
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlitPass->execute(pRenderContext, nullptr, mpAdditiveBlend);
        pRenderContext->popGraphicsVars();
        pState->popFbo();
    }

    void GodRays::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            if (pGui->addFloatVar("Medium Threshold", mThreshold))
            {
                mpFilter->setThreshold(mThreshold);
            }
            if (pGui->addFloatVar("Medium Density", mMediumDensity))
            {
                mpVars["GodRaySettings"]["gMedia.density"] = mMediumDensity;
            }
            if (pGui->addFloatVar("Medium Decay", mMediumDecay))
            {
                mpVars["GodRaySettings"]["gMedia.decay"] = mMediumDecay;
            }
            if (pGui->addFloatVar("Medium Weight", mMediumWeight))
            {
                mpVars["GodRaySettings"]["gMedia.weight"] = mMediumWeight;
            }
            if (pGui->addIntVar("Num Samples", mNumSamples, 0, 1000))
            {
                mpVars["GodRaySettings"]["numSamples"] = static_cast<float>(mNumSamples);
            }
            if (pGui->addIntVar("Light Index", mLightIndex))
            {
                mpVars["GodRaySettings"]["lightIndex"] = mLightIndex;
            }
            pGui->endGroup();
        }
    }

}
