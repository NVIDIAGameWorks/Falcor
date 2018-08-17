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
#include "Graphics/RenderGraph/RenderPassSerializer.h"

namespace Falcor
{
    GodRays::UniquePtr GodRays::create(float threshold, float mediumDensity, float mediumDecay, float mediumWeight, float exposer, int32_t numSamples)
    {
        return GodRays::UniquePtr(new GodRays(threshold, mediumDensity, mediumDecay, mediumWeight, exposer, numSamples));
    }

    GodRays::GodRays(float threshold, float mediumDensity, float mediumDecay, float mediumWeight, float exposer, int32_t numSamples)
        : mMediumDensity(mediumDensity), mMediumDecay(mediumDecay), mMediumWeight(mediumWeight), mNumSamples(numSamples), mExposer(exposer)
    {
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

        mpFilter = PassFilter::create(PassFilter::Type::HighPass, threshold);
        mpFilterResultFbo = Fbo::create();

        createShader();
    }

    UniquePtr GodRays::deserialize(const RenderPassSerializer& serializer)
    { 
        float threshold = static_cast<float>(serializer.getValue("threshold").d64);
        float mediumDensity = static_cast<float>(serializer.getValue("mediumDensity").d64);
        float mediumDecay = static_cast<float>(serializer.getValue("mediumDecay").d64);
        float mediumWeight = static_cast<float>(serializer.getValue("mediumWeight").d64);
        float exposer = static_cast<float>(serializer.getValue("exposer").d64);
        int32_t numSamples = serializer.getValue("numSamples").i32;

        return create(threshold, mediumDensity, mediumDecay, mediumWeight, exposer, numSamples); 
    }

    void GodRays::serialize()
    {

    }

    void GodRays::createShader()
    {
        Program::DefineList defines;
        defines.add("_NUM_SAMPLES", std::to_string(mNumSamples));

        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Effects/GodRays.ps.slang", defines);
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mSrcTexLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gColor");
        mSrcDepthLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gDepth");
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);

        mpVars["GodRaySettings"]["gMedia.density"] = mMediumDensity;
        mpVars["GodRaySettings"]["gMedia.decay"] = mMediumDecay;
        mpVars["GodRaySettings"]["gMedia.weight"] = mMediumWeight;
        mpVars["GodRaySettings"]["exposer"] = mExposer;
        mpVars["GodRaySettings"]["lightIndex"] = static_cast<uint32_t>(mLightIndex);
        

        mpVars->setSampler("gSampler", mpSampler);
    }

    void GodRays::updateLowResTexture(const Texture::SharedPtr& pTexture)
    {
        // Create FBO if not created already, or properties have changed since last use
        bool createFbo = mpLowResTexture == nullptr;

        float aspectRatio = (float)pTexture->getWidth() / (float)pTexture->getHeight();
        uint32_t lowResHeight = max(pTexture->getHeight() / 4, 256u);
        uint32_t lowResWidth = max(pTexture->getWidth() / 4, (uint32_t)(256.0f * aspectRatio));

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
        execute(pRenderContext, pFbo->getColorTexture(0), pFbo->getDepthStencilTexture(), pFbo);
    }

    void GodRays::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Texture::SharedPtr& pSrcDepthTex, Fbo::SharedPtr pFbo)
    {
        assert(pFbo->getWidth() == pSrcTex->getWidth() && pSrcTex->getHeight() == pFbo->getHeight());

        if (mDirty)
        {
            createShader();
            mDirty = false;
        }

        mpVars["GodRaySettings"]["gMedia.density"] = mMediumDensity;
        mpVars["GodRaySettings"]["gMedia.decay"] = mMediumDecay;
        mpVars["GodRaySettings"]["gMedia.weight"] = mMediumWeight;
        mpVars["GodRaySettings"]["exposer"] = mExposer;
        mpVars["GodRaySettings"]["lightIndex"] = static_cast<uint32_t>(mLightIndex);


        // experimenting with a down sampled image for GodRays
        updateLowResTexture(pSrcTex);
        pRenderContext->blit(pSrcTex->getSRV(), mpLowResTexture->getRTV());

        // Run high-pass filter and attach it to an FBO for blurring
        //Texture::SharedPtr pHighPassResult = mpFilter->execute(pRenderContext, mpLowResTexture);
        mpFilterResultFbo->attachColorTarget(mpLowResTexture, 0);

        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, mpLowResTexture->getSRV());
        mpVars->getDefaultBlock()->setSrv(mSrcDepthLoc, 0, pSrcDepthTex->getSRV());
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlitPass->execute(pRenderContext, nullptr, mpAdditiveBlend);
        pRenderContext->popGraphicsVars();
        pState->popFbo();
    }

    void GodRays::setNumSamples(int32_t numSamples)
    {
        mNumSamples = numSamples;
        mDirty = true;
    }

    void GodRays::renderUI(Gui* pGui, const char* uiGroup, const Scene::SharedPtr& pScene)
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
                setNumSamples(mNumSamples);
            }
            if (pScene)
            {
                if (pScene->getLightCount())
                {
                    Gui::DropdownList lightList;
                    for (uint32_t i = 0; i < pScene->getLightCount(); i++)
                    {
                        Gui::DropdownValue value;
                        value.label = pScene->getLight(i)->getName();
                        value.value = i;
                        lightList.push_back(value);
                    }

                    uint32_t lightIndex = mLightIndex;
                    if (pGui->addDropdown("Source Light", lightList, lightIndex))
                    {
                        mpVars["GodRaySettings"]["lightIndex"] = (mLightIndex = lightIndex);
                    }
                }
            }
            else
            {
                if (pGui->addIntVar("Light Index", mLightIndex, 0, 15))
                {
                    mpVars["GodRaySettings"]["lightIndex"] = mLightIndex;
                }
            }
            if (pGui->addFloatVar("Exposer", mExposer, 0.0f))
            {
                mpVars["GodRaySettings"]["exposer"] = mExposer;
            }
            pGui->endGroup();
        }
    }
}
