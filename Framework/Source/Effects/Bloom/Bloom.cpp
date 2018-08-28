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
#include "Bloom.h"
#include "Graphics/Program/ProgramVars.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"

namespace Falcor
{
    static const char* kSrcName = "src";
    static const char* kDstName = "dst";
    static Gui::DropdownList sDisplayOutput;

    Bloom::SharedPtr Bloom::create(float threshold, uint32_t kernelSize, float sigma)
    {
        return SharedPtr(new Bloom(threshold, kernelSize, sigma));
    }

    Bloom::Bloom(float threshold, uint32_t kernelSize, float sigma)
        : RenderPass("Bloom")
    {
        mpBlur = GaussianBlur::create(kernelSize, sigma);
        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Framework/Shaders/Blit.ps.slang");
        mSrcTexLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gTex");
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);

        BlendState::Desc desc;
        desc.setRtBlend(0, true);
        desc.setRtParams(0, 
            BlendState::BlendOp::Add, BlendState::BlendOp::Add, 
            BlendState::BlendFunc::One, BlendState::BlendFunc::One, 
            BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha);

        mpAdditiveBlend = BlendState::create(desc);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);
        mpVars->setSampler("gSampler", mpSampler);

        mpFilter = PassFilter::create(PassFilter::Type::HighPass, threshold);
        mpFilterResultFbo = Fbo::create();

        Gui::DropdownValue value;
        value.label = "Final Bloom";
        value.value = 0;
        sDisplayOutput.push_back(value);
        value.label = "HighPass Output";
        value.value = 1;
        sDisplayOutput.push_back(value);
        value.label = "Blur Texture";
        value.value = 2;
        sDisplayOutput.push_back(value);
    }

    Bloom::SharedPtr Bloom::deserialize(const RenderPassSerializer& serializer)
    {
        // if empty serializer, use default values
        Scene::UserVariable thresholdVar = serializer.getValue("Bloom.threshold");
        if (thresholdVar.type == Scene::UserVariable::Type::Unknown)
        {
            return create();
        }

        float threshold = static_cast<float>(thresholdVar.d64);
        uint32_t kernelSize = serializer.getValue("Bloom.kernelSize").u32;
        float sigma = static_cast<float>(serializer.getValue("Bloom.sigma").d64);


        return create(threshold, kernelSize, sigma);
    }

    void Bloom::serialize(RenderPassSerializer& renderPassSerializer)
    {
        renderPassSerializer.addVariable("Bloom.threshold", mpFilter->getThreshold());
        renderPassSerializer.addVariable("Bloom.kernelSize", mpBlur->getKernelWidth());
        renderPassSerializer.addVariable("Bloom.sigma", mpBlur->getSigma());
    }

    void Bloom::updateLowResTexture(const Texture::SharedPtr& pTexture)
    {
        // Create FBO if not created already, or properties have changed since last use
        bool createLowResTex = (mpLowResTexture == nullptr);

        float aspectRatio = (float)pTexture->getWidth() / (float)pTexture->getHeight();
        uint32_t lowResHeight = max(pTexture->getHeight() / 4, 256u);
        uint32_t lowResWidth = max(pTexture->getWidth() / 4, (uint32_t)(256.0f * aspectRatio));

        if (createLowResTex == false)
        {
            createLowResTex = (lowResWidth != mpLowResTexture->getWidth()) ||
                (lowResHeight != mpLowResTexture->getHeight()) ||
                (pTexture->getFormat() != mpLowResTexture->getFormat());
        }

        if (createLowResTex)
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

    void Bloom::execute(RenderContext* pRenderContext, const RenderData* pRenderData)
    {
        if (!mpTargetFbo)
        {
            mpTargetFbo = Fbo::create();
        }
        Texture::SharedPtr pSrcTex = pRenderData->getTexture(kSrcName);
        Texture::SharedPtr pDstTex = pRenderData->getTexture(kDstName);
        pRenderContext->blit(pSrcTex->getSRV(), pDstTex->getRTV());
        mpTargetFbo->attachColorTarget(pDstTex, 0);

        execute(pRenderContext, pSrcTex, mpTargetFbo);
    }

    void Bloom::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Fbo::SharedPtr& pFbo)
    {
        updateLowResTexture(pSrcTex);

        pRenderContext->blit(pSrcTex->getSRV(), mpLowResTexture->getRTV());

        if (mOutputMode == OutputMode::HighPassOutput)
        {
            mpFilter->execute(pRenderContext, mpLowResTexture, pFbo);
            return;
        }
        
        // Run high-pass filter and attach it to an FBO for blurring
        Texture::SharedPtr pHighPassResult = mpFilter->execute(pRenderContext, mpLowResTexture);

        mpFilterResultFbo->attachColorTarget(pHighPassResult, 0);
        mpBlur->execute(pRenderContext, pHighPassResult, mpFilterResultFbo);

        // Execute bloom
        if (mOutputMode == OutputMode::FinalBloom)
        {
            mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, pHighPassResult->getSRV());
            GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
            pState->pushFbo(pFbo);
            pRenderContext->pushGraphicsVars(mpVars);

            mpBlitPass->execute(pRenderContext, nullptr, mpAdditiveBlend);

            pRenderContext->popGraphicsVars();
            pState->popFbo();
        }
        else if (mOutputMode == OutputMode::BlurTexture)
        {
            pRenderContext->blit(mpFilterResultFbo->getColorTexture(0)->getSRV(), pFbo->getColorTexture(0)->getRTV());
        }
    }

    void Bloom::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            float threshold = mpFilter->getThreshold();
            if (pGui->addFloatVar("Threshold", threshold, 0.0f)) mpFilter->setThreshold(threshold);

            int32_t kernelWidth = mpBlur->getKernelWidth();
            if (pGui->addIntVar("Kernel Width", (int&)kernelWidth, 1, 15, 2))
            {
                mpBlur->setKernelWidth(kernelWidth);
            }

            float sigma = mpBlur->getSigma();
            if (pGui->addFloatVar("Sigma", sigma, 0.001f))
            {
                mpBlur->setSigma(sigma);
            }

            uint32_t outputMode = static_cast<uint32_t>(mOutputMode);
            if (pGui->addDropdown("Output texture", sDisplayOutput, outputMode))
            {
                mOutputMode = static_cast<OutputMode>(outputMode);
            }

            if (uiGroup) pGui->endGroup();
        }
    }

    void Bloom::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInput(kSrcName);
        reflector.addOutput(kDstName);
    }
}
