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
#include "MotionBlur.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/Camera/Camera.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"

namespace Falcor
{
    const uint32_t kMaxKernelSize = 15;
    static const char* kColor = "color";
    static const char* kMotionVecs = "motionVecs";
    static const char* kDstName = "dst";

    MotionBlur::SharedPtr MotionBlur::create(int32_t numSamples, float intensity)
    {
        return SharedPtr(new MotionBlur(numSamples, intensity));
    }

    MotionBlur::MotionBlur(int32_t numSamples, float intensity)
        : RenderPass("MotionBlur"), mNumSamples(numSamples), mIntensity(intensity)
    {
        BlendState::Desc desc;
        desc.setRtBlend(0, true);
        desc.setRtParams(0,
            BlendState::BlendOp::Add, BlendState::BlendOp::Add,
            BlendState::BlendFunc::One, BlendState::BlendFunc::Zero,
            BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha);

        mpAdditiveBlend = BlendState::create(desc);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);

        createShader();
    }

    void MotionBlur::createShader()
    {
        Program::DefineList programDefines;
        programDefines.add("_NUM_SAMPLES", std::to_string(mNumSamples));
        programDefines.add("_INTENSITY", std::to_string(mIntensity));

        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Effects/MotionBlur.ps.slang", programDefines);
        mSrcTexLoc   = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gSrcTex");
        mSrcVelocityLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gSrcVelocityTex");
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);
        mpVars->setSampler("gSampler", mpSampler);
    }

    void MotionBlur::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInput(kColor);
        reflector.addInput(kMotionVecs);
        reflector.addOutput(kDstName);
    }

    MotionBlur::SharedPtr MotionBlur::deserialize(const RenderPassSerializer& serializer)
    {
        return create(serializer.getValue("motionBlur.numSamples").u32, static_cast<float>(serializer.getValue("motionBlur.intensity").d64));
    }

    void MotionBlur::serialize(RenderPassSerializer& renderPassSerializer)
    {
        renderPassSerializer.addVariable("motionBlur.numSamples", mNumSamples);
        renderPassSerializer.addVariable("motionBlur.intensity", mNumSamples);
    }

    void MotionBlur::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        if (!mpTargetFbo) mpTargetFbo = Fbo::create();
        mpTargetFbo->attachColorTarget(pData->getTexture(kDstName), 0);

        execute(pRenderContext, pData->getTexture(kMotionVecs), mpTargetFbo);
    }

    void MotionBlur::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pVelocityTex, Fbo::SharedPtr pFbo)
    {
        Texture::SharedPtr pSrcColorTexture = pFbo->getColorTexture(0);

        if (mDirty)
        {
            createShader();
            mDirty = false;
        }

        if (!mpSrcTex ||
            (mpSrcTex->getWidth() != pSrcColorTexture->getWidth()) ||
            (mpSrcTex->getHeight() != pSrcColorTexture->getHeight()))
        {
            mpSrcTex = Texture::create2D(pSrcColorTexture->getWidth(), 
                pSrcColorTexture->getHeight(), pSrcColorTexture->getFormat(), 1, 1);
        }

        pRenderContext->blit(pSrcColorTexture->getSRV(), mpSrcTex->getRTV());

        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, mpSrcTex->getSRV());
        mpVars->getDefaultBlock()->setSrv(mSrcVelocityLoc, 0, pVelocityTex->getSRV());
        
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlitPass->execute(pRenderContext, nullptr, mpAdditiveBlend);
        pRenderContext->popGraphicsVars();
        pState->popFbo();
    }

    void MotionBlur::setNumSamples(int32_t numSamples)
    {
        mNumSamples = numSamples;
        mDirty = true;
    }

    void MotionBlur::setIntensity(float intensity)
    {
        mIntensity = intensity;
        mDirty = true;
    }

    void MotionBlur::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            if (pGui->addIntVar("Num Samples", mNumSamples, 0, 1000))
            {
                setNumSamples(mNumSamples);
            }

            if (pGui->addFloatVar("Intensity Scalar", mIntensity, 0.0f))
            {
                setIntensity(mIntensity);
            }

            pGui->endGroup();
        }
    }

}
