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
#include "BlendPass.h"
#include "Graphics/Program/ProgramVars.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"

namespace Falcor
{
    BlendPass::UniquePtr BlendPass::create()
    {
        return BlendPass::UniquePtr(new BlendPass());
    }

    BlendPass::BlendPass()
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

        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Framework/Shaders/Blit.ps.slang");
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mSrcTexLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("gTex");
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);
        mpVars->setSampler("gSampler", mpSampler);
    }

    void BlendPass::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, Fbo::SharedPtr pFbo)
    {
        assert(pFbo->getWidth() == pSrcTex->getWidth() && pSrcTex->getHeight() == pFbo->getHeight());

        mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, pSrcTex->getSRV());

        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pRenderContext->pushGraphicsVars(mpVars);
        mpBlitPass->execute(pRenderContext, nullptr, mpAdditiveBlend);
        pRenderContext->popGraphicsVars();
        pState->popFbo();
    }

    void BlendPass::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
        }
    }
}
