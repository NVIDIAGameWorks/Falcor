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
#include "PassFilter.h"
#include "API/RenderContext.h"
#include "Graphics/FboHelper.h"
#include "Utils/Gui.h"

namespace Falcor
{
    static std::string kShaderFilename("Effects/PassFilter.ps.slang");

    PassFilter::PassFilter(Type filterType, float threshold)
        : mFilterType(filterType)
        , mThreshold(threshold)
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);

        initProgram();
    }

    PassFilter::SharedPtr PassFilter::create(Type filterType, float threshold)
    {
        PassFilter* pBlur = new PassFilter(filterType, threshold);
        return PassFilter::SharedPtr(pBlur);
    }

    void PassFilter::updateResultFbo(const Texture* pSrc)
    {
        // Create FBO if not created already, or properties have changed since last use
        bool createFbo = mpResultFbo == nullptr;
        if(createFbo == false)
        {
            createFbo = (pSrc->getWidth() != mpResultFbo->getWidth()) ||
                (pSrc->getHeight() != mpResultFbo->getHeight()) ||
                (pSrc->getFormat() != mpResultFbo->getColorTexture(0)->getFormat());
        }

        if(createFbo)
        {
            Fbo::Desc fboDesc;
            fboDesc.setColorTarget(0, pSrc->getFormat());
            mpResultFbo = FboHelper::create2D(pSrc->getWidth(), pSrc->getHeight(), fboDesc, 1);
        }
    }

    void PassFilter::initProgram()
    {
        Program::DefineList defines;

        switch (mFilterType)
        {
        case Type::HighPass:
            defines.add("HIGH_PASS");
            break;
        case Type::LowPass:
            defines.add("LOW_PASS");
            break;
        }

        mpFilterPass = FullScreenPass::create(kShaderFilename, defines);

        ProgramReflection::SharedConstPtr pReflector = mpFilterPass->getProgram()->getReflector();
        mpVars = GraphicsVars::create(pReflector);
        mpParamCB = mpVars->getConstantBuffer("ParamCB");

        mBindLocations.sampler = pReflector->getDefaultParameterBlock()->getResourceBinding("gSampler");
        mBindLocations.srcTexture = pReflector->getDefaultParameterBlock()->getResourceBinding("gSrcTex");
    }

    Texture::SharedPtr PassFilter::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc)
    {
        updateResultFbo(pSrc.get());
        return execute(pRenderContext, pSrc, mpResultFbo);
    }

    Texture::SharedPtr PassFilter::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc, const Fbo::SharedPtr& pDst)
    {
        if (mDirty)
        {
            mpParamCB["gThreshold"] = mThreshold;
            mDirty = false;
        }

        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();

        // Horizontal pass
        ParameterBlock* pDefaultBlock = mpVars->getDefaultBlock().get();
        pDefaultBlock->setSampler(mBindLocations.sampler, 0, mpSampler);
        pDefaultBlock->setSrv(mBindLocations.srcTexture, 0, pSrc->getSRV());

        pState->pushFbo(pDst);
        pRenderContext->pushGraphicsVars(mpVars);
        mpFilterPass->execute(pRenderContext);
        pRenderContext->popGraphicsVars();
        pState->popFbo();

        return pDst->getColorTexture(0);
    }
}