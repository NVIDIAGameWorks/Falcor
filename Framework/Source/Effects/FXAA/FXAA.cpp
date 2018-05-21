/***************************************************************************
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
***************************************************************************/
#include "FXAA.h"

namespace Falcor
{
    static const char* kShaderFilename = "Effects/FXAA.slang";

    FXAA::~FXAA() = default;

    FXAA::FXAA()
    {
        mpPass = FullScreenPass::create(kShaderFilename);
        mpGraphicsVars = GraphicsVars::create(mpPass->getProgram()->getReflector());
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
        mpLinearSampler = Sampler::create(samplerDesc);
        mpGraphicsVars->setSampler("gSampler", mpLinearSampler);
    }

    FXAA::UniquePtr FXAA::create()
    {
        try
        {
            FXAA* pTaa = new FXAA();
            return FXAA::UniquePtr(pTaa);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    void FXAA::renderUI(Gui* pGui)
    {
    }

    void FXAA::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Fbo::SharedPtr& pDstFbo)
    {
        mpGraphicsVars->setTexture("gSrc", pSrcTex);
        float2 rcpFrame = 1.0f / float2(pSrcTex->getWidth(), pSrcTex->getHeight());
        mpGraphicsVars->getDefaultBlock()["PerFrameCB"]["rcpTexDim"] = rcpFrame;
        mpGraphicsVars->getDefaultBlock()["PerFrameCB"]["qualitySubPix"] = 0.75f;
        mpGraphicsVars->getDefaultBlock()["PerFrameCB"]["qualityEdgeThreshold"] = 0.166f;
        mpGraphicsVars->getDefaultBlock()["PerFrameCB"]["qualityEdgeThresholdMin"] = 0.0833f;
        mpGraphicsVars->setSampler("gSampler", mpLinearSampler);

        pRenderContext->pushGraphicsVars(mpGraphicsVars);
        pRenderContext->getGraphicsState()->pushFbo(pDstFbo);

        mpPass->execute(pRenderContext);

        pRenderContext->getGraphicsState()->popFbo();
        pRenderContext->popGraphicsVars();
    }
}