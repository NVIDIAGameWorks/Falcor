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

    void FXAA::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup || pGui->beginGroup(uiGroup))
        {
            pGui->addFloatVar("Sub-Pixel Quality", mQualitySubPix, 0, 1);
            pGui->addFloatVar("Edge Threshold", mQualityEdgeThreshold, 0, 1);
            pGui->addFloatVar("Edge Threhold Min", mQualityEdgeThresholdMin, 0, 1);
            pGui->addCheckBox("Early out", mEarlyOut);
            if (uiGroup) pGui->endGroup();
        }
    }

    void FXAA::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Fbo::SharedPtr& pDstFbo)
    {
        mpGraphicsVars->setTexture("gSrc", pSrcTex);
        float2 rcpFrame = 1.0f / float2(pSrcTex->getWidth(), pSrcTex->getHeight());
        auto& pCB = mpGraphicsVars->getDefaultBlock()["PerFrameCB"];
        pCB["rcpTexDim"] = rcpFrame;
        pCB["qualitySubPix"] = mQualitySubPix;
        pCB["qualityEdgeThreshold"] = mQualityEdgeThreshold;
        pCB["qualityEdgeThresholdMin"] = mQualityEdgeThresholdMin;
        pCB["earlyOut"] = mEarlyOut;

        mpGraphicsVars->setSampler("gSampler", mpLinearSampler);

        pRenderContext->pushGraphicsVars(mpGraphicsVars);
        pRenderContext->getGraphicsState()->pushFbo(pDstFbo);

        mpPass->execute(pRenderContext);

        pRenderContext->getGraphicsState()->popFbo();
        pRenderContext->popGraphicsVars();
    }
}