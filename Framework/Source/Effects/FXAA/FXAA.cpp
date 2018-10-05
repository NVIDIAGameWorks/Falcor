/***************************************************************************
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
***************************************************************************/
#include "FXAA.h"
#include "Utils/Gui.h"
#include "API/RenderContext.h"

namespace Falcor
{
    static const char* kShaderFilename = "Effects/FXAA.slang";

    FXAA::~FXAA() = default;

    FXAA::FXAA() : RenderPass("FXAA")
    {
        mpPass = FullScreenPass::create(kShaderFilename);
        mpGraphicsVars = GraphicsVars::create(mpPass->getProgram()->getReflector());
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
        mpLinearSampler = Sampler::create(samplerDesc);
        mpGraphicsVars->setSampler("gSampler", mpLinearSampler);
    }

    FXAA::SharedPtr FXAA::create(const Dictionary& dict)
    {
        try
        {
            FXAA* pFxaa = new FXAA();
            return FXAA::SharedPtr(pFxaa);
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
            pGui->addFloatVar("Edge Threshold Min", mQualityEdgeThresholdMin, 0, 1);
            pGui->addCheckBox("Early out", mEarlyOut);
            if (uiGroup) pGui->endGroup();
        }
    }

    void FXAA::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Fbo::SharedPtr& pDstFbo)
    {
        mpGraphicsVars->setTexture("gSrc", pSrcTex);
        vec2 rcpFrame = 1.0f / vec2(pSrcTex->getWidth(), pSrcTex->getHeight());
        auto pCB = mpGraphicsVars->getDefaultBlock()["PerFrameCB"];
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

    static const std::string kSrc = "src";
    static const std::string kDst = "dst";

    RenderPassReflection FXAA::reflect() const
    {
        RenderPassReflection reflector;
        reflector.addInput(kSrc);
        reflector.addOutput(kDst);
        return reflector;
    }

    void FXAA::execute(RenderContext* pContext, const RenderData* pData)
    {
        auto pSrc = pData->getTexture(kSrc);
        auto pDst = pData->getTexture(kDst);

        Fbo::SharedPtr pFbo = Fbo::create();
        pFbo->attachColorTarget(pDst, 0);
        execute(pContext, pSrc, pFbo);
    }
}