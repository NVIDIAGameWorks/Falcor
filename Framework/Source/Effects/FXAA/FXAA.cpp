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
        mpPointSampler = Sampler::create({});
        mpGraphicsVars->setSampler("gSampler", mpPointSampler);
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
        pRenderContext->pushGraphicsVars(mpGraphicsVars);
        pRenderContext->getGraphicsState()->pushFbo(pDstFbo);

        mpPass->execute(pRenderContext);

        pRenderContext->getGraphicsState()->popFbo();
        pRenderContext->popGraphicsVars();
    }
}