/***************************************************************************
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
***************************************************************************/
#include "TAA.h"
namespace Falcor
{
    static const char* kShaderFilename = "Effects\\TAA.ps.slang";

    TemporalAA::~TemporalAA() = default;

    TemporalAA::TemporalAA()
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mpLinearSampler = Sampler::create(samplerDesc);
        createProgram();
    }

    TemporalAA::UniquePtr TemporalAA::create()
    {
        TemporalAA* pTaa = new TemporalAA();
        return TemporalAA::UniquePtr(pTaa);
    }

    void TemporalAA::renderUI(Gui* pGui)
    {
            pGui->addFloatVar("Alpha", mControls.alpha, 0, 1.0f);
            pGui->addFloatVar("Color-Box Sigma", mControls.colorBoxSigma, 0, 15);
    }

    void TemporalAA::createProgram()
    {
        mpProgram = FullScreenPass::create(kShaderFilename);
        ProgramReflection::SharedConstPtr pReflector = mpProgram->getProgram()->getActiveVersion()->getReflector();
        mpProgVars = GraphicsVars::create(pReflector);
        mpCB = mpProgVars->getConstantBuffer("PerFrameCB");

        // Initialize the CB offsets
        mVarOffsets.alpha = mpCB->getVariableOffset("gAlpha");
        mVarOffsets.colorBoxSigma = mpCB->getVariableOffset("gColorBoxSigma");

        // Get the textures data
        mVarOffsets.colorTex = 0;// pReflector->getResourceDesc("gTexColor")->regIndex;
        mVarOffsets.prevColorTex = 0;// pReflector->getResourceDesc("gTexPrevColor")->regIndex;
        mVarOffsets.motionVecTex = 0;// pReflector->getResourceDesc("gTexMotionVec")->regIndex;
        mVarOffsets.sampler = 0;// pReflector->getResourceDesc("gSampler")->regIndex;
    }

    void TemporalAA::setVarsData(const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec)
    {
        // Make sure the dimensions match
        assert((pCurColor->getWidth() == pPrevColor->getWidth()) && (pCurColor->getWidth() == pMotionVec->getWidth()));
        assert((pCurColor->getHeight() == pPrevColor->getHeight()) && (pCurColor->getHeight() == pMotionVec->getHeight()));
        assert(pCurColor->getSampleCount() == 1 && pPrevColor->getSampleCount() == 1 && pMotionVec->getSampleCount() == 1);

        mpCB[mVarOffsets.alpha] = mControls.alpha;
        mpCB[mVarOffsets.colorBoxSigma] = mControls.colorBoxSigma;

        mpProgVars->setTexture("gTexColor", pCurColor);
        mpProgVars->setTexture("gTexPrevColor", pPrevColor);
        mpProgVars->setTexture("gTexMotionVec", pMotionVec);
        mpProgVars->setSampler("gSampler", mpLinearSampler);
    }

    void TemporalAA::execute(RenderContext* pRenderContext, const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec)
    {
        setVarsData(pCurColor, pPrevColor, pMotionVec);
        pRenderContext->pushGraphicsVars(mpProgVars);
        mpProgram->execute(pRenderContext);
        pRenderContext->popGraphicsVars();
    }
}