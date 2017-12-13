/***************************************************************************
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
***************************************************************************/
#include "TAA.h"
namespace Falcor
{
    static const char* kShaderFilename = "Effects/TAA.ps.slang";

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
        mVarLocations.alpha = mpCB->getVariableOffset("gAlpha");
        mVarLocations.colorBoxSigma = mpCB->getVariableOffset("gColorBoxSigma");

        // Get the textures data
        const auto& pDefaultBlock = pReflector->getDefaultParameterBlock();
        mVarLocations.colorTex = pDefaultBlock->getResourceBinding("gTexColor");
        mVarLocations.prevColorTex = pDefaultBlock->getResourceBinding("gTexPrevColor");
        mVarLocations.motionVecTex = pDefaultBlock->getResourceBinding("gTexMotionVec");
        mVarLocations.sampler = pDefaultBlock->getResourceBinding("gSampler");
    }

    void TemporalAA::setVarsData(const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec)
    {
        // Make sure the dimensions match
        assert((pCurColor->getWidth() == pPrevColor->getWidth()) && (pCurColor->getWidth() == pMotionVec->getWidth()));
        assert((pCurColor->getHeight() == pPrevColor->getHeight()) && (pCurColor->getHeight() == pMotionVec->getHeight()));
        assert(pCurColor->getSampleCount() == 1 && pPrevColor->getSampleCount() == 1 && pMotionVec->getSampleCount() == 1);

        mpCB[mVarLocations.alpha] = mControls.alpha;
        mpCB[mVarLocations.colorBoxSigma] = mControls.colorBoxSigma;

        ParameterBlock* pDefaultBlock = mpProgVars->getDefaultBlock().get();
        pDefaultBlock->setSrv(mVarLocations.colorTex, 0, pCurColor->getSRV());
        pDefaultBlock->setSrv(mVarLocations.prevColorTex, 0, pPrevColor->getSRV());
        pDefaultBlock->setSrv(mVarLocations.motionVecTex, 0, pMotionVec->getSRV());
        pDefaultBlock->setSampler(mVarLocations.sampler, 0, mpLinearSampler);
    }

    void TemporalAA::execute(RenderContext* pRenderContext, const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec)
    {
        setVarsData(pCurColor, pPrevColor, pMotionVec);
        pRenderContext->pushGraphicsVars(mpProgVars);
        mpProgram->execute(pRenderContext);
        pRenderContext->popGraphicsVars();
    }
}