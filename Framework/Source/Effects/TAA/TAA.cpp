/***************************************************************************
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
***************************************************************************/
#include "TAA.h"
#include "Utils/Gui.h"
#include "API/RenderContext.h"

namespace Falcor
{
    static const char* kShaderFilename = "Effects/TAA.ps.slang";

    TemporalAA::~TemporalAA() = default;

    TemporalAA::TemporalAA() : RenderPass("TemporalAA")
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mpLinearSampler = Sampler::create(samplerDesc);
        createProgram();
    }

    TemporalAA::SharedPtr TemporalAA::create(const Dictionary& dict)
    {
        TemporalAA* pTaa = new TemporalAA();
        return TemporalAA::SharedPtr(pTaa);
    }

    void TemporalAA::renderUI(Gui* pGui, const char* uiGroup)
    {
        if(!uiGroup || pGui->beginGroup(uiGroup))
        {
            pGui->addFloatVar("Alpha", mControls.alpha, 0, 1.0f);
            pGui->addFloatVar("Color-Box Sigma", mControls.colorBoxSigma, 0, 15);

            if (uiGroup) pGui->endGroup();
        }
    }

    void TemporalAA::createProgram()
    {
        mpProgram = FullScreenPass::create(kShaderFilename);
        ProgramReflection::SharedConstPtr pReflector = mpProgram->getProgram()->getReflector();
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

    static const std::string& kMotionVec = "motionVecs";
    static const std::string& kColorIn = "colorIn";
    static const std::string& kColorOut = "colorOut";

    RenderPassReflection TemporalAA::reflect() const
    {
        RenderPassReflection reflection;
        reflection.addInput(kMotionVec);
        reflection.addInput(kColorIn);
        reflection.addOutput(kColorOut);
        return reflection;
    }

    void TemporalAA::allocatePrevColor(const Texture* pColorOut)
    {
        bool allocate = mpPrevColor == nullptr;
        allocate = allocate || (mpPrevColor->getWidth() != pColorOut->getWidth());
        allocate = allocate || (mpPrevColor->getHeight() != pColorOut->getHeight());
        allocate = allocate || (mpPrevColor->getDepth() != pColorOut->getDepth());
        allocate = allocate || (mpPrevColor->getFormat() != pColorOut->getFormat());
        assert(pColorOut->getSampleCount() == 1);

        if (allocate)
        {
            mpPrevColor = Texture::create2D(pColorOut->getWidth(), pColorOut->getHeight(), pColorOut->getFormat(), 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
        }
    }

    void TemporalAA::execute(RenderContext* pContext, const RenderData* pData)
    {
        const auto& pColorIn = pData->getTexture(kColorIn);
        const auto& pColorOut = pData->getTexture(kColorOut);
        const auto& pMotionVec = pData->getTexture(kMotionVec);
        allocatePrevColor(pColorOut.get());

        Fbo::SharedPtr pFbo = Fbo::create();
        pFbo->attachColorTarget(pColorOut, 0);
        pContext->getGraphicsState()->pushFbo(pFbo);
        execute(pContext, pColorIn, mpPrevColor, pMotionVec);
        pContext->getGraphicsState()->popFbo();

        pContext->blit(pColorOut->getSRV(), mpPrevColor->getRTV());
    }
}