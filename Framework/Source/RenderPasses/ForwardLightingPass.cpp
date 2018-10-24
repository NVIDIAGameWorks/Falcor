/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "ForwardLightingPass.h"

namespace Falcor
{
    static std::string kDepth = "depth";
    static std::string kColor = "color";
    static std::string kMotionVecs = "motionVecs";
    static std::string kNormals = "normals";
    static std::string kVisBuffer = "visibilityBuffer";

    static std::string kSampleCount = "sampleCount";
    static std::string kSuperSampling = "enableSuperSampling";

    ForwardLightingPass::SharedPtr ForwardLightingPass::create(const Dictionary& dict)
    {
        auto pThis = SharedPtr(new ForwardLightingPass());
        pThis->setColorFormat(ResourceFormat::RGBA32Float).setMotionVecFormat(ResourceFormat::RG16Float).setNormalMapFormat(ResourceFormat::RGBA8Unorm).setSampleCount(1).usePreGeneratedDepthBuffer(true);

        for (const auto& v : dict)
        {
            if (v.key() == kSampleCount) pThis->setSampleCount(v.val());
            else if (v.key() == kSuperSampling) pThis->setSuperSampling(v.val());
            logWarning("Unknown field `" + v.key() + "` in a ForwardLightingPass dictionary");
        }

        return pThis;
    }

    Dictionary ForwardLightingPass::getScriptingDictionary() const 
    {
        Dictionary d;
        d[kSampleCount] = mSampleCount;
        d[kSuperSampling] = mEnableSuperSampling;
        return d;
    }

    ForwardLightingPass::ForwardLightingPass() : RenderPass("ForwardLightingPass")
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("RenderPasses/ForwardLightingPass.slang", "", "ps");
        mpState = GraphicsState::create();
        mpState->setProgram(pProgram);
        mpVars = GraphicsVars::create(pProgram->getReflector());
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        setSampler(Sampler::create(samplerDesc));

        mpFbo = Fbo::create();
        
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthTest(true).setDepthWriteMask(false).setStencilTest(false).setDepthFunc(DepthStencilState::Func::LessEqual);
        mpDsNoDepthWrite = DepthStencilState::create(dsDesc);        
    }

    RenderPassReflection ForwardLightingPass::reflect() const
    {
        RenderPassReflection reflector;

        reflector.addInput(kVisBuffer).setFlags(RenderPassReflection::Field::Flags::Optional);
        reflector.addInputOutput(kColor).setFormat(mColorFormat).setSampleCount(mSampleCount);

        auto& depthField = mUsePreGenDepth ? reflector.addInputOutput(kDepth) : reflector.addOutput(kDepth);
        depthField.setBindFlags(Resource::BindFlags::DepthStencil).setSampleCount(mSampleCount);
        
        if(mNormalMapFormat != ResourceFormat::Unknown)
        {
            reflector.addOutput(kNormals).setFormat(mNormalMapFormat).setSampleCount(mSampleCount);
        }

        if (mMotionVecFormat != ResourceFormat::Unknown)
        {
            reflector.addOutput(kMotionVecs).setFormat(mMotionVecFormat).setSampleCount(mSampleCount);
        }

        return reflector;
    }

    void ForwardLightingPass::setScene(const Scene::SharedPtr& pScene)
    {
        mpSceneRenderer = nullptr;
        if (pScene) mpSceneRenderer = SceneRenderer::create(pScene);
    }

    void ForwardLightingPass::initDepth(const RenderData* pRenderData)
    {
        const auto& pTexture = pRenderData->getTexture(kDepth);

        if (pTexture)
        {
            mpState->setDepthStencilState(mpDsNoDepthWrite);
            mpFbo->attachDepthStencilTarget(pTexture);
        }
        else
        {
            mpState->setDepthStencilState(nullptr);
            if(mpFbo->getDepthStencilTexture() == nullptr)
            {
                auto pDepth = Texture::create2D(mpFbo->getWidth(), mpFbo->getHeight(), ResourceFormat::D32Float, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
                mpFbo->attachDepthStencilTarget(pDepth);
            }
        }
    }

    void ForwardLightingPass::initFbo(RenderContext* pContext, const RenderData* pRenderData)
    {
        mpFbo->attachColorTarget(pRenderData->getTexture(kColor), 0);
        mpFbo->attachColorTarget(pRenderData->getTexture(kNormals), 1);
        mpFbo->attachColorTarget(pRenderData->getTexture(kMotionVecs), 2);

        for(uint32_t i = 1 ; i < 3 ; i++)
        {
            const auto& pRtv = mpFbo->getRenderTargetView(i).get();
            if(pRtv->getResource() != nullptr) pContext->clearRtv(pRtv, vec4(0));
        }

        // TODO Matt (not really matt, just need to fix that since if depth is not bound the pass crashes
        if (mUsePreGenDepth == false) pContext->clearDsv(pRenderData->getTexture(kDepth)->getDSV().get(), 1, 0);
    }

    void ForwardLightingPass::execute(RenderContext* pContext, const RenderData* pRenderData)
    {
        initDepth(pRenderData);
        initFbo(pContext, pRenderData);

        if (mpSceneRenderer)
        {
            mpVars["PerFrameCB"]["gRenderTargetDim"] = vec2(mpFbo->getWidth(), mpFbo->getHeight());
            mpVars->setTexture(kVisBuffer, pRenderData->getTexture(kVisBuffer));

            mpState->setFbo(mpFbo);
            pContext->pushGraphicsState(mpState);
            pContext->pushGraphicsVars(mpVars);
            mpSceneRenderer->renderScene(pContext);
            pContext->popGraphicsState();
            pContext->popGraphicsVars();
        }
    }

    void ForwardLightingPass::renderUI(Gui* pGui, const char* uiGroup)
    {
        static const Gui::DropdownList kSampleCountList =
        {
            { 1, "1" },
            { 2, "2" },
            { 4, "4" },
            { 8, "8" },
        };

        if(!uiGroup || pGui->beginGroup(uiGroup))
        {
            if (pGui->addDropdown("Sample Count", kSampleCountList, mSampleCount))              setSampleCount(mSampleCount);
            if (mSampleCount > 1 && pGui->addCheckBox("Super Sampling", mEnableSuperSampling))  setSuperSampling(mEnableSuperSampling);

            if (uiGroup) pGui->endGroup();
        }
    }

    ForwardLightingPass& ForwardLightingPass::setColorFormat(ResourceFormat format)
    {
        mColorFormat = format;
        mPassChangedCB();
        return *this;
    }

    ForwardLightingPass& ForwardLightingPass::setNormalMapFormat(ResourceFormat format)
    {
        mNormalMapFormat = format;
        mPassChangedCB();
        return *this;
    }

    ForwardLightingPass& ForwardLightingPass::setMotionVecFormat(ResourceFormat format)
    {
        mMotionVecFormat = format;
        if (mMotionVecFormat != ResourceFormat::Unknown)
        {
            mpState->getProgram()->addDefine("_OUTPUT_MOTION_VECTORS");
        }
        else
        {
            mpState->getProgram()->removeDefine("_OUTPUT_MOTION_VECTORS");
        }
        mPassChangedCB();
        return *this;
    }

    ForwardLightingPass& ForwardLightingPass::setSampleCount(uint32_t samples)
    {
        mSampleCount = samples;
        mPassChangedCB();
        return *this;
    }

    ForwardLightingPass& ForwardLightingPass::setSuperSampling(bool enable)
    {
        mEnableSuperSampling = enable;
        if (mEnableSuperSampling)
        {
            mpState->getProgram()->addDefine("INTERPOLATION_MODE", "sample");
        }
        else
        {
            mpState->getProgram()->removeDefine("INTERPOLATION_MODE");
        }

        return *this;
    }

    ForwardLightingPass& ForwardLightingPass::usePreGeneratedDepthBuffer(bool enable)
    {
        mUsePreGenDepth = enable;
        mPassChangedCB();
        mpState->setDepthStencilState(mUsePreGenDepth ? mpDsNoDepthWrite : nullptr);

        return *this;
    }

    ForwardLightingPass& ForwardLightingPass::setSampler(const Sampler::SharedPtr& pSampler)
    {
        mpVars->setSampler("gSampler", pSampler);
        return *this;
    }
}