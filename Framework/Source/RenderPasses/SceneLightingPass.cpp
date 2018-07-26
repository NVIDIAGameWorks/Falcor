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
#include "SceneLightingPass.h"

namespace Falcor
{
    static std::string kDepth = "depth";
    static std::string kColor = "color";
    static std::string kMotionVecs = "motionVecs";
    static std::string kNormals = "normals";
    static std::string kVisBuffer = "visibilityBuffer";

    SceneLightingPass::SharedPtr SceneLightingPass::create()
    {
        try
        {
            return SharedPtr(new SceneLightingPass());
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    SceneLightingPass::SharedPtr SceneLightingPass::deserialize(const RenderPassSerializer& serializer)
    {
        auto pThis = create();
        pThis->setColorFormat(ResourceFormat::RGBA32Float).setMotionVecFormat(ResourceFormat::RG16Float).setNormalMapFormat(ResourceFormat::RGBA8Unorm).setSampleCount(1).usePreGeneratedDepthBuffer(true);
        return pThis;
    }

    SceneLightingPass::SceneLightingPass() : RenderPass("SceneLightingPass")
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("RenderPasses/SceneLightingPass.slang", "", "ps");
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

    void SceneLightingPass::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInput(kVisBuffer).setFlags(RenderPassReflection::Field::Flags::Optional);
        reflector.addInputOutput(kColor).setFormat(mColorFormat).setSampleCount(mSampleCount);

        auto& depthField = mUsePreGenDepth ? reflector.addInputOutput(kDepth) : reflector.addOutput(kDepth);
        depthField.setBindFlags(Resource::BindFlags::DepthStencil);
        
        if(mNormalMapFormat != ResourceFormat::Unknown)
        {
            reflector.addOutput(kNormals).setFormat(mNormalMapFormat).setSampleCount(mSampleCount);
        }

        if (mMotionVecFormat != ResourceFormat::Unknown)
        {
            reflector.addOutput(kMotionVecs).setFormat(mMotionVecFormat).setSampleCount(mSampleCount);
        }
    }

    void SceneLightingPass::setScene(const Scene::SharedPtr& pScene)
    {
        mpSceneRenderer = nullptr;
        if (pScene) mpSceneRenderer = SceneRenderer::create(pScene);
    }

    void SceneLightingPass::initDepth(const RenderData* pRenderData)
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

    void SceneLightingPass::initFbo(RenderContext* pContext, const RenderData* pRenderData)
    {
        mpFbo->attachColorTarget(pRenderData->getTexture(kColor), 0);
        mpFbo->attachColorTarget(pRenderData->getTexture(kNormals), 1);
        mpFbo->attachColorTarget(pRenderData->getTexture(kMotionVecs), 2);

        for(uint32_t i = 1 ; i < 3 ; i++)
        {
            const auto& pRtv = mpFbo->getRenderTargetView(i).get();
            if(pRtv) pContext->clearRtv(pRtv, vec4(0));
        }

        if (mUsePreGenDepth == false) pContext->clearDsv(pRenderData->getTexture(kDepth)->getDSV().get(), 1, 0);
    }

    void SceneLightingPass::execute(RenderContext* pContext, const RenderData* pRenderData)
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

    void SceneLightingPass::renderUI(Gui* pGui, const char* uiGroup)
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

    SceneLightingPass& SceneLightingPass::setColorFormat(ResourceFormat format)
    {
        mColorFormat = format;
        mPassChangedCB();
        return *this;
    }

    SceneLightingPass& SceneLightingPass::setNormalMapFormat(ResourceFormat format)
    {
        mNormalMapFormat = format;
        mPassChangedCB();
        return *this;
    }

    SceneLightingPass& SceneLightingPass::setMotionVecFormat(ResourceFormat format)
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

    SceneLightingPass& SceneLightingPass::setSampleCount(uint32_t samples)
    {
        mSampleCount = samples;
        mPassChangedCB();
        return *this;
    }

    SceneLightingPass& SceneLightingPass::setSuperSampling(bool enable)
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

    SceneLightingPass& SceneLightingPass::usePreGeneratedDepthBuffer(bool enable)
    {
        mUsePreGenDepth = enable;
        mPassChangedCB();
        mpState->setDepthStencilState(mUsePreGenDepth ? mpDsNoDepthWrite : nullptr);

        return *this;
    }

    SceneLightingPass& SceneLightingPass::setSampler(const Sampler::SharedPtr& pSampler)
    {
        mpVars->setSampler("gSampler", pSampler);
        return *this;
    }
}