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
#include "ToneMapping.h"
#include "API/RenderContext.h"
#include "Graphics/FboHelper.h"

namespace Falcor
{
    static const char* kShaderFilename = "Effects/ToneMapping.ps.slang";
    const Gui::DropdownList kOperatorList = { 
    { (uint32_t)ToneMapping::Operator::Clamp, "Clamp to LDR" },
    { (uint32_t)ToneMapping::Operator::Linear, "Linear" }, 
    { (uint32_t)ToneMapping::Operator::Reinhard, "Reinhard" },
    { (uint32_t)ToneMapping::Operator::ReinhardModified, "Modified Reinhard" }, 
    { (uint32_t)ToneMapping::Operator::HejiHableAlu, "Heji's approximation" },
    { (uint32_t)ToneMapping::Operator::HableUc2, "Uncharted 2" },
    { (uint32_t)ToneMapping::Operator::Aces, "ACES" }
    };

    static const std::string kOperator = "operator";

    ToneMapping::~ToneMapping() = default;

    ToneMapping::ToneMapping(ToneMapping::Operator op) : RenderPass("ToneMapping")
    {
        createLuminancePass();
        createToneMapPass(op);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        mpPointSampler = Sampler::create(samplerDesc);
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
        mpLinearSampler = Sampler::create(samplerDesc);
    }

    ToneMapping::SharedPtr ToneMapping::create(Operator op)
    {
        ToneMapping* pTM = new ToneMapping(op);
        return ToneMapping::SharedPtr(pTM);
    }

    ToneMapping::SharedPtr ToneMapping::create(const Dictionary& dict)
    {
        Operator op = Operator::Aces;
        for (const auto& v : dict)
        {
            if (v.key() == kOperator) op = v.val();
            else logWarning("Unknown field `" + v.key() + "` in a ToneMapping dictionary");
        }
        return create(op);
    }

    Dictionary ToneMapping::getScriptingDictionary() const
    {
        Dictionary d;
        d[kOperator] = mOperator;
        return d;
    }

    void ToneMapping::createLuminanceFbo(const Texture::SharedPtr& pSrc)
    {
        bool createFbo = mpLuminanceFbo == nullptr;
        ResourceFormat srcFormat = pSrc->getFormat();
        uint32_t bytesPerChannel = getFormatBytesPerBlock(srcFormat) / getFormatChannelCount(srcFormat);
        
        // Find the required texture size and format
        ResourceFormat luminanceFormat = (bytesPerChannel == 32) ? ResourceFormat::R32Float : ResourceFormat::R16Float;
        uint32_t requiredHeight = getLowerPowerOf2(pSrc->getHeight());
        uint32_t requiredWidth = getLowerPowerOf2(pSrc->getWidth());

        if(createFbo == false)
        {
            createFbo = (requiredWidth != mpLuminanceFbo->getWidth()) ||
                (requiredHeight != mpLuminanceFbo->getHeight()) ||
                (luminanceFormat != mpLuminanceFbo->getColorTexture(0)->getFormat());
        }

        if(createFbo)
        {
            Fbo::Desc desc;
            desc.setColorTarget(0, luminanceFormat);
            mpLuminanceFbo = FboHelper::create2D(requiredWidth, requiredHeight, desc, 1, Fbo::kAttachEntireMipLevel);
        }
    }

    void ToneMapping::execute(RenderContext* pRenderContext, const Fbo::SharedPtr& pSrc, const Fbo::SharedPtr& pDst)
    {
        return execute(pRenderContext, pSrc->getColorTexture(0), pDst);
    }

    void ToneMapping::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc, const Fbo::SharedPtr& pDst)
    {
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        createLuminanceFbo(pSrc);

        //Set shared vars
        mpToneMapVars->getDefaultBlock()->setSrv(mBindLocations.colorTex, 0, pSrc->getSRV());
        mpLuminanceVars->getDefaultBlock()->setSrv(mBindLocations.colorTex, 0, pSrc->getSRV());
        mpToneMapVars->getDefaultBlock()->setSampler(mBindLocations.colorSampler, 0, mpPointSampler);
        mpLuminanceVars->getDefaultBlock()->setSampler(mBindLocations.colorSampler, 0, mpLinearSampler);

        //Calculate luminance
        pRenderContext->setGraphicsVars(mpLuminanceVars);
        pState->setFbo(mpLuminanceFbo);
        mpLuminancePass->execute(pRenderContext);
        mpLuminanceFbo->getColorTexture(0)->generateMips(pRenderContext);

        //Set Tone map vars
        if (mOperator != Operator::Clamp)
        {
            mpToneMapCBuffer->setBlob(&mConstBufferData, 0u, sizeof(mConstBufferData));
            mpToneMapVars->getDefaultBlock()->setSampler(mBindLocations.luminanceSampler, 0, mpLinearSampler);
            mpToneMapVars->getDefaultBlock()->setSrv(mBindLocations.luminanceTex, 0, mpLuminanceFbo->getColorTexture(0)->getSRV());
        }

        //Tone map
        pRenderContext->setGraphicsVars(mpToneMapVars);
        pState->setFbo(pDst);
        mpToneMapPass->execute(pRenderContext);
    }

    void ToneMapping::createToneMapPass(ToneMapping::Operator op)
    {
        mpToneMapPass = FullScreenPass::create(kShaderFilename);

        mOperator = op;
        switch(op)
        {
        case Operator::Clamp:
            mpToneMapPass->getProgram()->addDefine("_CLAMP");
            break;
        case Operator::Linear:
            mpToneMapPass->getProgram()->addDefine("_LINEAR");
            break;
        case Operator::Reinhard:
            mpToneMapPass->getProgram()->addDefine("_REINHARD");
            break;
        case Operator::ReinhardModified:
            mpToneMapPass->getProgram()->addDefine("_REINHARD_MOD");
            break;
        case Operator::HejiHableAlu:
            mpToneMapPass->getProgram()->addDefine("_HEJI_HABLE_ALU");
            break;
        case Operator::HableUc2:
            mpToneMapPass->getProgram()->addDefine("_HABLE_UC2");
            break;
        case Operator::Aces:
            mpToneMapPass->getProgram()->addDefine("_ACES");
            break;
        default:
            should_not_get_here();
        }

        const auto& pReflector = mpToneMapPass->getProgram()->getReflector();
        mpToneMapVars = GraphicsVars::create(pReflector);
        mpToneMapCBuffer = mpToneMapVars["PerImageCB"];
        const auto& pDefaultBlock = pReflector->getDefaultParameterBlock();
        mBindLocations.luminanceSampler = pDefaultBlock->getResourceBinding("gLuminanceTexSampler");
        mBindLocations.colorSampler     = pDefaultBlock->getResourceBinding("gColorSampler");
        mBindLocations.colorTex         = pDefaultBlock->getResourceBinding("gColorTex");
        mBindLocations.luminanceTex     = pDefaultBlock->getResourceBinding("gLuminanceTex");
    }

    void ToneMapping::createLuminancePass()
    {
        mpLuminancePass = FullScreenPass::create(kShaderFilename);
        mpLuminancePass->getProgram()->addDefine("_LUMINANCE");
        const auto& pReflector = mpLuminancePass->getProgram()->getReflector();
        mpLuminanceVars = GraphicsVars::create(pReflector);
    }

    void ToneMapping::renderUI(Gui* pGui, const char* uiGroup)
    {
        if((uiGroup == nullptr) || pGui->beginGroup(uiGroup))
        {
            uint32_t opIndex = static_cast<uint32_t>(mOperator);
            if (pGui->addDropdown("Operator", kOperatorList, opIndex))
            {
                mOperator = static_cast<Operator>(opIndex);
                createToneMapPass(mOperator);
            }

            pGui->addFloatVar("Exposure Key", mConstBufferData.exposureKey, 0.0001f, 200.0f);
            pGui->addFloatVar("Luminance LOD", mConstBufferData.luminanceLod, 0, 16, 0.025f);
            //Only give option to change these if the relevant operator is selected
            if (mOperator == Operator::ReinhardModified)
            {
                pGui->addFloatVar("White Luminance", mConstBufferData.whiteMaxLuminance, 0.1f, FLT_MAX, 0.2f);
            }
            else if (mOperator == Operator::HableUc2)
            {
                pGui->addFloatVar("Linear White", mConstBufferData.whiteScale, 0, 100, 0.01f);
            }

            if (uiGroup) pGui->endGroup();
        }
    }

    void ToneMapping::setOperator(Operator op)
    {
        if(op != mOperator)
        {
            createToneMapPass(op);
        }
    }

    void ToneMapping::setExposureKey(float exposureKey)
    {
        mConstBufferData.exposureKey = max(0.001f, exposureKey);
    }

    void ToneMapping::setWhiteMaxLuminance(float maxLuminance)
    {
        mConstBufferData.whiteMaxLuminance = maxLuminance;
    }

    void ToneMapping::setLuminanceLod(float lod)
    {
        mConstBufferData.luminanceLod = clamp(lod, 0.0f, 16.0f);
    }

    void ToneMapping::setWhiteScale(float whiteScale)
    {
        mConstBufferData.whiteScale = max(0.001f, whiteScale);
    }

    static const std::string kSrc = "src";
    static const std::string kDst = "dst";

    RenderPassReflection ToneMapping::reflect() const
    {
        RenderPassReflection reflector;
        reflector.addInput(kSrc);
        reflector.addOutput(kDst);
        return reflector;
    }

    void ToneMapping::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        Fbo::SharedPtr pFbo = Fbo::create();
        pFbo->attachColorTarget(pData->getTexture(kDst), 0);

        execute(pRenderContext, pData->getTexture(kSrc), pFbo);
    }
}