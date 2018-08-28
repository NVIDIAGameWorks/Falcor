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
    static const std::string kSrc = "src";
    static const std::string kDst = "dst";

    const Gui::DropdownList kOperatorList = { 
    { (uint32_t)ToneMapping::Operator::Clamp, "Clamp to LDR" },
    { (uint32_t)ToneMapping::Operator::Linear, "Linear" }, 
    { (uint32_t)ToneMapping::Operator::Reinhard, "Reinhard" },
    { (uint32_t)ToneMapping::Operator::ReinhardModified, "Modified Reinhard" }, 
    { (uint32_t)ToneMapping::Operator::HejiHableAlu, "Heji's approximation" },
    { (uint32_t)ToneMapping::Operator::HableUc2, "Uncharted 2" },
    { (uint32_t)ToneMapping::Operator::Aces, "ACES" }
    };

    ToneMapping::~ToneMapping() = default;

    ToneMapping::ToneMapping(ToneMapping::Operator op) : RenderPass("ToneMappingPass")
    {
        createLuminancePass();
        createToneMapPass(op);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        mpPointSampler = Sampler::create(samplerDesc);
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
        mpLinearSampler = Sampler::create(samplerDesc);
        mPrevTime = std::chrono::system_clock::now();
    }

    ToneMapping::SharedPtr ToneMapping::create(Operator op)
    {
        ToneMapping* pTM = new ToneMapping(op);
        return ToneMapping::SharedPtr(pTM);
    }

    void ToneMapping::createLuminanceFbo(const Texture::SharedPtr& pSrc)
    {
        bool createFbo = (mpLuminanceFbo == nullptr);
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

    ToneMapping::SharedPtr ToneMapping::deserialize(const RenderPassSerializer& serializer) 
    {
        Scene::UserVariable opVariable = serializer.getValue("toneMapping.operator");
        if (opVariable.type == Scene::UserVariable::Type::Unknown)
        {
            return create(Operator::Aces);
        }

        ToneMapping::SharedPtr pPass = create(static_cast<Operator>(opVariable.u32));
        pPass->mConstBufferData.exposureKey = static_cast<float>(serializer.getValue("toneMapping.exposureKey").d64);
        pPass->mConstBufferData.whiteMaxLuminance = static_cast<float>(serializer.getValue("toneMapping.whiteMaxLuminance").d64);
        pPass->mConstBufferData.whiteMaxLuminance = static_cast<float>(serializer.getValue("toneMapping.luminanceLOD").d64);
        pPass->mConstBufferData.whiteMaxLuminance = static_cast<float>(serializer.getValue("toneMapping.whiteScale").d64);
        pPass->mEnableEyeAdaptation = serializer.getValue("toneMapping.mEnableEyeAdaptation").b;
        pPass->mEyeAdaptationSettings.camEV100 = static_cast<float>(serializer.getValue("toneMapping.camEV100").d64);
        pPass->mEyeAdaptationSettings.camIso = static_cast<float>(serializer.getValue("toneMapping.camIso").d64);
        pPass->mEyeAdaptationSettings.speedDown = static_cast<float>(serializer.getValue("toneMapping.speedDown").d64);
        pPass->mEyeAdaptationSettings.speedUp = static_cast<float>(serializer.getValue("toneMapping.speedUp").d64);

        return pPass;
    }

    void ToneMapping::serialize(RenderPassSerializer& renderPassSerializer)
    {
        renderPassSerializer.addVariable("toneMapping.operator", static_cast<uint32_t>(mOperator));
        renderPassSerializer.addVariable("toneMapping.exposureKey", mConstBufferData.exposureKey);
        renderPassSerializer.addVariable("toneMapping.whiteMaxLuminance", mConstBufferData.whiteMaxLuminance);
        renderPassSerializer.addVariable("toneMapping.luminanceLOD", mConstBufferData.luminanceLod);
        renderPassSerializer.addVariable("toneMapping.whiteScale", mConstBufferData.whiteScale);
        renderPassSerializer.addVariable("toneMapping.mEnableEyeAdaptation", mEnableEyeAdaptation);
        renderPassSerializer.addVariable("toneMapping.camEV100", mEyeAdaptationSettings.camEV100);
        renderPassSerializer.addVariable("toneMapping.camIso", mEyeAdaptationSettings.camIso);
        renderPassSerializer.addVariable("toneMapping.speedDown", mEyeAdaptationSettings.speedDown);
        renderPassSerializer.addVariable("toneMapping.speedUp", mEyeAdaptationSettings.speedUp);
    }

    void ToneMapping::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        Fbo::SharedPtr pFbo = Fbo::create();
        pFbo->attachColorTarget(pData->getTexture(kDst), 0);

        execute(pRenderContext, pData->getTexture(kSrc), pFbo, mpCamera);
    }

    void ToneMapping::execute(RenderContext* pRenderContext, const Fbo::SharedPtr& pSrc, const Fbo::SharedPtr& pDst)
    {
        return execute(pRenderContext, pSrc->getColorTexture(0), pDst, mpCamera);
    }

    void ToneMapping::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc, const Fbo::SharedPtr& pDst, const Camera::SharedPtr& pCamera)
    {
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        createLuminanceFbo(pSrc);

        auto time = std::chrono::system_clock::now();
        float deltaTime = std::chrono::duration<float>(time - mPrevTime).count();
        mPrevTime = time;

        if (pCamera) mpCamera = pCamera;

        if (mEnableEyeAdaptation)
        {
            mpToneMapVars["PerImageCB"]["deltaTime"] = deltaTime;
            mpToneMapVars["PerImageCB"]["camEV100"] = calculateEV100();
            mpToneMapPass->getProgram()->addDefine("_ENABLE_EYE_ADAPTATION");
        }
        else
        {
            mpToneMapPass->getProgram()->removeDefine("_ENABLE_EYE_ADAPTATION");
        }
        
        //Set shared vars
        mpToneMapVars->getDefaultBlock()->setTexture("gColorTex", pSrc);
        mpLuminanceVars->getDefaultBlock()->setTexture("gColorTex", pSrc);
        mpToneMapVars->getDefaultBlock()->setSampler("gColorSampler", mpPointSampler);
        mpToneMapVars->getDefaultBlock()->setSampler("gColorSampler", mpLinearSampler);

        //Calculate luminance
        pRenderContext->setGraphicsVars(mpLuminanceVars);
        pState->setFbo(mpLuminanceFbo);
        mpLuminancePass->execute(pRenderContext);
        mpLuminanceFbo->getColorTexture(0)->generateMips(pRenderContext);

        //Set Tone map vars
        if (mOperator != Operator::Clamp)
        {
            mpToneMapCBuffer->setBlob(&mConstBufferData, 0u, sizeof(mConstBufferData));
            mpToneMapVars->getDefaultBlock()->setSampler("gLuminanceTexSampler", mpLinearSampler);
            mpToneMapVars->getDefaultBlock()->setTexture("gLuminanceTex", mpLuminanceFbo->getColorTexture(0));
        }

        //Tone map
        pRenderContext->setGraphicsVars(mpToneMapVars);
        pState->setFbo(pDst);
        mpToneMapPass->execute(pRenderContext);
    }

    float ToneMapping::calculateEV100()
    {
        float shutterSpeed = 1.0f;
        float apertureRadius = 0.0f;
        
        if (!mpCamera)
        {
            logWarning("No camera set in render pass. Unable to execute eye adaption.");
        }
        else
        {
            shutterSpeed = 1.0f / mpCamera->getFocalLength();
            apertureRadius = mpCamera->getApertureRadius();
        }

        if (apertureRadius <= 0.001f) apertureRadius = 77.0f; // good default value if camera set at 0
        float ev100 = std::log2(std::max(apertureRadius, 0.001f) * shutterSpeed * 100.0f / mEyeAdaptationSettings.camIso);
        mEyeAdaptationSettings.camEV100 = ev100;
        return ev100;
    }
    
    void ToneMapping::createToneMapPass(ToneMapping::Operator op)
    {
        mpToneMapPass = FullScreenPass::create(kShaderFilename);

        mOperator = op;
        switch (op)
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

        mpToneMapCBuffer["speedUp"]   = 1.0f / mEyeAdaptationSettings.speedUp;
        mpToneMapCBuffer["speedDown"] = 1.0f / mEyeAdaptationSettings.speedDown;
        mpToneMapCBuffer["camEV100"] = mEyeAdaptationSettings.camIso;

        mpToneMapVars->setRawBuffer("prevFrameInfo", Buffer::create(sizeof(float) * 4, Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None));
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

            pGui->addCheckBox("Enable Eye Adaptation", mEnableEyeAdaptation);

            if (mEnableEyeAdaptation)
            {
                if (pGui->addFloatVar("Speed Up", mEyeAdaptationSettings.speedUp, 0.0f))
                {
                    mpToneMapCBuffer["speedUp"] = 1.0f / mEyeAdaptationSettings.speedUp;
                }

                if (pGui->addFloatVar("Speed Down", mEyeAdaptationSettings.speedDown, 0.0f))
                {
                    mpToneMapCBuffer["speedDown"] = 1.0f / mEyeAdaptationSettings.speedDown;
                }
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

    void ToneMapping::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInput(kSrc);
        reflector.addOutput(kDst);
    }
}