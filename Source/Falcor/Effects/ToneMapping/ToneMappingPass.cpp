/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "ToneMappingPass.h"
#include "Utils/Color/ColorUtils.h"

namespace Falcor
{
    const char* ToneMappingPass::kDesc = "Tone-map a color-buffer. The resulting buffer is always in the [0, 1] range. The pass supports auto-exposure and eye-adaptation";

    namespace
    {
        const Gui::DropdownList kOperatorList =
        {
            { (uint32_t)ToneMappingPass::Operator::Clamp, "Clamp to LDR" },
            { (uint32_t)ToneMappingPass::Operator::Linear, "Linear" },
            { (uint32_t)ToneMappingPass::Operator::Reinhard, "Reinhard" },
            { (uint32_t)ToneMappingPass::Operator::ReinhardModified, "Modified Reinhard" },
            { (uint32_t)ToneMappingPass::Operator::HejiHableAlu, "Heji's approximation" },
            { (uint32_t)ToneMappingPass::Operator::HableUc2, "Uncharted 2" },
            { (uint32_t)ToneMappingPass::Operator::Aces, "ACES" },
            { (uint32_t)ToneMappingPass::Operator::Photo, "Photo" }
        };

        const std::string kSrc = "src";
        const std::string kDst = "dst";

        const std::string kOperator = "operator";
        const std::string kExposureKey = "exposureKey";
        const std::string kWhiteMaxLuminance = "whiteMaxLuminance";
        const std::string kLuminanceLod = "luminanceLod";
        const std::string kWhiteScale = "whiteScale";
        const std::string kExposureValue = "exposureValue";
        const std::string kFilmSpeed = "filmSpeed";
        const std::string kWhitePoint = "whitePoint";
        const std::string kApplyAcesCurve = "applyAcesCurve";

        const std::string kShaderFilename = "Effects/ToneMapping.ps.slang";
    }
    
    ToneMappingPass::ToneMappingPass(ToneMappingPass::Operator op)
    {
        createLuminancePass();
        createToneMapPass(op);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        mpPointSampler = Sampler::create(samplerDesc);
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
        mpLinearSampler = Sampler::create(samplerDesc);
    }

    void ToneMappingPass::createLuminancePass()
    {
        mpLuminancePass = FullScreenPass::create(kShaderFilename);
        mpLuminancePass->addDefine("_LUMINANCE");
    }

    void ToneMappingPass::createToneMapPass(ToneMappingPass::Operator op)
    {
        mpToneMapPass = FullScreenPass::create(kShaderFilename);

        mOperator = op;
        switch (op)
        {
        case Operator::Clamp:
            mpToneMapPass->addDefine("_CLAMP");
            break;
        case Operator::Linear:
            mpToneMapPass->addDefine("_LINEAR");
            break;
        case Operator::Reinhard:
            mpToneMapPass->addDefine("_REINHARD");
            break;
        case Operator::ReinhardModified:
            mpToneMapPass->addDefine("_REINHARD_MOD");
            break;
        case Operator::HejiHableAlu:
            mpToneMapPass->addDefine("_HEJI_HABLE_ALU");
            break;
        case Operator::HableUc2:
            mpToneMapPass->addDefine("_HABLE_UC2");
            break;
        case Operator::Aces:
            mpToneMapPass->addDefine("_ACES");
            break;
        case Operator::Photo:
            mpToneMapPass->getProgram()->addDefine("_PHOTO");
            break;
        default:
            should_not_get_here();
        }
    }

    ToneMappingPass::SharedPtr ToneMappingPass::create(RenderContext* pRenderContext, const Dictionary& dict)
    {
        ToneMappingPass* pTM = new ToneMappingPass(Operator::Aces);

        try
        {
            for (const auto& v : dict)
            {
                if (v.key() == kOperator) pTM->setOperator(v.val());
                else if (v.key() == kExposureKey) pTM->mToneMappingData.exposureKey = v.val();
                else if (v.key() == kWhiteMaxLuminance) pTM->mToneMappingData.whiteMaxLuminance = v.val();
                else if (v.key() == kLuminanceLod) pTM->mToneMappingData.luminanceLod = v.val();
                else if (v.key() == kWhiteScale) pTM->mToneMappingData.whiteScale = v.val();
                else if (v.key() == kExposureValue) pTM->mExposureValue = v.val();
                else if (v.key() == kFilmSpeed) pTM->mFilmSpeed = v.val();
                else if (v.key() == kWhitePoint) pTM->mWhitePoint = v.val();
                else if (v.key() == kApplyAcesCurve) pTM->mToneMappingData.applyAcesCurve = v.val();
                else logWarning("Unknown field `" + v.key() + "` in a ToneMapping dictionary");
            }
        }
        catch (std::exception& e)
        {
            logWarning(std::string("Unable to convert dictionary to expected type: ") + e.what());
        }

        // Prepare constant buffer.
        pTM->calculateColorTransform();
        pTM->updateConstants();

        return ToneMappingPass::SharedPtr(pTM);
    }

    Dictionary ToneMappingPass::getScriptingDictionary()
    {
        Dictionary d;
        d[kOperator] = mOperator;
        d[kExposureKey] = mToneMappingData.exposureKey;
        d[kWhiteMaxLuminance] = mToneMappingData.whiteMaxLuminance;
        d[kLuminanceLod] = mToneMappingData.luminanceLod;
        d[kWhiteScale] = mToneMappingData.whiteScale;
        d[kExposureValue] = mExposureValue;
        d[kFilmSpeed] = mFilmSpeed;
        d[kWhitePoint] = mWhitePoint;
        d[kApplyAcesCurve] = mToneMappingData.applyAcesCurve;
        return d;
    }

    RenderPassReflection ToneMappingPass::reflect(const CompileData& compileData)
    {
        RenderPassReflection reflector;
        reflector.addInput(kSrc, "Source texture");
        reflector.addOutput(kDst, "Tone-mapped output texture");
        return reflector;
    }

    void ToneMappingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
    {
        auto pSrc = renderData[kSrc]->asTexture();
        auto pDst = renderData[kDst]->asTexture();
        Fbo::SharedPtr pFbo = Fbo::create();
        pFbo->attachColorTarget(pDst, 0);

        createLuminanceFbo(pSrc);

        //Set shared vars
        mpToneMapPass["gColorTex"] = pSrc;
        mpLuminancePass["gColorTex"] = pSrc;
        mpToneMapPass["gColorSampler"] = mpPointSampler;
        mpLuminancePass["gColorSampler"] = mpLinearSampler;

        // Calculate luminance
        if (mOperator != Operator::Photo)
        {
            mpLuminancePass->execute(pRenderContext, mpLuminanceFbo);
            mpLuminanceFbo->getColorTexture(0)->generateMips(pRenderContext);
        }

        //Set Tone map vars
        if (mOperator != Operator::Clamp)
        {
            mpToneMapPass["PerImageCB"].setBlob(&mToneMappingData, 0, sizeof(mToneMappingData));
            mpToneMapPass["gLuminanceTexSampler"] = mpLinearSampler;
            mpToneMapPass["gLuminanceTex"] = mpLuminanceFbo->getColorTexture(0);
        }

        // Tone map
        mpToneMapPass->execute(pRenderContext, pFbo);
    }

    void ToneMappingPass::createLuminanceFbo(const Texture::SharedPtr& pSrc)
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
            mpLuminanceFbo = Fbo::create2D(requiredWidth, requiredHeight, desc, 1, Fbo::kAttachEntireMipLevel);
        }
    }

    void ToneMappingPass::renderUI(Gui::Widgets& widget)
    {
        uint32_t opIndex = static_cast<uint32_t>(mOperator);
        if (widget.dropdown("Operator", kOperatorList, opIndex))
        {
            mOperator = static_cast<Operator>(opIndex);
            createToneMapPass(mOperator);
        }

        if (mOperator != Operator::Photo)
    {
            widget.var("Exposure Key", mToneMappingData.exposureKey, 0.0001f, 200.0f, 0.0001f);
            widget.var("Luminance LOD", mToneMappingData.luminanceLod, 0.f, 16.f, 0.025f);
            //Only give option to change these if the relevant operator is selected
            if (mOperator == Operator::ReinhardModified)
            {
                widget.var("White Luminance", mToneMappingData.whiteMaxLuminance, 0.1f, FLT_MAX, 0.2f);
            }
            else if (mOperator == Operator::HableUc2)
            {
                widget.var("Linear White", mToneMappingData.whiteScale, 0.f, 100.f, 0.01f);
            }
        }
        else
        {
            bool recomputeConstants = false;

            recomputeConstants |= widget.var("Exposure value (EV)", mExposureValue, -24.f, 24.f, 0.1f, false, "%.1f");
            recomputeConstants |= widget.var("Film speed (ISO)", mFilmSpeed, 1.f, 6400.f, 1.f, false, "%.1f");

            // Note: Color temperatures < ~1905K are out-of-gamut in Rec.709.
            if (widget.var("White point (K)", mWhitePoint, 1905.f, 25000.f, 5.f, false, "%.0f"))
            {
                calculateColorTransform();
                recomputeConstants = true;
            }

            // Display color widget for the currently chosen white point.
            // We normalize the color so that max(RGB) = 1 for display purposes.
            glm::float3 w = mSourceWhite;
            w = w / std::max(std::max(w.r, w.g), w.b);
            widget.rgbColor("", w);

            widget.checkbox("Apply ACES curve", (bool&)mToneMappingData.applyAcesCurve);

            if (recomputeConstants) updateConstants();
        }
    }

    void ToneMappingPass::calculateColorTransform()
    {
        // Calculate color transform for the current white point.
        glm::float3x3 whiteBalance = calculateWhiteBalanceTransformRGB_Rec709(mWhitePoint);
        mColorTransform = (glm::float3x4)whiteBalance;

        // Calculate source illuminant, i.e. the color that transforms to a pure white (1, 1, 1) output at the current color settings.
        mSourceWhite = inverse(whiteBalance) * glm::float3(1, 1, 1);
    }

    void ToneMappingPass::updateConstants()
    {
        // Calculate final transform.
        mLinearScale = pow(2.f, -mExposureValue) * mFilmSpeed / 100.f;
        mToneMappingData.finalTransform = mColorTransform * mLinearScale;     // Note: linearScale is baked into the transform.
    }

    void ToneMappingPass::setOperator(Operator op)
    {
        if (op != mOperator) createToneMapPass(op);
    }

    void ToneMappingPass::setExposureValue(float exposureValue)
    {
        mExposureValue = clamp(exposureValue, -24.f, 24.f);
        updateConstants();
    }

    void ToneMappingPass::setFilmSpeed(float filmSpeed)
    {
        mFilmSpeed = clamp(filmSpeed, 1.f, 6400.f);
        updateConstants();
    }

    void ToneMappingPass::setWhitePoint(float whitePoint)
    {
        mWhitePoint = clamp(whitePoint, 1905.f, 25000.f);
        calculateColorTransform();
        updateConstants();
    }

    SCRIPT_BINDING(ToneMappingPass)
    {
        auto c = m.regClass(ToneMappingPass);
        c.func_("operator", &ToneMappingPass::setOperator);
        c.func_("operator", &ToneMappingPass::getOperator);
        c.func_("exposureKey", &ToneMappingPass::setExposureKey);
        c.func_("exposureKey", &ToneMappingPass::getExposureKey);
        c.func_("whiteMaxLuminance", &ToneMappingPass::setWhiteMaxLuminance);
        c.func_("whiteMaxLuminance", &ToneMappingPass::getWhiteMaxLuminance);
        c.func_("luminanceLod", &ToneMappingPass::setLuminanceLod);
        c.func_("luminanceLod", &ToneMappingPass::getLuminanceLod);
        c.func_("whiteScale", &ToneMappingPass::setWhiteScale);
        c.func_("whiteScale", &ToneMappingPass::getWhiteScale);
        c.func_("exposureValue", &ToneMappingPass::setExposureValue);
        c.func_("exposureValue", &ToneMappingPass::getExposureValue);
        c.func_("filmSpeed", &ToneMappingPass::setFilmSpeed);
        c.func_("filmSpeed", &ToneMappingPass::getFilmSpeed);
        c.func_("whitePoint", &ToneMappingPass::setWhitePoint);
        c.func_("whitePoint", &ToneMappingPass::getWhitePoint);

        auto op = m.enum_<ToneMappingPass::Operator>("ToneMapOp");
        op.regEnumVal(ToneMappingPass::Operator::Clamp).regEnumVal(ToneMappingPass::Operator::Linear).regEnumVal(ToneMappingPass::Operator::Reinhard);
        op.regEnumVal(ToneMappingPass::Operator::ReinhardModified).regEnumVal(ToneMappingPass::Operator::HejiHableAlu);
        op.regEnumVal(ToneMappingPass::Operator::HableUc2).regEnumVal(ToneMappingPass::Operator::Aces).regEnumVal(ToneMappingPass::Operator::Photo);
    }
}
