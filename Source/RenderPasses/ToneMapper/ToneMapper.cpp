/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "ToneMapper.h"
#include "Utils/Color/ColorUtils.h"

namespace
{
    const Gui::DropdownList kOperatorList =
    {
        { (uint32_t)ToneMapper::Operator::Linear, "Linear" },
        { (uint32_t)ToneMapper::Operator::Reinhard, "Reinhard" },
        { (uint32_t)ToneMapper::Operator::ReinhardModified, "Modified Reinhard" },
        { (uint32_t)ToneMapper::Operator::HejiHableAlu, "Heji's approximation" },
        { (uint32_t)ToneMapper::Operator::HableUc2, "Uncharted 2" },
        { (uint32_t)ToneMapper::Operator::Aces, "ACES" },
    };

    const Gui::DropdownList kExposureModeList =
    {
        { (uint32_t)ToneMapper::ExposureMode::AperturePriority, "Aperture priority" },
        { (uint32_t)ToneMapper::ExposureMode::ShutterPriority, "Shutter priority" },
    };

    const char kSrc[] = "src";
    const char kDst[] = "dst";

    const char kOutputFormat[] = "outputFormat";

    const char kExposureCompensation[] = "exposureCompensation";
    const char kAutoExposure[] = "autoExposure";
    const char kExposureValue[] = "exposureValue";
    const char kFilmSpeed[] = "filmSpeed";
    const char kFNumber[] = "fNumber";
    const char kShutter[] = "shutter";
    const char kExposureMode[] = "exposureMode";

    const char kWhiteBalance[] = "whiteBalance";
    const char kWhitePoint[] = "whitePoint";

    const char kOperator[] = "operator";
    const char kClamp[] = "clamp";
    const char kWhiteMaxLuminance[] = "whiteMaxLuminance";
    const char kWhiteScale[] = "whiteScale";

    const char kLuminanceFile[] = "RenderPasses/ToneMapper/Luminance.ps.slang";
    const char kToneMappingFile[] = "RenderPasses/ToneMapper/ToneMapping.ps.slang";

    const float kExposureCompensationMin = -12.f;
    const float kExposureCompensationMax = 12.f;

    const float kFilmSpeedMin = 1.f;
    const float kFilmSpeedMax = 6400.f;

    const float kFNumberMin = 0.1f;          // Minimum fNumber, > 0 to avoid numerical issues (i.e., non-physical values are allowed)
    const float kFNumberMax = 100.f;

    const float kShutterMin = 0.1f;          // Min reciprocal shutter time
    const float kShutterMax = 10000.f;       // Max reciprocal shutter time

    // EV is ultimately derived from shutter and fNumber; set its range based on that of its inputs
    const float kExposureValueMin = std::log2(kShutterMin * kFNumberMin * kFNumberMin);
    const float kExposureValueMax = std::log2(kShutterMax * kFNumberMax * kFNumberMax);

    // Note: Color temperatures < ~1905K are out-of-gamut in Rec.709.
    const float kWhitePointMin = 1905.f;
    const float kWhitePointMax = 25000.f;
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

static void regToneMapper(pybind11::module& m)
{
    pybind11::class_<ToneMapper, RenderPass, ToneMapper::SharedPtr> pass(m, "ToneMapper");
    pass.def_property(kExposureCompensation, &ToneMapper::getExposureCompensation, &ToneMapper::setExposureCompensation);
    pass.def_property(kAutoExposure, &ToneMapper::getAutoExposure, &ToneMapper::setAutoExposure);
    pass.def_property(kExposureValue, &ToneMapper::getExposureValue, &ToneMapper::setExposureValue);
    pass.def_property(kFilmSpeed, &ToneMapper::getFilmSpeed, &ToneMapper::setFilmSpeed);
    pass.def_property(kWhiteBalance, &ToneMapper::getWhiteBalance, &ToneMapper::setWhiteBalance);
    pass.def_property(kWhitePoint, &ToneMapper::getWhitePoint, &ToneMapper::setWhitePoint);
    pass.def_property(kOperator, &ToneMapper::getOperator, &ToneMapper::setOperator);
    pass.def_property(kClamp, &ToneMapper::getClamp, &ToneMapper::setClamp);
    pass.def_property(kWhiteMaxLuminance, &ToneMapper::getWhiteMaxLuminance, &ToneMapper::setWhiteMaxLuminance);
    pass.def_property(kWhiteScale, &ToneMapper::getWhiteScale, &ToneMapper::setWhiteScale);
    pass.def_property(kFNumber, &ToneMapper::getFNumber, &ToneMapper::setFNumber);
    pass.def_property(kShutter, &ToneMapper::getShutter, &ToneMapper::setShutter);
    pass.def_property(kExposureMode, &ToneMapper::getExposureMode, &ToneMapper::setExposureMode);

    pybind11::enum_<ToneMapper::Operator> op(m, "ToneMapOp");
    op.value("Linear", ToneMapper::Operator::Linear);
    op.value("Reinhard", ToneMapper::Operator::Reinhard);
    op.value("ReinhardModified", ToneMapper::Operator::ReinhardModified);
    op.value("HejiHableAlu", ToneMapper::Operator::HejiHableAlu);
    op.value("HableUc2", ToneMapper::Operator::HableUc2);
    op.value("Aces", ToneMapper::Operator::Aces);

    pybind11::enum_<ToneMapper::ExposureMode> exposureMode(m, "ExposureMode");
    exposureMode.value("AperturePriority", ToneMapper::ExposureMode::AperturePriority);
    exposureMode.value("ShutterPriority", ToneMapper::ExposureMode::ShutterPriority);
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("ToneMapper", "Tone-map a color-buffer", ToneMapper::create);
    ScriptBindings::registerBinding(regToneMapper);
}

const char* ToneMapper::kDesc = "Tone-map a color-buffer. The resulting buffer is always in the [0, 1] range. The pass supports auto-exposure and eye-adaptation";

ToneMapper::ToneMapper(ToneMapper::Operator op, ResourceFormat outputFormat) : mOperator(op), mOutputFormat(outputFormat)
{
    createLuminancePass();
    createToneMapPass();

    updateWhiteBalanceTransform();

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(samplerDesc);
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
    mpLinearSampler = Sampler::create(samplerDesc);
}

ToneMapper::SharedPtr ToneMapper::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    // outputFormat can only be set on construction
    ResourceFormat outputFormat = ResourceFormat::Unknown;
    if (dict.keyExists(kOutputFormat)) outputFormat = dict[kOutputFormat];

    ToneMapper* pTM = new ToneMapper(Operator::Aces, outputFormat);

    for (const auto& [key, value] : dict)
    {
        if (key == kExposureCompensation) pTM->setExposureCompensation(value);
        else if (key == kAutoExposure) pTM->setAutoExposure(value);
        else if (key == kExposureValue) pTM->setExposureValue(value);
        else if (key == kFilmSpeed) pTM->setFilmSpeed(value);
        else if (key == kWhiteBalance) pTM->setWhiteBalance(value);
        else if (key == kWhitePoint) pTM->setWhitePoint(value);
        else if (key == kOperator) pTM->setOperator(value);
        else if (key == kClamp) pTM->setClamp(value);
        else if (key == kWhiteMaxLuminance) pTM->setWhiteMaxLuminance(value);
        else if (key == kWhiteScale) pTM->setWhiteScale(value);
        else if (key == kFNumber) pTM->setFNumber(value);
        else if (key == kShutter) pTM->setShutter(value);
        else if (key == kExposureMode) pTM->setExposureMode(value);
        else logWarning("Unknown field '" + key + "' in a ToneMapping dictionary");
    }

    return ToneMapper::SharedPtr(pTM);
}

Dictionary ToneMapper::getScriptingDictionary()
{
    Dictionary d;
    if (mOutputFormat != ResourceFormat::Unknown) d[kOutputFormat] = mOutputFormat;
    d[kExposureCompensation] = mExposureCompensation;
    d[kAutoExposure] = mAutoExposure;
    d[kExposureValue] = mExposureValue;
    d[kFilmSpeed] = mFilmSpeed;
    d[kWhiteBalance] = mWhiteBalance;
    d[kWhitePoint] = mWhitePoint;
    d[kOperator] = mOperator;
    d[kClamp] = mClamp;
    d[kWhiteMaxLuminance] = mWhiteMaxLuminance;
    d[kWhiteScale] = mWhiteScale;
    d[kFNumber] = mFNumber;
    d[kShutter] = mShutter;
    d[kExposureMode] = mExposureMode;
    return d;
}

RenderPassReflection ToneMapper::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kSrc, "Source texture");
    auto& output = reflector.addOutput(kDst, "Tone-mapped output texture");
    if (mOutputFormat != ResourceFormat::Unknown)
    {
        output.format(mOutputFormat);
    }

    return reflector;
}

void ToneMapper::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pSrc = renderData[kSrc]->asTexture();
    auto pDst = renderData[kDst]->asTexture();
    Fbo::SharedPtr pFbo = Fbo::create();
    pFbo->attachColorTarget(pDst, 0);

    // Run luminance pass if auto exposure is enabled
    if (mAutoExposure)
    {
        createLuminanceFbo(pSrc);

        mpLuminancePass["gColorTex"] = pSrc;
        mpLuminancePass["gColorSampler"] = mpLinearSampler;

        mpLuminancePass->execute(pRenderContext, mpLuminanceFbo);
        mpLuminanceFbo->getColorTexture(0)->generateMips(pRenderContext);
    }

    // Run main pass
    if (mRecreateToneMapPass)
    {
        createToneMapPass();
        mUpdateToneMapPass = true;
        mRecreateToneMapPass = false;
    }

    if (mUpdateToneMapPass)
    {
        updateWhiteBalanceTransform();
        updateColorTransform();

        ToneMapperParams params;
        params.whiteScale = mWhiteScale;
        params.whiteMaxLuminance = mWhiteMaxLuminance;
        params.colorTransform = static_cast<float3x4>(mColorTransform);
        mpToneMapPass->getRootVar()["PerImageCB"]["gParams"].setBlob(&params, sizeof(params));
        mUpdateToneMapPass = false;
    }

    mpToneMapPass["gColorTex"] = pSrc;
    mpToneMapPass["gColorSampler"] = mpPointSampler;

    if (mAutoExposure)
    {
        mpToneMapPass["gLuminanceTexSampler"] = mpLinearSampler;
        mpToneMapPass["gLuminanceTex"] = mpLuminanceFbo->getColorTexture(0);
    }

    mpToneMapPass->execute(pRenderContext, pFbo);
}

void ToneMapper::createLuminanceFbo(const Texture::SharedPtr& pSrc)
{
    bool createFbo = mpLuminanceFbo == nullptr;
    ResourceFormat srcFormat = pSrc->getFormat();
    uint32_t bytesPerChannel = getFormatBytesPerBlock(srcFormat) / getFormatChannelCount(srcFormat);

    // Find the required texture size and format
    ResourceFormat luminanceFormat = (bytesPerChannel == 32) ? ResourceFormat::R32Float : ResourceFormat::R16Float;
    uint32_t requiredHeight = getLowerPowerOf2(pSrc->getHeight());
    uint32_t requiredWidth = getLowerPowerOf2(pSrc->getWidth());

    if (createFbo == false)
    {
        createFbo = (requiredWidth != mpLuminanceFbo->getWidth()) ||
            (requiredHeight != mpLuminanceFbo->getHeight()) ||
            (luminanceFormat != mpLuminanceFbo->getColorTexture(0)->getFormat());
    }

    if (createFbo)
    {
        Fbo::Desc desc;
        desc.setColorTarget(0, luminanceFormat);
        mpLuminanceFbo = Fbo::create2D(requiredWidth, requiredHeight, desc, 1, Fbo::kAttachEntireMipLevel);
    }
}

// Update camera settings based on EV and exposure mode
void ToneMapper::updateCameraSettings()
{
    if (mExposureMode == ExposureMode::AperturePriority)
    {
        // Set shutter based on EV and aperture
        mShutter = std::pow(2.f, mExposureValue) / (mFNumber * mFNumber);
        // Clamp to plausible value
        mShutter = glm::clamp(mShutter, kShutterMin, kShutterMax);
        // Recompute EV based on clamped shutter
        updateExposureValue();
    }
    else if (mExposureMode == ExposureMode::ShutterPriority)
    {
        // Set aperture based on EV and shutter
        mFNumber = std::sqrt(std::pow(2.f, mExposureValue) / mShutter);
        // Clamp to plausible value
        mFNumber = glm::clamp(mFNumber, kFNumberMin, kFNumberMax);
        // Recompute EV based on clamped aperture
        updateExposureValue();
    }
    else
    {
        should_not_get_here();
    }
}

// Set EV based on fNumber and shutter
void ToneMapper::updateExposureValue()
{
    mExposureValue = std::log2(mShutter * mFNumber * mFNumber);
}

void ToneMapper::renderUI(Gui::Widgets& widget)
{
    if (auto exposureGroup = widget.group("Exposure", true))
    {
        mUpdateToneMapPass |= exposureGroup.var("Exposure Compensation", mExposureCompensation, kExposureCompensationMin, kExposureCompensationMax, 0.1f, false, "%.1f");

        mRecreateToneMapPass |= exposureGroup.checkbox("Auto Exposure", mAutoExposure);

        if (!mAutoExposure)
        {
            uint32_t emIndex = static_cast<uint32_t>(mExposureMode);
            if (exposureGroup.dropdown("Exposure mode", kExposureModeList, emIndex))
            {
                setExposureMode(ExposureMode(emIndex));
            }

            if (exposureGroup.var("Exposure Value (EV)", mExposureValue, kExposureValueMin, kExposureValueMax, 0.1f, false, "%.1f"))
            {
                setExposureValue(mExposureValue);
            }

            mUpdateToneMapPass |= exposureGroup.var("Film Speed (ISO)", mFilmSpeed, kFilmSpeedMin, kFilmSpeedMax, 0.1f, false, "%.1f");

            if (exposureGroup.var("f-Number", mFNumber, kFNumberMin, kFNumberMax, 0.1f, false, "%.1f"))
            {
                setFNumber(mFNumber);
            }

            if (exposureGroup.var("Shutter", mShutter, kShutterMin, kShutterMax, 0.1f, false, "%.1f"))
            {
                setShutter(mShutter);
            }
        }
    }

    if (auto colorgradingGroup = widget.group("Color Grading", true))
    {
        mUpdateToneMapPass |= colorgradingGroup.checkbox("White Balance", mWhiteBalance);

        if (mWhiteBalance)
        {
            if (colorgradingGroup.var("White Point (K)", mWhitePoint, kWhitePointMin, kWhitePointMax, 5.f, false, "%.0f"))
            {
                updateWhiteBalanceTransform();
                mUpdateToneMapPass = true;
            }

            // Display color widget for the currently chosen white point.
            // We normalize the color so that max(RGB) = 1 for display purposes.
            float3 w = mSourceWhite;
            w = w / std::max(std::max(w.r, w.g), w.b);
            colorgradingGroup.rgbColor("", w);
        }
    }

    if (auto tonemappingGroup = widget.group("Tonemapping", true))
    {
        uint32_t opIndex = static_cast<uint32_t>(mOperator);
        if (tonemappingGroup.dropdown("Operator", kOperatorList, opIndex))
        {
            setOperator(Operator(opIndex));
        }

        if (mOperator == Operator::ReinhardModified)
        {
            mUpdateToneMapPass |= tonemappingGroup.var("White Luminance", mWhiteMaxLuminance, 0.1f, FLT_MAX, 0.2f);
        }
        else if (mOperator == Operator::HableUc2)
        {
            mUpdateToneMapPass |= tonemappingGroup.var("Linear White", mWhiteScale, 0.f, 100.f, 0.01f);
        }

        mRecreateToneMapPass |= tonemappingGroup.checkbox("Clamp Output", mClamp);
    }
}

void ToneMapper::setExposureCompensation(float exposureCompensation)
{
    mExposureCompensation = glm::clamp(exposureCompensation, kExposureCompensationMin, kExposureCompensationMax);
    mUpdateToneMapPass = true;
}

void ToneMapper::setAutoExposure(bool autoExposure)
{
    mAutoExposure = autoExposure;
    mRecreateToneMapPass = true;
}

void ToneMapper::setExposureValue(float exposureValue)
{
    mExposureValue = glm::clamp(exposureValue, kExposureValueMin, kExposureValueMax);
    updateCameraSettings();
    mUpdateToneMapPass = true;
}

void ToneMapper::setFilmSpeed(float filmSpeed)
{
    mFilmSpeed = glm::clamp(filmSpeed, kFilmSpeedMin, kFilmSpeedMax);
    mUpdateToneMapPass = true;
}

void ToneMapper::setWhiteBalance(bool whiteBalance)
{
    mWhiteBalance = whiteBalance;
    mUpdateToneMapPass = true;
}

void ToneMapper::setWhitePoint(float whitePoint)
{
    mWhitePoint = glm::clamp(whitePoint, kWhitePointMin, kWhitePointMax);
    mUpdateToneMapPass = true;
}

void ToneMapper::setOperator(Operator op)
{
    if (op != mOperator)
    {
        mOperator = op;
        mRecreateToneMapPass = true;
    }
}

void ToneMapper::setClamp(bool clamp)
{
    if (clamp != mClamp)
    {
        mClamp = clamp;
        mRecreateToneMapPass = true;
    }
}

void ToneMapper::setWhiteMaxLuminance(float maxLuminance)
{
    mWhiteMaxLuminance = maxLuminance;
    mUpdateToneMapPass = true;
}

void ToneMapper::setWhiteScale(float whiteScale)
{
    mWhiteScale = std::max(0.001f, whiteScale);
    mUpdateToneMapPass = true;
}

void ToneMapper::setFNumber(float fNumber)
{
    mFNumber = glm::clamp(fNumber, kFNumberMin, kFNumberMax);
    updateExposureValue();
    mUpdateToneMapPass = true;
}

void ToneMapper::setShutter(float shutter)
{
    mShutter = glm::clamp(shutter, kShutterMin, kShutterMax);
    updateExposureValue();
    mUpdateToneMapPass = true;
}

void ToneMapper::setExposureMode(ExposureMode mode)
{
    mExposureMode = mode;
}

void ToneMapper::createLuminancePass()
{
    mpLuminancePass = FullScreenPass::create(kLuminanceFile);
}

void ToneMapper::createToneMapPass()
{
    Program::DefineList defines;
    defines.add("_TONE_MAPPER_OPERATOR", std::to_string(static_cast<uint32_t>(mOperator)));
    if (mAutoExposure) defines.add("_TONE_MAPPER_AUTO_EXPOSURE");
    if (mClamp) defines.add("_TONE_MAPPER_CLAMP");

    mpToneMapPass = FullScreenPass::create(kToneMappingFile, defines);
}

void ToneMapper::updateWhiteBalanceTransform()
{
    // Calculate color transform for the current white point.
    mWhiteBalanceTransform = mWhiteBalance ? calculateWhiteBalanceTransformRGB_Rec709(mWhitePoint) : glm::identity<float3x3>();
    // Calculate source illuminant, i.e. the color that transforms to a pure white (1, 1, 1) output at the current color settings.
    mSourceWhite = inverse(mWhiteBalanceTransform) * float3(1, 1, 1);
}

void ToneMapper::updateColorTransform()
{
    // Exposure scale due to exposure compensation.
    float exposureScale = pow(2.f, mExposureCompensation);
    float manualExposureScale = 1.f;
    if (!mAutoExposure)
    {
        float normConstant = 1.f / 100.f;
        manualExposureScale = (normConstant * mFilmSpeed) / (mShutter * mFNumber * mFNumber);
    }
    // Calculate final transform.
    mColorTransform = mWhiteBalanceTransform * exposureScale * manualExposureScale;
}
