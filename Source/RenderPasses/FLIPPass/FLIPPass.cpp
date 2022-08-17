/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "FLIPPass.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "Utils/Algorithm/ComputeParallelReduction.h"

const RenderPass::Info FLIPPass::kInfo
{
    "FLIPPass",

    "FLIP Metric Pass.\n\n"
    "If the input has high dynamic range, check the \"Compute HDR-FLIP\" box below.\n\n"
    "The errorMapDisplay shows the FLIP error map. When HDR-FLIP is computed, the user may also show the HDR-FLIP exposure map.\n\n"
    "When \"List all output\" is checked, the user may also store the errorMap. This is a high-precision, linear buffer "
    "which is transformed to sRGB before display. NOTE: This sRGB transform will make the displayed output look different compared "
    "to the errorMapDisplay. The transform is only added before display, however, and will NOT affect the output when it is saved to disk."
};

namespace
{
    const char kFLIPShaderFile[] = "RenderPasses/FLIPPass/FLIPPass.cs.slang";
    const char kComputeLuminanceShaderFile[] = "RenderPasses/FLIPPass/ComputeLuminance.cs.slang";

    const char kTestImageInput[] = "testImage";
    const char kReferenceImageInput[] = "referenceImage";
    const char kErrorMapOutput[] = "errorMap"; // High-precision FLIP error map - use for computations (not display).
    const char kErrorMapDisplayOutput[] = "errorMapDisplay"; // Low-precision FLIP error map - use for display / analysis (not computations).
    const char kExposureMapDisplayOutput[] = "exposureMapDisplay"; // Low-precision HDR-FLIP exposure map.

    const Gui::DropdownList kToneMapperList =
    {
        { (uint32_t)FLIPToneMapperType::ACES, "ACES" },
        { (uint32_t)FLIPToneMapperType::Hable, "Hable"},
        { (uint32_t)FLIPToneMapperType::Reinhard, "Reinhard"},
    };

    const char kEnabled[] = "enabled";
    const char kIsHDR[] = "isHDR";
    const char kToneMapper[] = "toneMapper";
    const char kUseCustomExposureParameters[] = "useCustomExposureParameters";
    const char kStartExposure[] = "startExposure";
    const char kStopExposure[] = "stopExposure";
    const char kNumExposures[] = "numExposures";
    const char kUseMagma[] = "useMagma";
    const char kClampInput[] = "clampInput";
    const char kMonitorWidthPixels[] = "monitorWidthPixels";
    const char kMonitorWidthMeters[] = "monitorWidthMeters";
    const char kMonitorDistance[] = "monitorDistanceMeters";
    const char kComputePooledFLIPValues[] = "computePooledFLIPValues";
    const char kUseRealMonitorInfo[] = "useRealMonitorInfo";
}

// Don't remove this. it's required for hot-reload to function properly.
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(FLIPPass::kInfo, FLIPPass::create);
    ScriptBindings::registerBinding(FLIPPass::registerBindings);
}

void FLIPPass::registerBindings(pybind11::module& m)
{
    pybind11::enum_<FLIPToneMapperType> toneMapper(m, "FLIPToneMapperType");
    toneMapper.value("ACES", FLIPToneMapperType::ACES);
    toneMapper.value("Hable", FLIPToneMapperType::Hable);
    toneMapper.value("Reinhard", FLIPToneMapperType::Reinhard);
}

FLIPPass::SharedPtr FLIPPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new FLIPPass(dict));
}

FLIPPass::FLIPPass(const Dictionary& dict)
    : RenderPass(kInfo)
{
    parseDictionary(dict);

    Program::DefineList defines;
    defines.add("TONE_MAPPER", std::to_string((uint32_t)mToneMapper));

    mpFLIPPass = ComputePass::create(kFLIPShaderFile, "main", defines);
    mpComputeLuminancePass = ComputePass::create(kComputeLuminanceShaderFile, "computeLuminance", Program::DefineList());

    // Create parallel reduction helper.
    mpParallelReduction = ComputeParallelReduction::create();

    // Fill some reasonable defaults for monitor information.
    mMonitorWidthPixels = 3840;
    mMonitorWidthMeters = 0.7f;
    mMonitorDistanceMeters = 0.7f;

    // Evaluate monitor information.
    std::vector<MonitorInfo::MonitorDesc> monitorDescs = MonitorInfo::getMonitorDescs();

    // Override defaults by real monitor info, if available.
    if (mUseRealMonitorInfo && (monitorDescs.size() > 0))
    {
        // Assume first monitor is used.
        size_t monitorIndex = 0;
        if (monitorDescs[monitorIndex].resolution.x > 0) mMonitorWidthPixels = monitorDescs[0].resolution.x;
        if (monitorDescs[monitorIndex].physicalSize.x > 0) mMonitorWidthMeters = monitorDescs[0].physicalSize.x * 0.0254f; //< Convert from inches to meters
    }

}

Dictionary FLIPPass::getScriptingDictionary()
{
    Dictionary d;
    d[kEnabled] = mEnabled;
    d[kUseMagma] = mUseMagma;
    d[kClampInput] = mClampInput;
    d[kIsHDR] = mIsHDR;
    d[kToneMapper] = mToneMapper;
    d[kUseCustomExposureParameters] = mUseCustomExposureParameters;
    d[kStartExposure] = mStartExposure;
    d[kStopExposure] = mStopExposure;
    d[kNumExposures] = mNumExposures;
    d[kMonitorWidthPixels] = mMonitorWidthPixels;
    d[kMonitorWidthMeters] = mMonitorWidthMeters;
    d[kMonitorDistance] = mMonitorDistanceMeters;
    d[kComputePooledFLIPValues] = mComputePooledFLIPValues;
    d[kUseRealMonitorInfo] = mUseRealMonitorInfo;
    return d;
}

void FLIPPass::parseDictionary(const Dictionary& dict)
{
    // Read settings.
    for (const auto& [key, value] : dict)
    {
        if (key == kEnabled) mEnabled = value;
        else if (key == kIsHDR) mIsHDR = value;
        else if (key == kToneMapper) mToneMapper = FLIPToneMapperType(value);
        else if (key == kUseCustomExposureParameters) mUseCustomExposureParameters = value;
        else if (key == kStartExposure) mStartExposure = value;
        else if (key == kStopExposure) mStopExposure = value;
        else if (key == kNumExposures) mNumExposures = value;
        else if (key == kUseMagma) mUseMagma = value;
        else if (key == kClampInput) mClampInput = value;
        else if (key == kMonitorWidthPixels) mMonitorWidthPixels = value;
        else if (key == kMonitorWidthMeters) mMonitorWidthMeters = value;
        else if (key == kMonitorDistance) mMonitorDistanceMeters = value;
        else if (key == kComputePooledFLIPValues) mComputePooledFLIPValues = value;
        else if (key == kUseRealMonitorInfo) mUseRealMonitorInfo = value;
        else logWarning("Unknown field '{}' in a FLIPPass dictionary.", key);
    }
}


RenderPassReflection FLIPPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kTestImageInput, "Test image").bindFlags(Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    reflector.addInput(kReferenceImageInput, "Reference image").bindFlags(Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    reflector.addOutput(kErrorMapOutput, "FLIP error map for computations").format(ResourceFormat::RGBA32Float).bindFlags(Falcor::Resource::BindFlags::UnorderedAccess | Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    reflector.addOutput(kErrorMapDisplayOutput, "FLIP error map for display").format(ResourceFormat::RGBA8UnormSrgb).bindFlags(Falcor::Resource::BindFlags::RenderTarget).texture2D(0, 0);
    reflector.addOutput(kExposureMapDisplayOutput, "HDR-FLIP exposure map for display").format(ResourceFormat::RGBA8UnormSrgb).bindFlags(Falcor::Resource::BindFlags::RenderTarget).texture2D(0, 0);
    return reflector;
}

void FLIPPass::updatePrograms()
{
    if (mRecompile == false)
    {
        return;
    }

    Program::DefineList defines;
    defines.add("TONE_MAPPER", std::to_string((uint32_t)mToneMapper));

    mpFLIPPass->getProgram()->addDefines(defines);

    mRecompile = false;
}

static void solveSecondDegree(const float a, const float b, float c, float& xMin, float& xMax)
{
    //  Solve a * x^2 + b * x + c = 0.
    if (a == 0.0f)
    {
        xMin = xMax = -c / b;
        return;
    }

    float d1 = -0.5f * (b / a);
    float d2 = std::sqrt((d1 * d1) - (c / a));
    xMin = d1 - d2;
    xMax = d1 + d2;
}

// At some point, we want to replace this with a compute pass. The BitonicSort needs to be updated so it can handle
// large amounts of floatintpoint numbers.
static void computeMedianMax(const float* values, const uint32_t numValues, float& median, float& max)
{
    std::vector<float> sortedValues(values, values + numValues);
    std::sort(sortedValues.begin(), sortedValues.end());
    if (numValues & 1)      // Odd number of values.
    {
        median = sortedValues[numValues / 2];
    }
    else                    // Even number of values.
    {
        uint32_t medianLocation = numValues / 2 - 1;
        median = (sortedValues[medianLocation] + sortedValues[medianLocation + 1]) * 0.5f;
    }
    max = sortedValues[numValues - 1];
}

void FLIPPass::computeExposureParameters(const float Ymedian, const float Ymax)
{
    std::vector<float> tmCoefficients;
    if (mToneMapper == FLIPToneMapperType::Reinhard)
    {
        tmCoefficients = { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f };
    }
    else if (mToneMapper == FLIPToneMapperType::ACES)
    {
        tmCoefficients = { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f };  // 0.6 is pre-exposure cancellation.
    }
    else if (mToneMapper == FLIPToneMapperType::Hable)
    {
        tmCoefficients = { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f };
    }
    else
    {
        FALCOR_UNREACHABLE();
    }

    const float t = 0.85f;
    const float a = tmCoefficients[0] - t * tmCoefficients[3];
    const float b = tmCoefficients[1] - t * tmCoefficients[4];
    const float c = tmCoefficients[2] - t * tmCoefficients[5];

    float xMin = 0.0f;
    float xMax = 0.0f;
    solveSecondDegree(a, b, c, xMin, xMax);

    mStartExposure = std::log2(xMax / Ymax);
    float stopExposure = std::log2(xMax / Ymedian);

    mNumExposures = uint32_t(std::max(2.0f, std::ceil(stopExposure - mStartExposure)));
    mExposureDelta = (stopExposure - mStartExposure) / (mNumExposures - 1.0f);
}

void FLIPPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mEnabled) return;

    // Pick up resources from render graph.
    const auto& pTestImageInput = renderData.getTexture(kTestImageInput);
    const auto& pReferenceImageInput = renderData.getTexture(kReferenceImageInput);
    const auto& pErrorMapOutput = renderData.getTexture(kErrorMapOutput);
    const auto& pErrorMapDisplayOutput = renderData.getTexture(kErrorMapDisplayOutput);
    const auto& pExposureMapDisplayOutput = renderData.getTexture(kExposureMapDisplayOutput);

    updatePrograms();

    // Check for mandatory resources.
    if (!pTestImageInput || !pReferenceImageInput || !pErrorMapOutput || !pErrorMapDisplayOutput || !pExposureMapDisplayOutput)
    {
        logWarning("FLIPPass::execute() - missing mandatory resources");
        return;
    }

    // Refresh internal high precision buffer for FLIP results.
    uint2 outputResolution = uint2(pReferenceImageInput->getWidth(), pReferenceImageInput->getHeight());
    if (!mpFLIPErrorMapDisplay || !mpExposureMapDisplay || mpFLIPErrorMapDisplay->getWidth() != outputResolution.x || mpFLIPErrorMapDisplay->getHeight() != outputResolution.y)
    {
        mpFLIPErrorMapDisplay = Texture::create2D(outputResolution.x, outputResolution.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
        mpExposureMapDisplay = Texture::create2D(outputResolution.x, outputResolution.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    }

    if (!mpLuminance)
    {
        mpLuminance = Buffer::create(outputResolution.x * outputResolution.y * sizeof(float), ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None);
    }

    // Bind resources.
    mpFLIPPass["gTestImage"] = pTestImageInput;
    mpFLIPPass["gReferenceImage"] = pReferenceImageInput;
    mpFLIPPass["gFLIPErrorMap"] = pErrorMapOutput;
    mpFLIPPass["gFLIPErrorMapDisplay"] = mpFLIPErrorMapDisplay;
    mpFLIPPass["gExposureMapDisplay"] = mpExposureMapDisplay;

    // Bind CB settings.
    auto var = mpFLIPPass["PerFrameCB"];
    var["gIsHDR"] = mIsHDR;
    var["gUseMagma"] = mUseMagma;
    var["gClampInput"] = mUseMagma;
    var["gResolution"] = outputResolution;
    var["gMonitorWidthPixels"] = mMonitorWidthPixels;
    var["gMonitorWidthMeters"] = mMonitorWidthMeters;
    var["gMonitorDistance"] = mMonitorDistanceMeters;
    var["gStartExposure"] = mStartExposure;
    var["gExposureDelta"] = mExposureDelta;
    var["gNumExposures"] = mNumExposures;

    // Do we need to compute parameters for HDR-FLIP?
    if (!mUseCustomExposureParameters && mIsHDR)
    {
        // Bind resources.
        mpComputeLuminancePass["gInputImage"] = pReferenceImageInput;
        mpComputeLuminancePass["gOutputLuminance"] = mpLuminance;
        // Bind CB settings.
        mpComputeLuminancePass["PerFrameCB"]["gResolution"] = outputResolution;
        // Compute luminance of the reference image.
        mpComputeLuminancePass->execute(pRenderContext, uint3(outputResolution.x, outputResolution.y, 1u));
        pRenderContext->flush(true);

        float Ymedian, Ymax;
        const float* luminanceValues = (float*)mpLuminance->map(Buffer::MapType::Read);
        computeMedianMax(luminanceValues, outputResolution.x * outputResolution.y, Ymedian, Ymax);
        mpLuminance->unmap();

        computeExposureParameters(Ymedian, Ymax);
    }

    // Compute FLIP error map and exposure map.
    mpFLIPPass->execute(pRenderContext, outputResolution.x, outputResolution.y);

    // Convert display output to sRGB and reduce precision.
    pRenderContext->blit(mpFLIPErrorMapDisplay->getSRV(), pErrorMapDisplayOutput->getRTV());
    pRenderContext->blit(mpExposureMapDisplay->getSRV(), pExposureMapDisplayOutput->getRTV());

    // Compute mean, min, and max using parallel reduction.
    if (mComputePooledFLIPValues)
    {
        float4 FLIPSum, FLIPMinMax[2];
        mpParallelReduction->execute<float4>(pRenderContext, pErrorMapOutput, ComputeParallelReduction::Type::Sum, &FLIPSum);
        mpParallelReduction->execute<float4>(pRenderContext, pErrorMapOutput, ComputeParallelReduction::Type::MinMax, &FLIPMinMax[0]);
        pRenderContext->flush(true);

        // Extract metrics from readback values. RGB channels contain magma mapping, and the alpa channel contains FLIP value.
        mAverageFLIP = FLIPSum.a / (outputResolution.x * outputResolution.y);
        mMinFLIP = FLIPMinMax[0].a;
        mMaxFLIP = FLIPMinMax[1].a;
    }
}

void FLIPPass::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    dirty |= widget.checkbox("Enabled", mEnabled);

    widget.text("FLIP Settings:");
    dirty |= widget.checkbox("Use Magma", mUseMagma);
    dirty |= widget.checkbox("Clamp input", mClampInput);
    widget.tooltip("Clamp FLIP input to the expected range ([0,1] for LDR-FLIP and [0, inf) for HDR-FLIP).");
    dirty |= widget.checkbox("Input is HDR", mIsHDR);
    widget.tooltip("If input has high dynamic range, use HDR-FLIP instead of the default (LDR-FLIP) which only works for low dynamic range input.");

    if (mIsHDR)
    {
        dirty |= widget.dropdown("Tone mapper", kToneMapperList, reinterpret_cast<uint32_t&>(mToneMapper));
        widget.tooltip("The tone mapper assumed by HDR-FLIP.");
        dirty |= widget.checkbox("Use custom exposure parameters", mUseCustomExposureParameters);
        widget.tooltip("Check to manually choose start and stop exposure as well as number of exposures used for HDR-FLIP.");
        if (mUseCustomExposureParameters)
        {
            dirty |= widget.var("Start exposure", mStartExposure, -20.0f, 20.0f, 0.01f);
            dirty |= widget.var("Stop exposure", mStopExposure, -20.0f, 20.0f, 0.01f);
            dirty |= widget.var("Number of exposures", mNumExposures, 2u, 20u, 1u);
            mExposureDelta = (mStopExposure - mStartExposure) / (mNumExposures - 1.0f);
        }
        else
        {
            mStopExposure = mStartExposure + (mNumExposures - 1.0f) * mExposureDelta;
            widget.text("Start exposure: " + std::to_string(mStartExposure));
            widget.text("Stop exposure: " + std::to_string(mStopExposure));
            widget.text("Number of exposures: " + std::to_string(mNumExposures));
        }
    }

    dirty |= widget.checkbox("Per-frame metrics", mComputePooledFLIPValues);

    if (mComputePooledFLIPValues)
    {
        widget.indent(10.0f);
        widget.text("Mean: " + std::to_string(mAverageFLIP));
        widget.text("Min: " + std::to_string(mMinFLIP));
        widget.text("Max: " + std::to_string(mMaxFLIP));
        widget.indent(-10.0f);
    }

    widget.separator();
    widget.text("Monitor information:");
    dirty |= widget.var("Distance to monitor (meters)", mMonitorDistanceMeters, 0.3f, 1.5f, 0.01f);
    dirty |= widget.var("Monitor width (meters)", mMonitorWidthMeters, 0.3f, 1.5f, 0.01f);
    dirty |= widget.var("Monitor resolution (horizontal)", mMonitorWidthPixels, 720u, 7680u);

    if (dirty)
    {
        mRecompile = true;
    }
}
