/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "AccumulatePass.h"
#include "RenderGraph/RenderPassStandardFlags.h"

static void regAccumulatePass(pybind11::module& m)
{
    pybind11::class_<AccumulatePass, RenderPass, ref<AccumulatePass>> pass(m, "AccumulatePass");
    pass.def_property("enabled", &AccumulatePass::isEnabled, &AccumulatePass::setEnabled);
    pass.def("reset", &AccumulatePass::reset);
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AccumulatePass>();
    ScriptBindings::registerBinding(regAccumulatePass);
}

namespace
{
const char kShaderFile[] = "RenderPasses/AccumulatePass/Accumulate.cs.slang";

const char kInputChannel[] = "input";
const char kOutputChannel[] = "output";

// Serialized parameters
const char kEnabled[] = "enabled";
const char kOutputFormat[] = "outputFormat";
const char kOutputSize[] = "outputSize";
const char kFixedOutputSize[] = "fixedOutputSize";
const char kAutoReset[] = "autoReset";
const char kPrecisionMode[] = "precisionMode";
const char kMaxFrameCount[] = "maxFrameCount";
const char kOverflowMode[] = "overflowMode";
} // namespace

AccumulatePass::AccumulatePass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // Deserialize pass from dictionary.
    for (const auto& [key, value] : props)
    {
        if (key == kEnabled)
            mEnabled = value;
        else if (key == kOutputFormat)
            mOutputFormat = value;
        else if (key == kOutputSize)
            mOutputSizeSelection = value;
        else if (key == kFixedOutputSize)
            mFixedOutputSize = value;
        else if (key == kAutoReset)
            mAutoReset = value;
        else if (key == kPrecisionMode)
            mPrecisionMode = value;
        else if (key == kMaxFrameCount)
            mMaxFrameCount = value;
        else if (key == kOverflowMode)
            mOverflowMode = value;
        else
            logWarning("Unknown property '{}' in AccumulatePass properties.", key);
    }

    if (props.has("enableAccumulation"))
    {
        logWarning("'enableAccumulation' is deprecated. Use 'enabled' instead.");
        if (!props.has(kEnabled))
            mEnabled = props["enableAccumulation"];
    }

    mpState = ComputeState::create(mpDevice);
}

Properties AccumulatePass::getProperties() const
{
    Properties props;
    props[kEnabled] = mEnabled;
    if (mOutputFormat != ResourceFormat::Unknown)
        props[kOutputFormat] = mOutputFormat;
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        props[kFixedOutputSize] = mFixedOutputSize;
    props[kAutoReset] = mAutoReset;
    props[kPrecisionMode] = mPrecisionMode;
    props[kMaxFrameCount] = mMaxFrameCount;
    props[kOverflowMode] = mOverflowMode;
    return props;
}

RenderPassReflection AccumulatePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);
    const auto fmt = mOutputFormat != ResourceFormat::Unknown ? mOutputFormat : ResourceFormat::RGBA32Float;

    reflector.addInput(kInputChannel, "Input data to be temporally accumulated").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutputChannel, "Output data that is temporally accumulated")
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(fmt)
        .texture2D(sz.x, sz.y);
    return reflector;
}

void AccumulatePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mAutoReset)
    {
        // Query refresh flags passed down from the application and other passes.
        auto& dict = renderData.getDictionary();
        auto refreshFlags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);

        // If any refresh flag is set, we reset frame accumulation.
        if (refreshFlags != RenderPassRefreshFlags::None)
            reset();

        // Reset accumulation upon all scene changes, except camera jitter and history changes.
        // TODO: Add UI options to select which changes should trigger reset
        if (mpScene)
        {
            auto sceneUpdates = mpScene->getUpdates();
            if ((sceneUpdates & ~Scene::UpdateFlags::CameraPropertiesChanged) != Scene::UpdateFlags::None)
            {
                reset();
            }
            if (is_set(sceneUpdates, Scene::UpdateFlags::CameraPropertiesChanged))
            {
                auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
                auto cameraChanges = mpScene->getCamera()->getChanges();
                if ((cameraChanges & ~excluded) != Camera::Changes::None)
                    reset();
            }
        }
    }

    // Check if we reached max number of frames to accumulate and handle overflow.
    if (mMaxFrameCount > 0 && mFrameCount == mMaxFrameCount)
    {
        switch (mOverflowMode)
        {
        case OverflowMode::Stop:
            return;
        case OverflowMode::Reset:
            reset();
            break;
        case OverflowMode::EMA:
            break;
        }
    }

    // Grab our input/output buffers.
    ref<Texture> pSrc = renderData.getTexture(kInputChannel);
    ref<Texture> pDst = renderData.getTexture(kOutputChannel);
    FALCOR_ASSERT(pSrc && pDst);

    const uint2 resolution = uint2(pSrc->getWidth(), pSrc->getHeight());
    const bool resolutionMatch = pDst->getWidth() == resolution.x && pDst->getHeight() == resolution.y;

    // Reset accumulation when resolution changes.
    if (any(resolution != mFrameDim))
    {
        mFrameDim = resolution;
        reset();
    }

    // Verify that output is non-integer format. It shouldn't be since reflect() requests a floating-point format.
    if (isIntegerFormat(pDst->getFormat()))
        FALCOR_THROW("AccumulatePass: Output to integer format is not supported");

    // Issue error and disable pass if unsupported I/O size. The user can hit continue and fix the config or abort.
    if (mEnabled && !resolutionMatch)
    {
        logError("AccumulatePass I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    // Decide action based on current configuration:
    // - The accumulation pass supports integer input but requires matching I/O size.
    // - Blit supports mismatching size but requires non-integer format.
    // - As a fallback, issue warning and clear the output.

    if (!mEnabled && !isIntegerFormat(pSrc->getFormat()))
    {
        // Only blit mip 0 and array slice 0, because that's what the accumulation uses otherwise.
        pRenderContext->blit(pSrc->getSRV(0, 1, 0, 1), pDst->getRTV(0, 0, 1));
    }
    else if (resolutionMatch)
    {
        accumulate(pRenderContext, pSrc, pDst);
    }
    else
    {
        logWarning("AccumulatePass unsupported I/O configuration. The output will be cleared.");
        pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0));
    }
}

void AccumulatePass::accumulate(RenderContext* pRenderContext, const ref<Texture>& pSrc, const ref<Texture>& pDst)
{
    FALCOR_ASSERT(pSrc && pDst);
    FALCOR_ASSERT(pSrc->getWidth() == mFrameDim.x && pSrc->getHeight() == mFrameDim.y);
    FALCOR_ASSERT(pDst->getWidth() == mFrameDim.x && pDst->getHeight() == mFrameDim.y);
    const FormatType srcType = getFormatType(pSrc->getFormat());

    // If for the first time, or if the input format type has changed, (re)compile the programs.
    if (mpProgram.empty() || srcType != mSrcType)
    {
        DefineList defines;
        switch (srcType)
        {
        case FormatType::Uint:
            defines.add("_INPUT_FORMAT", "INPUT_FORMAT_UINT");
            break;
        case FormatType::Sint:
            defines.add("_INPUT_FORMAT", "INPUT_FORMAT_SINT");
            break;
        default:
            defines.add("_INPUT_FORMAT", "INPUT_FORMAT_FLOAT");
            break;
        }
        // Create accumulation programs.
        // Note only compensated summation needs precise floating-point mode.
        mpProgram[Precision::Double] =
            Program::createCompute(mpDevice, kShaderFile, "accumulateDouble", defines, SlangCompilerFlags::TreatWarningsAsErrors);
        mpProgram[Precision::Single] =
            Program::createCompute(mpDevice, kShaderFile, "accumulateSingle", defines, SlangCompilerFlags::TreatWarningsAsErrors);
        mpProgram[Precision::SingleCompensated] = Program::createCompute(
            mpDevice,
            kShaderFile,
            "accumulateSingleCompensated",
            defines,
            SlangCompilerFlags::FloatingPointModePrecise | SlangCompilerFlags::TreatWarningsAsErrors
        );
        mpVars = ProgramVars::create(mpDevice, mpProgram[mPrecisionMode]->getReflector());

        mSrcType = srcType;
    }

    // Setup accumulation.
    prepareAccumulation(pRenderContext, mFrameDim.x, mFrameDim.y);

    // Set shader parameters.
    auto var = mpVars->getRootVar();
    var["PerFrameCB"]["gResolution"] = mFrameDim;
    var["PerFrameCB"]["gAccumCount"] = mFrameCount;
    var["PerFrameCB"]["gAccumulate"] = mEnabled;
    var["PerFrameCB"]["gMovingAverageMode"] = (mMaxFrameCount > 0);
    var["gCurFrame"] = pSrc;
    var["gOutputFrame"] = pDst;

    // Bind accumulation buffers. Some of these may be nullptr's.
    var["gLastFrameSum"] = mpLastFrameSum;
    var["gLastFrameCorr"] = mpLastFrameCorr;
    var["gLastFrameSumLo"] = mpLastFrameSumLo;
    var["gLastFrameSumHi"] = mpLastFrameSumHi;

    // Update the frame count.
    // The accumulation limit (mMaxFrameCount) has a special value of 0 (no limit) and is not supported in the SingleCompensated mode.
    if (mMaxFrameCount == 0 || mPrecisionMode == Precision::SingleCompensated || mFrameCount < mMaxFrameCount)
    {
        mFrameCount++;
    }

    // Run the accumulation program.
    auto pProgram = mpProgram[mPrecisionMode];
    FALCOR_ASSERT(pProgram);
    uint3 numGroups = div_round_up(uint3(mFrameDim.x, mFrameDim.y, 1u), pProgram->getReflector()->getThreadGroupSize());
    mpState->setProgram(pProgram);
    pRenderContext->dispatch(mpState.get(), mpVars.get(), numGroups);
}

void AccumulatePass::renderUI(Gui::Widgets& widget)
{
    // Controls for output size.
    // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
    if (widget.dropdown("Output size", mOutputSizeSelection))
        requestRecompile();
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
    {
        if (widget.var("Size in pixels", mFixedOutputSize, 32u, 16384u))
            requestRecompile();
    }

    if (bool enabled = isEnabled(); widget.checkbox("Enabled", enabled))
        setEnabled(enabled);

    if (mEnabled)
    {
        if (widget.button("Reset", true))
            reset();

        widget.checkbox("Auto Reset", mAutoReset);
        widget.tooltip("Reset accumulation automatically upon scene changes and refresh flags.");

        if (widget.dropdown("Mode", mPrecisionMode))
        {
            // Reset accumulation when mode changes.
            reset();
        }

        if (mPrecisionMode != Precision::SingleCompensated)
        {
            // When mMaxFrameCount is nonzero, the accumulate pass will only compute the average of
            // up to that number of frames. Further frames will be accumulated in the exponential moving
            // average fashion, i.e. every next frame is blended with the history using the same weight.
            if (widget.var("Max Frames", mMaxFrameCount, 0u))
            {
                reset();
            }
            widget.tooltip("Maximum number of frames to accumulate before triggering overflow. 0 means infinite accumulation.");

            if (widget.dropdown("Overflow Mode", mOverflowMode))
            {
                reset();
            }
            widget.tooltip(
                "What to do after maximum number of frames are accumulated:\n"
                "  Stop: Stop accumulation and retain accumulated image.\n"
                "  Reset: Reset accumulation.\n"
                "  EMA: Switch to exponential moving average accumulation.\n"
            );
        }

        const std::string text = std::string("Frames accumulated ") + std::to_string(mFrameCount);
        widget.text(text);
    }
}

void AccumulatePass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    // Reset accumulation when the scene changes.
    reset();
}

void AccumulatePass::onHotReload(HotReloadFlags reloaded)
{
    // Reset accumulation if programs changed.
    if (is_set(reloaded, HotReloadFlags::Program))
        reset();
}

void AccumulatePass::setEnabled(bool enabled)
{
    if (enabled != mEnabled)
    {
        mEnabled = enabled;
        reset();
    }
}

void AccumulatePass::reset()
{
    mFrameCount = 0;
}

void AccumulatePass::prepareAccumulation(RenderContext* pRenderContext, uint32_t width, uint32_t height)
{
    // Allocate/resize/clear buffers for intermedate data. These are different depending on accumulation mode.
    // Buffers that are not used in the current mode are released.
    auto prepareBuffer = [&](ref<Texture>& pBuf, ResourceFormat format, bool bufUsed)
    {
        if (!bufUsed)
        {
            pBuf = nullptr;
            return;
        }
        // (Re-)create buffer if needed.
        if (!pBuf || pBuf->getWidth() != width || pBuf->getHeight() != height)
        {
            pBuf = mpDevice->createTexture2D(
                width, height, format, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
            );
            FALCOR_ASSERT(pBuf);
            reset();
        }
        // Clear data if accumulation has been reset (either above or somewhere else).
        if (mFrameCount == 0)
        {
            if (getFormatType(format) == FormatType::Float)
                pRenderContext->clearUAV(pBuf->getUAV().get(), float4(0.f));
            else
                pRenderContext->clearUAV(pBuf->getUAV().get(), uint4(0));
        }
    };

    prepareBuffer(
        mpLastFrameSum, ResourceFormat::RGBA32Float, mPrecisionMode == Precision::Single || mPrecisionMode == Precision::SingleCompensated
    );
    prepareBuffer(mpLastFrameCorr, ResourceFormat::RGBA32Float, mPrecisionMode == Precision::SingleCompensated);
    prepareBuffer(mpLastFrameSumLo, ResourceFormat::RGBA32Uint, mPrecisionMode == Precision::Double);
    prepareBuffer(mpLastFrameSumHi, ResourceFormat::RGBA32Uint, mPrecisionMode == Precision::Double);
}
