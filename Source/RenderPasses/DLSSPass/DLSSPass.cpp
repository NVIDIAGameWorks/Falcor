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
#include "DLSSPass.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info DLSSPass::kInfo { "DLSSPass", "DL antialiasing/upscaling." };

namespace
{
    const Gui::DropdownList kProfileChoices =
    {
        { (uint32_t)DLSSPass::Profile::MaxPerf, "Max Performance" },
        { (uint32_t)DLSSPass::Profile::Balanced, "Balanced" },
        { (uint32_t)DLSSPass::Profile::MaxQuality, "Max Quality" }
    };

    const Gui::DropdownList kMotionVectorScaleChoices =
    {
        { (uint32_t)DLSSPass::MotionVectorScale::Absolute, "Absolute" },
        { (uint32_t)DLSSPass::MotionVectorScale::Relative, "Relative" },
    };

    const char kColorInput[] = "color";
    const char kDepthInput[] = "depth";
    const char kMotionVectorsInput[] = "mvec";
    const char kOutput[] = "output";

    // Scripting options.
    const char kEnabled[] = "enabled";
    const char kOutputSize[] = "outputSize";
    const char kProfile[] = "profile";
    const char kMotionVectorScale[] = "motionVectorScale";
    const char kIsHDR[] = "isHDR";
    const char kSharpness[] = "sharpness";
    const char kExposure[] = "exposure";
};

static void registerDLSSPass(pybind11::module& m)
{
    pybind11::class_<DLSSPass, RenderPass, DLSSPass::SharedPtr> pass(m, "DLSSPass");

    pybind11::enum_<DLSSPass::Profile> profile(m, "DLSSProfile");
    profile.value("MaxPerf", DLSSPass::Profile::MaxPerf);
    profile.value("Balanced", DLSSPass::Profile::Balanced);
    profile.value("MaxQuality", DLSSPass::Profile::MaxQuality);

    pybind11::enum_<DLSSPass::MotionVectorScale> motionVectorScale(m, "DLSSMotionVectorScale");
    motionVectorScale.value("Absolute", DLSSPass::MotionVectorScale::Absolute);
    motionVectorScale.value("Relative", DLSSPass::MotionVectorScale::Relative);
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(DLSSPass::kInfo, DLSSPass::create);
    ScriptBindings::registerBinding(registerDLSSPass);
}

DLSSPass::SharedPtr DLSSPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new DLSSPass(dict));
}

DLSSPass::DLSSPass(const Dictionary& dict)
    : RenderPass(kInfo)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kEnabled) mEnabled = value;
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kProfile) mProfile = value;
        else if (key == kMotionVectorScale) mMotionVectorScale = value;
        else if (key == kIsHDR) mIsHDR = value;
        else if (key == kSharpness) mSharpness = value;
        else if (key == kExposure) { mExposure = value; mExposureUpdated = true; }
        else logWarning("Unknown field '{}' in a DLSSPass dictionary.", key);
    }

    mpExposure = Texture::create2D(1, 1, ResourceFormat::R32Float, 1, 1);
}

Dictionary DLSSPass::getScriptingDictionary()
{
    Dictionary d;
    d[kEnabled] = mEnabled;
    d[kOutputSize] = mOutputSizeSelection;
    d[kProfile] = mProfile;
    d[kMotionVectorScale] = mMotionVectorScale;
    d[kIsHDR] = mIsHDR;
    d[kSharpness] = mSharpness;
    d[kExposure] = mExposure;
    return d;
}

RenderPassReflection DLSSPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mDLSSOutputSize, compileData.defaultTexDims);

    r.addInput(kColorInput, "Color input").bindFlags(ResourceBindFlags::ShaderResource);
    r.addInput(kDepthInput, "Depth input").bindFlags(ResourceBindFlags::ShaderResource);
    r.addInput(kMotionVectorsInput, "Motion vectors input").bindFlags(ResourceBindFlags::ShaderResource);
    r.addOutput(kOutput, "Color output").format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget).texture2D(sz.x, sz.y);

    return r;
}

void DLSSPass::setScene(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene)
{
    mpScene = pScene;
}

void DLSSPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE("DLSSPass");

    executeInternal(pRenderContext, renderData);
}

void DLSSPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enable", mEnabled);

    // Controls for output size.
    // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
    if (widget.dropdown("Output size", RenderPassHelpers::kIOSizeList, (uint32_t&)mOutputSizeSelection)) requestRecompile();
    widget.tooltip("Specifies the pass output size.\n"
        "'Default' means that the output is sized based on requirements of connected passes.\n"
        "'Fixed' means the output is fixed to the optimal size determined by DLSS.\n"
        "If the output is of a different size than the DLSS output resolution, the image will be rescaled bilinearly.", true);

    if (mEnabled)
    {
        mRecreate |= widget.dropdown("Profile", kProfileChoices, reinterpret_cast<uint32_t&>(mProfile));

        widget.dropdown("Motion vector scale", kMotionVectorScaleChoices, reinterpret_cast<uint32_t&>(mMotionVectorScale));
        widget.tooltip(
            "Absolute: Motion vectors are provided in absolute screen space length (pixels)\n"
            "Relative: Motion vectors are provided in relative screen space length (pixels divided by screen width/height).");

        mRecreate |= widget.checkbox("HDR", mIsHDR);
        widget.tooltip("Enable if input color is HDR.");

        widget.slider("Sharpness", mSharpness, -1.f, 1.f);
        widget.tooltip("Sharpening value between 0.0 and 1.0.");

        if (widget.var("Exposure", mExposure, -10.f, 10.f, 0.01f)) mExposureUpdated = true;

        widget.text("Input resolution: " + std::to_string(mInputSize.x) + "x" + std::to_string(mInputSize.y));
        widget.text("DLSS output resolution: " + std::to_string(mDLSSOutputSize.x) + "x" + std::to_string(mDLSSOutputSize.y));
        widget.text("Pass output resolution: " + std::to_string(mPassOutputSize.x) + "x" + std::to_string(mPassOutputSize.y));
    }
}

void DLSSPass::initializeDLSS(RenderContext* pRenderContext)
{
    if (!mpNGXWrapper) mpNGXWrapper.reset(new NGXWrapper(gpDevice.get(), getExecutableDirectory().c_str()));

    Texture* target = nullptr;  // Not needed for D3D12 implementation
    bool depthInverted = false;
    NVSDK_NGX_PerfQuality_Value perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
    switch (mProfile)
    {
        case Profile::MaxPerf: perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case Profile::Balanced: perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case Profile::MaxQuality: perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
    }

    auto optimalSettings = mpNGXWrapper->queryOptimalSettings(mInputSize, perfQuality);

    mDLSSOutputSize = uint2(float2(mInputSize) * float2(mInputSize) / float2(optimalSettings.optimalRenderSize));
    mpOutput = Texture::create2D(mDLSSOutputSize.x, mDLSSOutputSize.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    mpNGXWrapper->releaseDLSS();
    mpNGXWrapper->initializeDLSS(pRenderContext, mInputSize, mDLSSOutputSize, target, mIsHDR, depthInverted, perfQuality);
}

void DLSSPass::executeInternal(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Determine pass I/O sizes based on bound textures.
    const auto& pOutput = renderData.getTexture(kOutput);
    const auto& pColor = renderData.getTexture(kColorInput);
    FALCOR_ASSERT(pColor && pOutput);
    mPassOutputSize = { pOutput->getWidth(), pOutput->getHeight() };
    const uint2 inputSize = { pColor->getWidth(), pColor->getHeight() };

    if (!mEnabled || !mpScene)
    {
        pRenderContext->blit(pColor->getSRV(), pOutput->getRTV());
        return;
    }

    if (mExposureUpdated)
    {
        float exposure = pow(2.f, mExposure);
        pRenderContext->updateTextureData(mpExposure.get(), &exposure);
        mExposureUpdated = false;
    }

    if (mRecreate || inputSize != mInputSize)
    {
        mRecreate = false;
        mInputSize = inputSize;

        initializeDLSS(pRenderContext);

        // If pass output is configured to be fixed to DLSS output, but the sizes don't match,
        // we'll trigger a graph recompile to update the pass I/O size requirements.
        // This causes a one frame delay, but unfortunately we don't know the size until after initializeDLSS().
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed && mPassOutputSize != mDLSSOutputSize) requestRecompile();
    }

    {
        // Fetch inputs and verify their dimensions.
        auto getInput = [=](const std::string& name)
        {
            const auto& tex = renderData.getTexture(name);
            if (!tex)
            {
                throw RuntimeError("DLSSPass: Missing input '{}'", name);
            }
            if (tex->getWidth() != mInputSize.x || tex->getHeight() != mInputSize.y)
            {
                throw RuntimeError("DLSSPass: Input '{}' has mismatching size. All inputs must be of the same size.", name);
            }
            return tex;
        };

        const auto& color = getInput(kColorInput);
        const auto& depth = getInput(kDepthInput);
        const auto& motionVectors = getInput(kMotionVectorsInput);

        // Determine if we can write directly to the render pass output.
        // Otherwise we'll output to an internal buffer and blit to the pass output.
        FALCOR_ASSERT(mpOutput->getWidth() == mDLSSOutputSize.x && mpOutput->getHeight() == mDLSSOutputSize.y);
        bool useInternalBuffer = (mDLSSOutputSize != mPassOutputSize || pOutput->getFormat() != mpOutput->getFormat());

        const auto& output = useInternalBuffer ? mpOutput : pOutput;

        // In DLSS X-jitter should go left-to-right, Y-jitter should go top-to-bottom.
        // Falcor is using glm::perspective() that gives coordinate system with
        // X from -1 to 1, left-to-right, and Y from -1 to 1, bottom-to-top.
        // Therefore, we need to flip the Y-jitter only.
        const auto& camera = mpScene->getCamera();
        float2 jitterOffset = float2(camera->getJitterX(), -camera->getJitterY()) * float2(inputSize);
        float2 motionVectorScale = float2(1.f, 1.f);
        if (mMotionVectorScale == MotionVectorScale::Relative) motionVectorScale = inputSize;

        mpNGXWrapper->evaluateDLSS(pRenderContext, color.get(), output.get(), motionVectors.get(), depth.get(), mpExposure.get(), false, mSharpness, jitterOffset, motionVectorScale);

        // Resample the upscaled result from DLSS to the pass output if needed.
        if (useInternalBuffer)
        {
            pRenderContext->blit(mpOutput->getSRV(), pOutput->getRTV());
        }
    }
}
