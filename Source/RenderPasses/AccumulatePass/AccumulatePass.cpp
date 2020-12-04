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
#include "AccumulatePass.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

static void regAccumulatePass(pybind11::module& m)
{
    pybind11::class_<AccumulatePass, RenderPass, AccumulatePass::SharedPtr> pass(m, "AccumulatePass");
    pass.def("reset", &AccumulatePass::reset);

    pybind11::enum_<AccumulatePass::Precision> precision(m, "AccumulatePrecision");
    precision.value("Double", AccumulatePass::Precision::Double);
    precision.value("Single", AccumulatePass::Precision::Single);
    precision.value("SingleCompensated", AccumulatePass::Precision::SingleCompensated);
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("AccumulatePass", "Temporal accumulation", AccumulatePass::create);
    ScriptBindings::registerBinding(regAccumulatePass);
}

namespace
{
    const char kShaderFile[] = "RenderPasses/AccumulatePass/Accumulate.cs.slang";

    const char kInputChannel[] = "input";
    const char kOutputChannel[] = "output";

    // Serialized parameters
    const char kEnableAccumulation[] = "enableAccumulation";
    const char kAutoReset[] = "autoReset";
    const char kPrecisionMode[] = "precisionMode";
    const char kSubFrameCount[] = "subFrameCount";

    const Gui::DropdownList kModeSelectorList =
    {
        { (uint32_t)AccumulatePass::Precision::Double, "Double precision" },
        { (uint32_t)AccumulatePass::Precision::Single, "Single precision" },
        { (uint32_t)AccumulatePass::Precision::SingleCompensated, "Single precision (compensated)" },
    };
}

AccumulatePass::SharedPtr AccumulatePass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new AccumulatePass(dict));
}

AccumulatePass::AccumulatePass(const Dictionary& dict)
{
    // Deserialize pass from dictionary.
    for (const auto& [key, value] : dict)
    {
        if (key == kEnableAccumulation) mEnableAccumulation = value;
        else if (key == kAutoReset) mAutoReset = value;
        else if (key == kPrecisionMode) mPrecisionMode = value;
        else if (key == kSubFrameCount) mSubFrameCount = value;
        else logWarning("Unknown field '" + key + "' in AccumulatePass dictionary");
    }

    // Create accumulation programs.
    // Note only compensated summation needs precise floating-point mode.
    mpProgram[Precision::Double] = ComputeProgram::createFromFile(kShaderFile, "accumulateDouble", Program::DefineList(), Shader::CompilerFlags::TreatWarningsAsErrors);
    mpProgram[Precision::Single] = ComputeProgram::createFromFile(kShaderFile, "accumulateSingle", Program::DefineList(), Shader::CompilerFlags::TreatWarningsAsErrors);
    mpProgram[Precision::SingleCompensated] = ComputeProgram::createFromFile(kShaderFile, "accumulateSingleCompensated", Program::DefineList(), Shader::CompilerFlags::FloatingPointModePrecise | Shader::CompilerFlags::TreatWarningsAsErrors);
    mpVars = ComputeVars::create(mpProgram[Precision::Single]->getReflector());

    mpState = ComputeState::create();
}

Dictionary AccumulatePass::getScriptingDictionary()
{
    Dictionary dict;
    dict[kEnableAccumulation] = mEnableAccumulation;
    dict[kAutoReset] = mAutoReset;
    dict[kPrecisionMode] = mPrecisionMode;
    dict[kSubFrameCount] = mSubFrameCount;
    return dict;
}

RenderPassReflection AccumulatePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputChannel, "Input data to be temporally accumulated").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutputChannel, "Output data that is temporally accumulated").bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource).format(ResourceFormat::RGBA32Float);
    return reflector;
}

void AccumulatePass::compile(RenderContext* pContext, const CompileData& compileData)
{
    // Reset accumulation when resolution changes.
    if (compileData.defaultTexDims != mFrameDim)
    {
        mFrameCount = 0;
        mFrameDim = compileData.defaultTexDims;
    }
}

void AccumulatePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mAutoReset)
    {
        if (mSubFrameCount > 0) // Option to accumulate N frames. Works also for motion blur. Overrides logic for automatic reset on scene changes.
        {
            if (mFrameCount == mSubFrameCount)
            {
                mFrameCount = 0;
            }
        }
        else
        {
            // Query refresh flags passed down from the application and other passes.
            auto& dict = renderData.getDictionary();
            auto refreshFlags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);

            // If any refresh flag is set, we reset frame accumulation.
            if (refreshFlags != RenderPassRefreshFlags::None) mFrameCount = 0;

            // Reset accumulation upon all scene changes, except camera jitter and history changes.
            // TODO: Add UI options to select which changes should trigger reset
            if (mpScene)
            {
                auto sceneUpdates = mpScene->getUpdates();
                if ((sceneUpdates & ~Scene::UpdateFlags::CameraPropertiesChanged) != Scene::UpdateFlags::None)
                {
                    mFrameCount = 0;
                }
                if (is_set(sceneUpdates, Scene::UpdateFlags::CameraPropertiesChanged))
                {
                    auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
                    auto cameraChanges = mpScene->getCamera()->getChanges();
                    if ((cameraChanges & ~excluded) != Camera::Changes::None) mFrameCount = 0;
                }
            }
        }
    }

    // Grab our input/output buffers.
    Texture::SharedPtr pSrc = renderData[kInputChannel]->asTexture();
    Texture::SharedPtr pDst = renderData[kOutputChannel]->asTexture();

    assert(pSrc && pDst);
    assert(pSrc->getWidth() == pDst->getWidth() && pSrc->getHeight() == pDst->getHeight());
    const uint2 resolution = uint2(pSrc->getWidth(), pSrc->getHeight());

    // If accumulation is disabled, just blit the source to the destination and return.
    if (!mEnableAccumulation)
    {
        // Only blit mip 0 and array slice 0, because that's what the accumulation uses otherwise otherwise.
        pRenderContext->blit(pSrc->getSRV(0, 1, 0, 1), pDst->getRTV(0, 0, 1));
        return;
    }

    // Setup accumulation.
    prepareAccumulation(pRenderContext, resolution.x, resolution.y);

    // Set shader parameters.
    mpVars["PerFrameCB"]["gResolution"] = resolution;
    mpVars["PerFrameCB"]["gAccumCount"] = mFrameCount++;
    mpVars["gCurFrame"] = pSrc;
    mpVars["gOutputFrame"] = pDst;

    // Bind accumulation buffers. Some of these may be nullptr's.
    mpVars["gLastFrameSum"] = mpLastFrameSum;
    mpVars["gLastFrameCorr"] = mpLastFrameCorr;
    mpVars["gLastFrameSumLo"] = mpLastFrameSumLo;
    mpVars["gLastFrameSumHi"] = mpLastFrameSumHi;

    // Run the accumulation program.
    auto pProgram = mpProgram[mPrecisionMode];
    assert(pProgram);
    uint3 numGroups = div_round_up(uint3(resolution.x, resolution.y, 1u), pProgram->getReflector()->getThreadGroupSize());
    mpState->setProgram(pProgram);
    pRenderContext->dispatch(mpState.get(), mpVars.get(), numGroups);
}

void AccumulatePass::renderUI(Gui::Widgets& widget)
{
    if (widget.checkbox("Accumulate temporally", mEnableAccumulation))
    {
        // Reset accumulation when it is toggled.
        mFrameCount = 0;
    }

    if (mEnableAccumulation)
    {
        if (widget.dropdown("Mode", kModeSelectorList, (uint32_t&)mPrecisionMode))
        {
            // Reset accumulation when mode changes.
            mFrameCount = 0;
        }

        const std::string text = std::string("Frames accumulated ") + std::to_string(mFrameCount);
        widget.text(text);
    }
}

void AccumulatePass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    // Reset accumulation when the scene changes.
    mFrameCount = 0;
    mpScene = pScene;
}

void AccumulatePass::onHotReload(HotReloadFlags reloaded)
{
    // Reset accumulation if programs changed.
    if (is_set(reloaded, HotReloadFlags::Program)) mFrameCount = 0;
}

void AccumulatePass::prepareAccumulation(RenderContext* pRenderContext, uint32_t width, uint32_t height)
{
    // Allocate/resize/clear buffers for intermedate data. These are different depending on accumulation mode.
    // Buffers that are not used in the current mode are released.
    auto prepareBuffer = [&](Texture::SharedPtr& pBuf, ResourceFormat format, bool bufUsed)
    {
        if (!bufUsed)
        {
            pBuf = nullptr;
            return;
        }
        // (Re-)create buffer if needed.
        if (!pBuf || pBuf->getWidth() != width || pBuf->getHeight() != height)
        {
            pBuf = Texture::create2D(width, height, format, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
            assert(pBuf);
            mFrameCount = 0;
        }
        // Clear data if accumulation has been reset (either above or somewhere else).
        if (mFrameCount == 0)
        {
            if (getFormatType(format) == FormatType::Float) pRenderContext->clearUAV(pBuf->getUAV().get(), float4(0.f));
            else pRenderContext->clearUAV(pBuf->getUAV().get(), uint4(0));
        }
    };

    prepareBuffer(mpLastFrameSum, ResourceFormat::RGBA32Float, mPrecisionMode == Precision::Single || mPrecisionMode == Precision::SingleCompensated);
    prepareBuffer(mpLastFrameCorr, ResourceFormat::RGBA32Float, mPrecisionMode == Precision::SingleCompensated);
    prepareBuffer(mpLastFrameSumLo, ResourceFormat::RGBA32Uint, mPrecisionMode == Precision::Double);
    prepareBuffer(mpLastFrameSumHi, ResourceFormat::RGBA32Uint, mPrecisionMode == Precision::Double);
}
