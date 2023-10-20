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
#include "CrossFade.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kShaderFile("RenderPasses/Utils/CrossFade/CrossFade.cs.slang");

const std::string kInputA = "A";
const std::string kInputB = "B";
const std::string kOutput = "out";

const std::string kOutputFormat = "outputFormat";
const std::string kEnableAutoFade = "enableAutoFade";
const std::string kWaitFrameCount = "waitFrameCount";
const std::string kFadeFrameCount = "fadeFrameCount";
const std::string kFadeFactor = "fadeFactor";
} // namespace

CrossFade::CrossFade(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // Parse dictionary.
    for (const auto& [key, value] : props)
    {
        if (key == kOutputFormat)
            mOutputFormat = value;
        else if (key == kEnableAutoFade)
            mEnableAutoFade = value;
        else if (key == kWaitFrameCount)
            mWaitFrameCount = value;
        else if (key == kFadeFrameCount)
            mFadeFrameCount = value;
        else if (key == kFadeFactor)
            mFadeFactor = value;
        else
            logWarning("Unknown property '{}' in CrossFade pass properties.", key);
    }

    // Create resources.
    mpFadePass = ComputePass::create(mpDevice, kShaderFile, "main");
}

Properties CrossFade::getProperties() const
{
    Properties props;
    if (mOutputFormat != ResourceFormat::Unknown)
        props[kOutputFormat] = mOutputFormat;
    props[kEnableAutoFade] = mEnableAutoFade;
    props[kWaitFrameCount] = mWaitFrameCount;
    props[kFadeFrameCount] = mFadeFrameCount;
    props[kFadeFactor] = mFadeFactor;
    return props;
}

RenderPassReflection CrossFade::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputA, "Input A").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kInputB, "Input B").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(kOutput, "Output").bindFlags(ResourceBindFlags::UnorderedAccess).format(mOutputFormat);
    return reflector;
}

void CrossFade::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

void CrossFade::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void CrossFade::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    bool shouldReset = false;

    // Query refresh flags passed down from the application and other passes.
    auto& dict = renderData.getDictionary();
    auto refreshFlags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);

    // If any refresh flag is set, we reset frame accumulation.
    if (refreshFlags != RenderPassRefreshFlags::None)
        shouldReset = true;

    // Reset accumulation upon all scene changes, except camera jitter and history changes.
    // TODO: Add UI options to select which changes should trigger reset
    if (mpScene)
    {
        auto sceneUpdates = mpScene->getUpdates();
        if ((sceneUpdates & ~Scene::UpdateFlags::CameraPropertiesChanged) != Scene::UpdateFlags::None)
        {
            shouldReset = true;
        }
        if (is_set(sceneUpdates, Scene::UpdateFlags::CameraPropertiesChanged))
        {
            auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
            auto cameraChanges = mpScene->getCamera()->getChanges();
            if ((cameraChanges & ~excluded) != Camera::Changes::None)
                shouldReset = true;
        }
        if (is_set(sceneUpdates, Scene::UpdateFlags::SDFGeometryChanged))
        {
            shouldReset = true;
        }
    }

    if (shouldReset)
    {
        mMixFrame = 0;
    }
    else
    {
        mMixFrame++;
    }

    float mix = mEnableAutoFade ? math::clamp((float(mMixFrame) - mWaitFrameCount) / mFadeFrameCount, 0.f, 1.f)
                                : math::clamp(mFadeFactor, 0.f, 1.f);

    mScaleA = 1.f - mix;
    mScaleB = mix;

    // Prepare program.
    const auto& pOutput = renderData.getTexture(kOutput);
    FALCOR_ASSERT(pOutput);
    mOutputFormat = pOutput->getFormat();
    FALCOR_CHECK(!isIntegerFormat(mOutputFormat), "Output cannot be an integer format.");

    // Bind resources.
    auto var = mpFadePass->getRootVar();
    var["CB"]["frameDim"] = mFrameDim;
    var["CB"]["scaleA"] = mScaleA;
    var["CB"]["scaleB"] = mScaleB;

    var["A"] = renderData.getTexture(kInputA); // Can be nullptr
    var["B"] = renderData.getTexture(kInputB); // Can be nullptr
    var["output"] = pOutput;
    mpFadePass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void CrossFade::renderUI(Gui::Widgets& widget)
{
    widget.text("This pass fades between inputs A and B");
    widget.checkbox("Enable Auto Fade", mEnableAutoFade);
    if (mEnableAutoFade)
    {
        widget.var("Wait Frame Count", mWaitFrameCount);
        widget.var("Fade Frame Count", mFadeFrameCount);
    }
    else
    {
        widget.var("Fade Factor", mFadeFactor, 0.f, 1.f);
    }
}
