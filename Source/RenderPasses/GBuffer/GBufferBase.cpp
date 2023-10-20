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
#include "GBufferBase.h"
#include "GBuffer/GBufferRaster.h"
#include "GBuffer/GBufferRT.h"
#include "VBuffer/VBufferRaster.h"
#include "VBuffer/VBufferRT.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/SampleGenerators/DxSamplePattern.h"
#include "Utils/SampleGenerators/HaltonSamplePattern.h"
#include "Utils/SampleGenerators/StratifiedSamplePattern.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, GBufferRaster>();
    registry.registerClass<RenderPass, GBufferRT>();
    registry.registerClass<RenderPass, VBufferRaster>();
    registry.registerClass<RenderPass, VBufferRT>();
}

namespace
{
// Scripting options.
const char kOutputSize[] = "outputSize";
const char kFixedOutputSize[] = "fixedOutputSize";
const char kSamplePattern[] = "samplePattern";
const char kSampleCount[] = "sampleCount";
const char kUseAlphaTest[] = "useAlphaTest";
const char kDisableAlphaTest[] = "disableAlphaTest"; ///< Deprecated for "useAlphaTest".
const char kAdjustShadingNormals[] = "adjustShadingNormals";
const char kForceCullMode[] = "forceCullMode";
const char kCullMode[] = "cull";
} // namespace

void GBufferBase::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kOutputSize)
            mOutputSizeSelection = value;
        else if (key == kFixedOutputSize)
            mFixedOutputSize = value;
        else if (key == kSamplePattern)
            mSamplePattern = value;
        else if (key == kSampleCount)
            mSampleCount = value;
        else if (key == kUseAlphaTest)
            mUseAlphaTest = value;
        else if (key == kAdjustShadingNormals)
            mAdjustShadingNormals = value;
        else if (key == kForceCullMode)
            mForceCullMode = value;
        else if (key == kCullMode)
            mCullMode = value;
        // TODO: Check for unparsed fields, including those parsed in derived classes.
    }

    // Handle deprecated "disableAlphaTest" value.
    if (props.has(kDisableAlphaTest) && !props.has(kUseAlphaTest))
        mUseAlphaTest = !props[kDisableAlphaTest];
}

Properties GBufferBase::getProperties() const
{
    Properties props;
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        props[kFixedOutputSize] = mFixedOutputSize;
    props[kSamplePattern] = mSamplePattern;
    props[kSampleCount] = mSampleCount;
    props[kUseAlphaTest] = mUseAlphaTest;
    props[kAdjustShadingNormals] = mAdjustShadingNormals;
    props[kForceCullMode] = mForceCullMode;
    props[kCullMode] = mCullMode;
    return props;
}

void GBufferBase::renderUI(Gui::Widgets& widget)
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

    // Sample pattern controls.
    bool updatePattern = widget.dropdown("Sample pattern", mSamplePattern);
    widget.tooltip(
        "Selects sample pattern for anti-aliasing over multiple frames.\n\n"
        "The camera jitter is set at the start of each frame based on the chosen pattern.\n"
        "All render passes should see the same jitter.\n"
        "'Center' disables anti-aliasing by always sampling at the center of the pixel.",
        true
    );
    if (mSamplePattern != SamplePattern::Center)
    {
        updatePattern |= widget.var("Sample count", mSampleCount, 1u);
        widget.tooltip("Number of samples in the anti-aliasing sample pattern.", true);
    }
    if (updatePattern)
    {
        updateSamplePattern();
        mOptionsChanged = true;
    }

    // Misc controls.
    mOptionsChanged |= widget.checkbox("Alpha Test", mUseAlphaTest);
    widget.tooltip("Use alpha testing on non-opaque triangles.");

    mOptionsChanged |= widget.checkbox("Adjust shading normals", mAdjustShadingNormals);
    widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.", true);

    // Cull mode controls.
    mOptionsChanged |= widget.checkbox("Force cull mode", mForceCullMode);
    widget.tooltip(
        "Enable this option to override the default cull mode.\n\n"
        "Otherwise the default for rasterization is to cull backfacing geometry, "
        "and for ray tracing to disable culling.",
        true
    );

    if (mForceCullMode)
    {
        if (auto cullMode = mCullMode; widget.dropdown("Cull mode", cullMode))
        {
            setCullMode(cullMode);
            mOptionsChanged = true;
        }
    }
}

void GBufferBase::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // Pass flag for adjust shading normals to subsequent passes via the dictionary.
    // Adjusted shading normals cannot be passed via the VBuffer, so this flag allows consuming passes to compute them when enabled.
    dict[Falcor::kRenderPassGBufferAdjustShadingNormals] = mAdjustShadingNormals;
}

void GBufferBase::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mFrameCount = 0;
    updateSamplePattern();

    if (pScene)
    {
        // Trigger graph recompilation if we need to change the V-buffer format.
        ResourceFormat format = pScene->getHitInfo().getFormat();
        if (format != mVBufferFormat)
        {
            mVBufferFormat = format;
            requestRecompile();
        }
    }
}

static ref<CPUSampleGenerator> createSamplePattern(GBufferBase::SamplePattern type, uint32_t sampleCount)
{
    switch (type)
    {
    case GBufferBase::SamplePattern::Center:
        return nullptr;
    case GBufferBase::SamplePattern::DirectX:
        return DxSamplePattern::create(sampleCount);
    case GBufferBase::SamplePattern::Halton:
        return HaltonSamplePattern::create(sampleCount);
    case GBufferBase::SamplePattern::Stratified:
        return StratifiedSamplePattern::create(sampleCount);
    default:
        FALCOR_UNREACHABLE();
        return nullptr;
    }
}

void GBufferBase::updateFrameDim(const uint2 frameDim)
{
    FALCOR_ASSERT(frameDim.x > 0 && frameDim.y > 0);
    mFrameDim = frameDim;
    mInvFrameDim = 1.f / float2(frameDim);

    // Update sample generator for camera jitter.
    if (mpScene)
        mpScene->getCamera()->setPatternGenerator(mpSampleGenerator, mInvFrameDim);
}

void GBufferBase::updateSamplePattern()
{
    mpSampleGenerator = createSamplePattern(mSamplePattern, mSampleCount);
    if (mpSampleGenerator)
        mSampleCount = mpSampleGenerator->getSampleCount();
}

ref<Texture> GBufferBase::getOutput(const RenderData& renderData, const std::string& name) const
{
    // This helper fetches the render pass output with the given name and verifies it has the correct size.
    FALCOR_ASSERT(mFrameDim.x > 0 && mFrameDim.y > 0);
    auto pTex = renderData.getTexture(name);
    if (pTex && (pTex->getWidth() != mFrameDim.x || pTex->getHeight() != mFrameDim.y))
    {
        FALCOR_THROW("GBufferBase: Pass output '{}' has mismatching size. All outputs must be of the same size.", name);
    }
    return pTex;
}
