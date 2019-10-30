/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "GBuffer.h"
#include "GBufferRaster.h"
#include "GBufferRT.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

static void regGBufferPass(Falcor::ScriptBindings::Module& m)
{
    auto lodeModeBinding = m.enum_<GBufferRT::LODMode>("LODMode");
    lodeModeBinding.regEnumVal(GBufferRT::LODMode::UseMip0);
    lodeModeBinding.regEnumVal(GBufferRT::LODMode::RayDifferentials);
    // lodeModeBinding.regEnumVal(GBufferRT::LODMode::TexLODCone) not implemented

    auto samplePatternEnum = m.enum_<GBuffer::SamplePattern>("SamplePattern");
    samplePatternEnum.regEnumVal(GBuffer::SamplePattern::Center);
    samplePatternEnum.regEnumVal(GBuffer::SamplePattern::DirectX);
    samplePatternEnum.regEnumVal(GBuffer::SamplePattern::Halton);
    samplePatternEnum.regEnumVal(GBuffer::SamplePattern::Stratified);
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("GBufferRaster", "Raster-Based GBuffer Creation", GBufferRaster::create);
    lib.registerClass("GBufferRT", "Ray Tracing-Based GBuffer Creation", GBufferRT::create);
    Falcor::ScriptBindings::registerBinding(regGBufferPass);
}

// Note that channel order should correspond to SV_TARGET index order used in
// GBufferRaster's primary fragment shader.
const ChannelList GBuffer::kGBufferChannels =
{
    { "posW",           "gPosW",            "world space position",         true /* optional */, ResourceFormat::RGBA32Float },
    { "normW",          "gNormW",           "world space normal",           true /* optional */, ResourceFormat::RGBA32Float },
    { "bitangentW",     "gBitangentW",      "world space bitangent",        true /* optional */, ResourceFormat::RGBA32Float },
    { "texC",           "gTexC",            "texture coordinates",          true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseOpacity", "gDiffuseOpacity",  "diffuse color and opacity",    true /* optional */, ResourceFormat::RGBA32Float },
    { "specRough",      "gSpecRough",       "specular color and roughness", true /* optional */, ResourceFormat::RGBA32Float },
    { "emissive",       "gEmissive",        "emissive color",               true /* optional */, ResourceFormat::RGBA32Float },
    { "matlExtra",      "gMatlExtra",       "additional material data",     true /* optional */, ResourceFormat::RGBA32Float },
};

namespace
{
    // Serialized parameters
    const char kForceCullMode[] = "forceCullMode";
    const char kCullMode[] = "cull";
    const char kSamplePattern[] = "samplePattern";
    const char kSampleCount[] = "sampleCount";

    // UI variables
    const Gui::DropdownList kCullModeList =
    {
        { (uint32_t)RasterizerState::CullMode::None, "None" },
        { (uint32_t)RasterizerState::CullMode::Back, "Back" },
        { (uint32_t)RasterizerState::CullMode::Front, "Front" },
    };

    const Gui::DropdownList kSamplePatternList =
    {
        { (uint32_t)GBuffer::SamplePattern::Center, "Center" },
        { (uint32_t)GBuffer::SamplePattern::DirectX, "DirectX" },
        { (uint32_t)GBuffer::SamplePattern::Halton, "Halton" },
        { (uint32_t)GBuffer::SamplePattern::Stratified, "Stratified" },
    };
}

GBuffer::GBuffer() : mGBufferParams{}
{
}

bool GBuffer::parseDictionary(const Dictionary& dict)
{
    for (const auto& v : dict)
    {
        if (v.key() == kForceCullMode) mForceCullMode = v.val();
        else if (v.key() == kCullMode) setCullMode((RasterizerState::CullMode)v.val());
        else if (v.key() == kSamplePattern) mSamplePattern = (SamplePattern)v.val();
        else if (v.key() == kSampleCount) mSampleCount = v.val();
        else
        {
            logWarning("Unknown field `" + v.key() + "` in a GBuffer dictionary");
        }
    }
    return true;
}

Dictionary GBuffer::getScriptingDictionary()
{
    Dictionary dict;
    dict[kForceCullMode] = mForceCullMode;
    dict[kCullMode] = mCullMode;
    dict[kSamplePattern] = mSamplePattern;
    dict[kSampleCount] = mSampleCount;
    return dict;
}

void GBuffer::renderUI(Gui::Widgets& widget)
{
    // Cull mode controls.
    mOptionsChanged |= widget.checkbox("Force cull mode", mForceCullMode);
    widget.tooltip("Enable this option to force the same cull mode for all geometry.\n\n"
        "Otherwise the default for rasterization is to set the cull mode automatically based on triangle winding, and for ray tracing to disable culling.", true);

    if (mForceCullMode)
    {
        uint32_t cullMode = (uint32_t)mCullMode;
        if (widget.dropdown("Cull mode", kCullModeList, cullMode))
        {
            setCullMode((RasterizerState::CullMode)cullMode);
            mOptionsChanged = true;
        }
    }

    // Sample pattern controls.
    bool updatePattern = widget.dropdown("Sample pattern", kSamplePatternList, (uint32_t&)mSamplePattern);
    widget.tooltip("Selects sample pattern for anti-aliasing over multiple frames.\n\n"
        "The camera jitter is set at the start of each frame based on the chosen pattern. All render passes should see the same jitter.\n"
        "'Center' disables anti-aliasing by always sampling at the center of the pixel.", true);
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
}

void GBuffer::compile(RenderContext* pContext, const CompileData& compileData)
{
    mGBufferParams.frameSize = vec2(compileData.defaultTexDims);
    mGBufferParams.invFrameSize = 1.f / mGBufferParams.frameSize;

    if (mpScene)
    {
        auto pCamera = mpScene->getCamera();
        pCamera->setPatternGenerator(pCamera->getPatternGenerator(), mGBufferParams.invFrameSize);
    }
    return;
}

void GBuffer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mGBufferParams.frameCount = 0;
    updateSamplePattern();
}

void GBuffer::updateSamplePattern()
{
    if (mpScene)
    {
        auto pGen = createSamplePattern(mSamplePattern, mSampleCount);
        if (pGen) mSampleCount = pGen->getSampleCount();
        mpScene->getCamera()->setPatternGenerator(pGen, mGBufferParams.invFrameSize);
    }
}

CPUSampleGenerator::SharedPtr GBuffer::createSamplePattern(SamplePattern type, uint32_t sampleCount)
{
    switch (type)
    {
    case SamplePattern::Center:
        return nullptr;
    case SamplePattern::DirectX:
        return DxSamplePattern::create(sampleCount);
    case SamplePattern::Halton:
        return HaltonSamplePattern::create(sampleCount);
    case SamplePattern::Stratified:
        return StratifiedSamplePattern::create(sampleCount);
    default:
        should_not_get_here();
        return nullptr;
    }
}
