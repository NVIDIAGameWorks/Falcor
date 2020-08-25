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
#include "GBufferBase.h"
#include "GBuffer/GBufferRaster.h"
#include "GBuffer/GBufferRT.h"
#include "VBuffer/VBufferRaster.h"
#include "VBuffer/VBufferRT.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("GBufferRaster", GBufferRaster::kDesc, GBufferRaster::create);
    lib.registerClass("GBufferRT", GBufferRT::kDesc, GBufferRT::create);
    lib.registerClass("VBufferRaster", VBufferRaster::kDesc, VBufferRaster::create);
    lib.registerClass("VBufferRT", VBufferRT::kDesc, VBufferRT::create);

    Falcor::ScriptBindings::registerBinding(GBufferBase::registerBindings);
    Falcor::ScriptBindings::registerBinding(GBufferRT::registerBindings);
}

void GBufferBase::registerBindings(pybind11::module& m)
{
    pybind11::enum_<GBufferBase::SamplePattern> samplePattern(m, "SamplePattern");
    samplePattern.value("Center", GBufferBase::SamplePattern::Center);
    samplePattern.value("DirectX", GBufferBase::SamplePattern::DirectX);
    samplePattern.value("Halton", GBufferBase::SamplePattern::Halton);
    samplePattern.value("Stratified", GBufferBase::SamplePattern::Stratified);
}

namespace
{
    // Scripting options.
    const char kSamplePattern[] = "samplePattern";
    const char kSampleCount[] = "sampleCount";
    const char kDisableAlphaTest[] = "disableAlphaTest";

    // UI variables.
    const Gui::DropdownList kSamplePatternList =
    {
        { (uint32_t)GBufferBase::SamplePattern::Center, "Center" },
        { (uint32_t)GBufferBase::SamplePattern::DirectX, "DirectX" },
        { (uint32_t)GBufferBase::SamplePattern::Halton, "Halton" },
        { (uint32_t)GBufferBase::SamplePattern::Stratified, "Stratified" },
    };
}

void GBufferBase::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kSamplePattern) mSamplePattern = value;
        else if (key == kSampleCount) mSampleCount = value;
        else if (key == kDisableAlphaTest) mDisableAlphaTest = value;
        // TODO: Check for unparsed fields, including those parsed in derived classes.
    }
}

Dictionary GBufferBase::getScriptingDictionary()
{
    Dictionary dict;
    dict[kSamplePattern] = mSamplePattern;
    dict[kSampleCount] = mSampleCount;
    dict[kDisableAlphaTest] = mDisableAlphaTest;
    return dict;
}

void GBufferBase::renderUI(Gui::Widgets& widget)
{
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

    mOptionsChanged |=  widget.checkbox("Disable Alpha Test", mDisableAlphaTest);
}

void GBufferBase::compile(RenderContext* pContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
    mInvFrameDim = 1.f / float2(mFrameDim);

    if (mpScene)
    {
        auto pCamera = mpScene->getCamera();
        pCamera->setPatternGenerator(pCamera->getPatternGenerator(), mInvFrameDim);
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

    // Setup camera with sample generator.
    if (mpScene) mpScene->getCamera()->setPatternGenerator(mpSampleGenerator, mInvFrameDim);
}

void GBufferBase::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    updateSamplePattern();
}

static CPUSampleGenerator::SharedPtr createSamplePattern(GBufferBase::SamplePattern type, uint32_t sampleCount)
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
        should_not_get_here();
        return nullptr;
    }
}

void GBufferBase::updateSamplePattern()
{
    mpSampleGenerator = createSamplePattern(mSamplePattern, mSampleCount);
    if (mpSampleGenerator) mSampleCount = mpSampleGenerator->getSampleCount();
}
