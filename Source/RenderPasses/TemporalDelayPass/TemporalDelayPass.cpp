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
#include "TemporalDelayPass.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info TemporalDelayPass::kInfo { "TemporalDelayPass", "Delays frame rendering by a specified amount of frames." };

extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

void regTemporalDelayPass(pybind11::module& m)
{
    pybind11::class_<TemporalDelayPass, RenderPass, TemporalDelayPass::SharedPtr> pass(m, "TemporalDelayPass");
    pass.def_property("delay", &TemporalDelayPass::getDelay, &TemporalDelayPass::setDelay);
}

extern "C" FALCOR_API_EXPORT void getPasses(RenderPassLibrary& lib)
{
    lib.registerPass(TemporalDelayPass::kInfo, TemporalDelayPass::create);
    ScriptBindings::registerBinding(regTemporalDelayPass);
}

namespace
{
    const std::string kSrc = "src";
    const std::string kMaxDelay = "maxDelay";
    const std::string kDelay = "delay";
}

TemporalDelayPass::TemporalDelayPass() : RenderPass(kInfo) {}

TemporalDelayPass::SharedPtr TemporalDelayPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new TemporalDelayPass());
    for (const auto& [key, value] : dict)
    {
        if (key == kDelay) pPass->mDelay = value;
        else logWarning("Unknown field '{}' in a TemporalDelayPass dictionary.", key);
    }
    return pPass;
}

RenderPassReflection TemporalDelayPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    mReady = false;
    if (compileData.connectedResources.getFieldCount() > 0)
    {
        const RenderPassReflection::Field* edge = compileData.connectedResources.getField(kSrc);
        RenderPassReflection::Field::Type srcType = edge->getType();
        ResourceFormat srcFormat = edge->getFormat();
        uint32_t srcWidth = edge->getWidth();
        uint32_t srcHeight = edge->getHeight();
        uint32_t srcDepth = edge->getDepth();
        uint32_t srcSampleCount = edge->getSampleCount();
        uint32_t srcMipCount = edge->getMipCount();
        uint32_t srcArraySize = edge->getArraySize();

        auto formatField = [=](RenderPassReflection::Field& f) {
            return f.format(srcFormat).resourceType(srcType, srcWidth, srcHeight, srcDepth, srcSampleCount, srcMipCount, srcArraySize);
        };

        formatField(r.addInput(kSrc, "Current frame"));
        formatField(r.addOutput(kMaxDelay, std::to_string(mDelay) + " frame(s) delayed"));
        if (mDelay > 0)
        {
            for (uint32_t i = mDelay - 1; i > 0; --i) formatField(r.addOutput(kMaxDelay + "-" + std::to_string(i), std::to_string(mDelay - i) + " frame(s) delayed"));
            formatField(r.addInternal(kMaxDelay + "-" + std::to_string(mDelay), "Internal copy of the current frame"));
        }
        mReady = true;
    }
    else
    {
        r.addInput(kSrc, "Current frame");
        r.addOutput(kMaxDelay, std::to_string(mDelay) + " frame(s) delayed");
        if (mDelay > 0)
        {
            for (uint32_t i = mDelay - 1; i > 0; --i) r.addOutput(kMaxDelay + "-" + std::to_string(i), std::to_string(mDelay - i) + " frame(s) delayed");
            r.addInternal(kMaxDelay + "-" + std::to_string(mDelay), "Internal copy of the current frame");
        }
    }
    return r;
}

void TemporalDelayPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mReady) throw RuntimeError("TemporalDelayPass: Missing incoming reflection information");
}

void TemporalDelayPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mDelay == 0)
    {
        pRenderContext->copyResource(renderData[kMaxDelay].get(), renderData[kSrc].get());
        return;
    }
    for (uint32_t copyDst = 0; copyDst <= mDelay; ++copyDst)
    {
        uint32_t copySrc = copyDst + 1;
        if (copyDst == 0) pRenderContext->copyResource(renderData[kMaxDelay].get(), renderData[kMaxDelay + "-" + std::to_string(copySrc)].get());
        else if (copyDst == mDelay) pRenderContext->copyResource(renderData[kMaxDelay + "-" + std::to_string(copyDst)].get(), renderData[kSrc].get());
        else pRenderContext->copyResource(renderData[kMaxDelay + "-" + std::to_string(copyDst)].get(), renderData[kMaxDelay + "-" + std::to_string(copySrc)].get());
    }
}

Dictionary TemporalDelayPass::getScriptingDictionary()
{
    Dictionary d;
    d[kDelay] = mDelay;
    return d;
}

void TemporalDelayPass::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Delay", mDelay)) setDelay(mDelay);
}

TemporalDelayPass& TemporalDelayPass::setDelay(uint32_t delay)
{
    mDelay = delay;
    requestRecompile();
    return *this;
}
