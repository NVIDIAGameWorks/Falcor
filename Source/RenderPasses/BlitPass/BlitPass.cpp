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
#include "BlitPass.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info BlitPass::kInfo { "BlitPass", "Blit a texture into a different texture." };

namespace
{
    const char kDst[] = "dst";
    const char kSrc[] = "src";
    const char kFilter[] = "filter";

    void regBlitPass(pybind11::module& m)
    {
        pybind11::class_<BlitPass, RenderPass, BlitPass::SharedPtr> pass(m, "BlitPass");
        pass.def_property(kFilter, &BlitPass::getFilter, &BlitPass::setFilter);
    }
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(BlitPass::kInfo, BlitPass::create);
    ScriptBindings::registerBinding(regBlitPass);
}

RenderPassReflection BlitPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(kDst, "The destination texture");
    r.addInput(kSrc, "The source texture");
    return r;
}

void BlitPass::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kFilter) setFilter(value);
        else logWarning("Unknown field '{}' in a BlitPass dictionary.", key);
    }
}

BlitPass::SharedPtr BlitPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new BlitPass(dict));
}

BlitPass::BlitPass(const Dictionary& dict)
    : RenderPass(kInfo)
{
    parseDictionary(dict);
}

Dictionary BlitPass::getScriptingDictionary()
{
    Dictionary d;
    d[kFilter] = mFilter;
    return d;
}

void BlitPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pSrcTex = renderData.getTexture(kSrc);
    const auto& pDstTex = renderData.getTexture(kDst);

    if (pSrcTex && pDstTex)
    {
        pRenderContext->blit(pSrcTex->getSRV(), pDstTex->getRTV(), RenderContext::kMaxRect, RenderContext::kMaxRect, mFilter);
    }
    else
    {
        logWarning("BlitPass::execute() - missing an input or output resource");
    }
}

void BlitPass::renderUI(Gui::Widgets& widget)
{
    static const Gui::DropdownList kFilterList =
    {
        { (uint32_t)Sampler::Filter::Linear, "Linear" },
        { (uint32_t)Sampler::Filter::Point, "Point" },
    };
    uint32_t f = (uint32_t)mFilter;

    if (widget.dropdown("Filter", kFilterList, f)) setFilter((Sampler::Filter)f);
}
