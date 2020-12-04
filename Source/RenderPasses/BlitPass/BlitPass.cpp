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
#include "BlitPass.h"

namespace
{
    const char kDesc[] = "Blit a texture into a different texture";

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
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("BlitPass", kDesc, BlitPass::create);
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
        else logWarning("Unknown field '" + key + "' in a BlitPass dictionary");
    }
}

BlitPass::SharedPtr BlitPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new BlitPass(dict));
}

BlitPass::BlitPass(const Dictionary& dict)
{
    parseDictionary(dict);
}

std::string BlitPass::getDesc() { return kDesc; }

Dictionary BlitPass::getScriptingDictionary()
{
    Dictionary d;
    d[kFilter] = mFilter;
    return d;
}

void BlitPass::execute(RenderContext* pContext, const RenderData& renderData)
{
    const auto& pSrcTex = renderData[kSrc]->asTexture();
    const auto& pDstTex = renderData[kDst]->asTexture();

    if (pSrcTex && pDstTex)
    {
        pContext->blit(pSrcTex->getSRV(), pDstTex->getRTV(), uint4(-1), uint4(-1), mFilter);
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
