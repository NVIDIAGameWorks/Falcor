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
#include "BlitPass.h"

namespace
{
const char kDst[] = "dst";
const char kSrc[] = "src";
const char kFilter[] = "filter";
const char kOutputFormat[] = "outputFormat";

void regBlitPass(pybind11::module& m)
{
    pybind11::class_<BlitPass, RenderPass, ref<BlitPass>> pass(m, "BlitPass");
    pass.def_property(
        "filter",
        [](const BlitPass& self) { return enumToString(self.getFilter()); },
        [](BlitPass& self, const std::string& value) { self.setFilter(stringToEnum<TextureFilteringMode>(value)); }
    );
}
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, BlitPass>();
    ScriptBindings::registerBinding(regBlitPass);
}

BlitPass::BlitPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);
}

RenderPassReflection BlitPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(kDst, "The destination texture").format(mOutputFormat);
    r.addInput(kSrc, "The source texture");
    return r;
}

void BlitPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kFilter)
            setFilter(value);
        if (key == kOutputFormat)
            mOutputFormat = value;
        else
            logWarning("Unknown property '{}' in a BlitPass properties.", key);
    }
}

Properties BlitPass::getProperties() const
{
    Properties props;
    props[kFilter] = mFilter;
    if (mOutputFormat != ResourceFormat::Unknown)
        props[kOutputFormat] = mOutputFormat;
    return props;
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
    if (auto filter = mFilter; widget.dropdown("Filter", filter))
        setFilter(filter);
}
