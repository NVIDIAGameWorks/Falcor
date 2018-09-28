/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "SamplePassLibrary.h"

static const std::string kDst = "dst";
static const std::string kSrc = "src";
static const std::string kFilter = "filter";

RenderPassReflection MyBlitPass::reflect() const
{
    RenderPassReflection reflector;

    reflector.addOutput(kDst);
    reflector.addInput(kSrc);

    return reflector;
}

static bool parseDictionary(MyBlitPass* pPass, const Dictionary& dict)
{
    for (const auto& v : dict)
    {
        if (v.key() == kFilter)
        {
            Sampler::Filter f = (Sampler::Filter)v.val();
            pPass->setFilter(f);
        }
        else
        {
            logWarning("Unknown field `" + v.key() + "` in a MyBlitPass dictionary");
        }
    }
    return true;
}

MyBlitPass::SharedPtr MyBlitPass::create(const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new MyBlitPass);
    return parseDictionary(pPass.get(), dict) ? pPass : nullptr;
}

Dictionary MyBlitPass::getScriptingDictionary() const
{
    Dictionary dict;
    dict[kFilter] = mFilter;
    return dict;
}

MyBlitPass::MyBlitPass() : RenderPass("MyBlitPass")
{
}

void MyBlitPass::execute(RenderContext* pContext, const RenderData* pRenderData)
{
    const auto& pSrcTex = pRenderData->getTexture(kSrc);
    const auto& pDstTex = pRenderData->getTexture(kDst);

    if (pSrcTex && pDstTex)
    {
        pContext->blit(pSrcTex->getSRV(), pDstTex->getRTV(), uvec4(-1), uvec4(-1), mFilter);
    }
    else
    {
        logWarning("MyBlitPass::execute() - missing an input or output resource");
    }
}

void MyBlitPass::renderUI(Gui* pGui, const char* uiGroup)
{
    if (!uiGroup || pGui->beginGroup(uiGroup))
    {
        static const Gui::DropdownList kFilterList =
        {
            { (uint32_t)Sampler::Filter::Linear, "Linear" },
            { (uint32_t)Sampler::Filter::Point, "Point" },
        };

        uint32_t f = (uint32_t)mFilter;
        if (pGui->addDropdown("Filter", kFilterList, f)) setFilter((Sampler::Filter)f);

        if (uiGroup) pGui->endGroup();
    }
}

extern "C" __declspec(dllexport) void getPasses(RenderPassLibrary& lib)
{
    lib.registerClass("MyBlitPass", "My Blit Class", MyBlitPass::create);
}