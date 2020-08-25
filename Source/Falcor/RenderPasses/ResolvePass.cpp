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
#include "stdafx.h"
#include "ResolvePass.h"

namespace Falcor
{
    const char* ResolvePass::kDesc = "Resolve a multi-sampled texture";

    static const std::string kDst = "dst";
    static const std::string kSrc = "src";

    RenderPassReflection ResolvePass::reflect(const CompileData& compileData)
    {
        RenderPassReflection reflector;
        reflector.addInput(kSrc, "Multi-sampled texture").format(mFormat).texture2D(0, 0, 0);
        reflector.addOutput(kDst, "Destination texture. Must have a single sample").format(mFormat).texture2D(0, 0, 1);
        return reflector;
    }

    ResolvePass::SharedPtr ResolvePass::create(RenderContext* pRenderContext, const Dictionary& dictionary)
    {
        return SharedPtr(new ResolvePass);
    }
    
    void ResolvePass::execute(RenderContext* pContext, const RenderData& renderData)
    {
        auto pSrcTex = renderData[kSrc]->asTexture();
        auto pDstTex = renderData[kDst]->asTexture();

        if (pSrcTex && pDstTex)
        {
            if (pSrcTex->getSampleCount() == 1)
            {
                logWarning("ResolvePass::execute() - Cannot resolve from a non-multisampled texture.");
                return;
            }

            pContext->resolveResource(pSrcTex, pDstTex);
        }
        else
        {
            logWarning("ResolvePass::execute() - missing an input or output resource");
        }
    }
}
