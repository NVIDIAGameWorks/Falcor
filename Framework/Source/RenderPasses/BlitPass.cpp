/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "BlitPass.h"
#include "API/RenderContext.h"

#include "Utils/Gui.h"

namespace Falcor
{
    static const std::string kDst = "dst";
    static const std::string kSrc = "src";

    void BlitPass::reflect(RenderPassReflection& reflector) const
    {
        reflector.addOutput(kDst);
        reflector.addInput(kSrc);
    }

    BlitPass::SharedPtr BlitPass::create()
    {
        try
        {
            return SharedPtr(new BlitPass);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    BlitPass::BlitPass() : RenderPass("BlitPass")
    {
    }

    void BlitPass::execute(RenderContext* pContext, const RenderData* pRenderData)
    {
        const auto& pSrcTex = pRenderData->getTexture(kSrc);
        const auto& pDstTex = pRenderData->getTexture(kDst);

        if(pSrcTex && pDstTex)
        {
            pContext->blit(pSrcTex->getSRV(), pDstTex->getRTV());
        }
        else
        {
            logWarning("BlitPass::execute() - missing an input or output resource");
        }
    }
}
