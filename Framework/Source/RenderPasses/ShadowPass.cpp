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
#include "ShadowPass.h"

namespace Falcor
{
    static std::string kDepth = "depth";
    static std::string kShadowMap = "shadowMap";

    void ShadowPass::describe(RenderPassReflection& reflector) const
    {
        const auto& pTex2DType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D);
        reflector.addOutput(kShadowMap).setFormat(ResourceFormat::RGBA16Float);
        reflector.addInput(kDepth).setFlags(RenderPassReflection::Field::Flags::Optional);
    }

    ShadowPass::SharedPtr ShadowPass::create(uint32_t width, uint32_t height)
    {
        try
        {
            return SharedPtr(new ShadowPass(width, height));
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    ShadowPass::ShadowPass(uint32_t width, uint32_t height) : RenderPass("ShadowMapPass"), mSmHeight(height), mSmWidth(width)
    {
    }

    void ShadowPass::execute(RenderContext* pContext, const RenderData* pRenderData)
    {   
        const auto& pDepthIn = std::dynamic_pointer_cast<Texture>(pRenderData->getResource(kDepth));
        const auto& pShadowMap = std::dynamic_pointer_cast<Texture>(pRenderData->getResource(kShadowMap));

        if(!mpCsm)
        {
            mpCsm = CascadedShadowMaps::create(mSmWidth, mSmHeight, pShadowMap->getWidth(), pShadowMap->getHeight(), mpScene->getLight(0), mpScene);
        }

        auto& pVisBuffer = mpCsm->generateVisibilityBuffer(pContext, mpScene->getActiveCamera().get(), pDepthIn);
        pContext->blit(pVisBuffer->getSRV(0, 1, 0, 1), pShadowMap->getRTV(0, 0, 1));
    }

    void ShadowPass::renderUI(Gui* pGui)
    {
        if (mpCsm) mpCsm->renderUi(pGui, "Shadow Pass");
    }
}