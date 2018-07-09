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

    void ShadowPass::createRenderPassData()
    {
        mRenderPassData = PassData();

        RenderPass::PassData::Field shadowMap;
        shadowMap.bindFlags = Resource::BindFlags::RenderTarget;
        shadowMap.name = kShadowMap;
        shadowMap.pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::Read);
        shadowMap.format = ResourceFormat::RGBA16Float;
        mRenderPassData.outputs.push_back(shadowMap);

        RenderPass::PassData::Field depth;
        depth.name = kDepth;
        depth.required = false;
        depth.format = ResourceFormat::Unknown;
        depth.bindFlags = Resource::BindFlags::ShaderResource;
        mRenderPassData.inputs.push_back(depth);
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

    ShadowPass::ShadowPass(uint32_t width, uint32_t height) : RenderPass("ShadowMapPass", nullptr), mSmHeight(height), mSmWidth(width)
    {
        createRenderPassData();
    }

    bool ShadowPass::isValid(std::string& log)
    {
        bool b = true;
        if (mpShadowMap == nullptr)
        {
            log += "ShadowPass must have an shadow-map output attached\n";
            b = false;
        }

        return b;
    }

    bool ShadowPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (name == kDepth)
        {
            Texture::SharedPtr pDepth = std::dynamic_pointer_cast<Texture>(pResource);
            mpDepthIn = pDepth;
        }
        else
        {
            logError("SceneRenderPass::setInput() - trying to set `" + name + "` which doesn't exist in this render-pass");
            return false;
        }
        return false;
    }

    bool ShadowPass::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (name == kShadowMap)
        {
            mpShadowMap = std::dynamic_pointer_cast<Texture>(pResource);
            mpCsm = CascadedShadowMaps::create(mSmWidth, mSmHeight, mpShadowMap->getWidth(), mpShadowMap->getHeight(), mpScene->getLight(0), mpScene);
        }
        else
        {
            logError("SceneRenderPass::setOutput() - trying to set `" + name + "` which doesn't exist in this render-pass");
            return false;
        }

        return true;
    }

    void ShadowPass::execute(RenderContext* pContext)
    {
        assert(mpCsm);
        assert(mpShadowMap);
        auto& pVisBuffer = mpCsm->generateVisibilityBuffer(pContext, mpScene->getActiveCamera().get(), nullptr);
        pContext->blit(pVisBuffer->getSRV(0, 1, 0, 1), mpShadowMap->getRTV(0, 0, 1));
    }

    std::shared_ptr<Resource> ShadowPass::getOutput(const std::string& name) const
    {
        if (name == kShadowMap)
        {
            return mpShadowMap;
        }        
        else return RenderPass::getOutput(name);
    }

    std::shared_ptr<Resource> ShadowPass::getInput(const std::string& name) const
    {
        if (name == kDepth)
        {
            return mpDepthIn;
        }
        else return RenderPass::getInput(name);
    }

    void ShadowPass::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
    {
        if (mpCsm) mpCsm->renderUi(pGui, "Shadow Pass");
    }
}