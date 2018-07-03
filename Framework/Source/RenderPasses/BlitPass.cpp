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

// #include "Externals/RapidJson/include/rapidjson/rapidjson.h"
#include "Utils/Gui.h"

namespace Falcor
{
    static const std::string kDst = "dst";
    static const std::string kSrc = "src";

    static BlitPass::PassData createRenderPassData()
    {
        RenderPass::PassData data;
        RenderPass::PassData::Field dstField;
        dstField.bindFlags = Resource::BindFlags::RenderTarget;
        dstField.name = kDst;
        dstField.pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::Undefined);
        data.outputs.push_back(dstField);

        RenderPass::PassData::Field srcField;
        srcField.bindFlags = Resource::BindFlags::None;
        srcField.name = kSrc;
        srcField.pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::Undefined);
        data.inputs.push_back(srcField);

        return data;
    }

    const BlitPass::PassData BlitPass::kRenderPassData = createRenderPassData();

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

    BlitPass::BlitPass() : RenderPass("BlitPass", nullptr)
    {
    }

    bool BlitPass::isValid(std::string& log)
    {
        bool b = true;

        if (!mpSrc)
        {
            log += "BlitPass must have a source texture attached\n";
            b = false;
        }

        if (!mpDst)
        {
            log += "BlitPass must have a destination texture attached\n";
            b = false;
        }

        return b;
    }

    static bool verifyPassInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (name != kSrc)
        {
            logError("BlitPass doesn't have an input named `" + name + "`");
            return false;
        }

        if(pResource)
        {
            Texture::SharedPtr pSrc = std::dynamic_pointer_cast<Texture>(pResource);

            if (!pSrc)
            {
                logError("BlitPass - the source resource must be a texture");
                return false;
            }

            if (is_set(pSrc->getBindFlags(), Resource::BindFlags::ShaderResource) == false)
            {
                logError("BlitPass - the source resource must be created with the ShaderResource bind-flag");
                return false;
            }
        }
        return true;
    }

    bool BlitPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (verifyPassInput(name, pResource) == false) return false;

        mpSrc = std::dynamic_pointer_cast<Texture>(pResource);
        return true;
    }

    static bool verifyPassOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (name != kDst)
        {
            logError("BlitPass doesn't have an output named `" + name + "`");
            return false;
        }

        if(pResource)
        {
            Texture::SharedPtr pDst = std::dynamic_pointer_cast<Texture>(pResource);

            if (!pDst)
            {
                logError("BlitPass - the destination resource must be a texture");
                return false;
            }

            if (is_set(pDst->getBindFlags(), Resource::BindFlags::RenderTarget) == false)
            {
                logError("BlitPass - the destination resource must be created with the RenderTarget bind-flag");
                return false;
            }
        }

        return true;
    }

    bool BlitPass::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (verifyPassOutput(name, pResource) == false) return false;

        mpDst = std::dynamic_pointer_cast<Texture>(pResource);
        return true;
    }

    void BlitPass::execute(RenderContext* pContext)
    {
        assert(isValid(std::string()));
        pContext->blit(mpSrc->getSRV(), mpDst->getRTV());
    }

    void BlitPass::renderUI(Gui* pGui, const std::string& name)
    {
    }

    std::shared_ptr<Resource> BlitPass::getOutput(const std::string& name) const
    {
        if (verifyPassOutput(name, nullptr) == false) nullptr;
        return mpDst;
    }

    std::shared_ptr<Resource> BlitPass::getInput(const std::string& name) const
    {
        if (verifyPassInput(name, nullptr) == false) return nullptr;
        return mpSrc;
    }
}
