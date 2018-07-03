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
#include "NodeGraphGuiPass.h"

namespace Falcor
{
    static std::string kColor = "color";
    static std::string kDepth = "depth";

    static NodeGraphGuiPass::PassData createRenderPassData()
    {
        RenderPass::PassData data;
        RenderPass::PassData::Field output;
        output.bindFlags = Resource::BindFlags::RenderTarget;
        output.name = kColor;
        output.pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::Read);
        data.outputs.push_back(output);

        output.name = kDepth;
        output.format = ResourceFormat::D32Float;
        output.bindFlags = Resource::BindFlags::DepthStencil;
        data.outputs.push_back(output);

        return data;
    }

    const NodeGraphGuiPass::PassData NodeGraphGuiPass::kRenderPassData = createRenderPassData();

    NodeGraphGuiPass::SharedPtr NodeGraphGuiPass::create()
    {
        try
        {
            return SharedPtr(new NodeGraphGuiPass);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    NodeGraphGuiPass::NodeGraphGuiPass() : RenderPass("NodeGraphGuiPass", nullptr)
    {
        mpState = GraphicsState::create();
        mpFbo = Fbo::create();
    }

    bool NodeGraphGuiPass::isValid(std::string& log)
    {
        bool b = true;
        const auto& pColor = mpFbo->getColorTexture(0).get();

        if (!pColor)
        {
            log += "NodeGraphGuiPass must have a color texture attached\n";
            b = false;
        }

        if (mpFbo->checkStatus() == false)
        {
            log += "NodeGraphGuiPass FBO is invalid, probably incorrect dimensions";
            b = false;
        }

        return b;
    }

    bool NodeGraphGuiPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        logError("NodeGraphGuiPass::setInput() - trying to set `" + name + "` but this render-pass requires no inputs");
        return false;
    }

    bool NodeGraphGuiPass::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (!mpFbo)
        {
            logError("NodeGraphGuiPass::setOutput() - please call onResizeSwapChain() before setting an input");
            return false;
        }

        if (name == kColor)
        {
            Texture::SharedPtr pColor = std::dynamic_pointer_cast<Texture>(pResource);
            mpFbo->attachColorTarget(pColor, 0);
        }
        else if (name == kDepth)
        {
            Texture::SharedPtr pDepth = std::dynamic_pointer_cast<Texture>(pResource);
            mpFbo->attachDepthStencilTarget(pDepth, 0);
        }
        else
        {
            logError("NodeGraphGuiPass::setOutput() - trying to set `" + name + "` which doesn't exist in this render-pass");
            return false;
        }

        return true;
    }

    void NodeGraphGuiPass::execute(RenderContext* pContext)
    {
        pContext->clearFbo(mpFbo.get(), mClearColor, 1, 0);
        // draw the imgui stuff now
    }

    void NodeGraphGuiPass::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
    {
        pGui->addRgbaColor("Clear color", mClearColor);
    }
}