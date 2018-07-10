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
#include "DepthPass.h"

namespace Falcor
{
    static std::string kDepth = "depth";

    DepthPass::SharedPtr DepthPass::create()
    {
        try
        {
            return SharedPtr(new DepthPass);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    DepthPass::DepthPass() : RenderPass("DepthPass", nullptr)
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::create({});
        mpState = GraphicsState::create();
        mpState->setProgram(pProgram);
        mpVars = GraphicsVars::create(pProgram->getReflector());
        mpFbo = Fbo::create();
        
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthTest(false).setStencilTest(false);

        Reflection::Field depth;
        depth.name = kDepth;
        depth.optional = false;
        depth.format = ResourceFormat::D32Float;
        depth.bindFlags = Resource::BindFlags::DepthStencil;
        mReflection.outputs.push_back(depth);
    }

    void DepthPass::sceneChangedCB()
    {
        mpSceneRenderer = nullptr;
        if (mpScene)
        {
            mpSceneRenderer = SceneRenderer::create(mpScene);
        }
    }

    bool DepthPass::isValid(std::string& log)
    {
        bool b = true;
        if (mpSceneRenderer == nullptr)
        {
            log += "DepthPass must have a scene attached to it\n";
            b = false;
        }

        const auto& pDepth = mpFbo->getDepthStencilTexture().get();
        if (!pDepth)
        {
            log += "DepthPass must have a depth texture attached\n";
            b = false;
        }

        if (mpFbo->checkStatus() == false)
        {
            log += "DepthPass FBO is invalid";
            b = false;
        }

        return b;
    }

    bool DepthPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        logError("DepthPass::setInput() - trying to set `" + name + "` which doesn't exist in this render-pass");
        return false;
    }

    bool DepthPass::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        if (!mpFbo)
        {
            logError("DepthPass::setOutput() - please call onResizeSwapChain() before setting an input");
            return false;
        }

        if (name == kDepth)
        {
            Texture::SharedPtr pDepth = std::dynamic_pointer_cast<Texture>(pResource);
            mpFbo->attachDepthStencilTarget(pDepth);
        }
        else
        {
            logError("DepthPass::setOutput() - trying to set `" + name + "` which doesn't exist in this render-pass");
            return false;
        }

        return true;
    }

    void DepthPass::execute(RenderContext* pContext)
    {
        pContext->clearDsv(mpFbo->getDepthStencilView().get(), 1, 0);
        if (mpSceneRenderer)
        {
            mpState->setFbo(mpFbo);
            pContext->pushGraphicsState(mpState);
            pContext->pushGraphicsVars(mpVars);
            mpSceneRenderer->renderScene(pContext);
            pContext->popGraphicsState();
            pContext->popGraphicsVars();
        }
    }

    std::shared_ptr<Resource> DepthPass::getOutput(const std::string& name) const
    {
        if (name == kDepth)
        {
            return mpFbo->getDepthStencilTexture();
        }        
        else return RenderPass::getOutput(name);
    }
    
    void DepthPass::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
    {
    }
}