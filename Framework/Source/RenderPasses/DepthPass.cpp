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
    static const std::string kDepth = "depth";
    static const std::string kDepthFormat = "depthFormat";

    static bool parseDictionary(DepthPass* pPass, const Dictionary& dict)
    {
        for (const auto& v : dict)
        {
            if (v.key() == kDepthFormat)
            {
                ResourceFormat f = (ResourceFormat)v.val();
                pPass->setDepthBufferFormat(f);
            }
            else
            {
                logWarning("Unknown field `" + v.key() + "` in a DepthPass dictionary");
           }
        }
        return true;
    }

    Dictionary DepthPass::getScriptingDictionary() const
    {
        Dictionary d;
        d[kDepthFormat] = mDepthFormat;
        return d;
    }

    DepthPass::SharedPtr DepthPass::create(const Dictionary& dict)
    {
        DepthPass* pThis = new DepthPass;
        return parseDictionary(pThis, dict) ? SharedPtr(pThis) : nullptr;
    }

    DepthPass::DepthPass() : RenderPass("DepthPass")
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::create({});
        mpState = GraphicsState::create();
        mpState->setProgram(pProgram);
        mpVars = GraphicsVars::create(pProgram->getReflector());
        mpFbo = Fbo::create();
    }

    RenderPassReflection DepthPass::reflect() const
    {
        RenderPassReflection reflector;
        reflector.addOutput(kDepth).setBindFlags(Resource::BindFlags::DepthStencil).setFormat(mDepthFormat).setSampleCount(0);
        return reflector;
    }

    void DepthPass::setScene(const Scene::SharedPtr& pScene)
    {
        mpSceneRenderer = nullptr;
        if (pScene)
        {
            mpSceneRenderer = SceneRenderer::create(pScene);
        }
    }

    void DepthPass::execute(RenderContext* pContext, const RenderData* pData)
    {
        if(mpSceneRenderer)
        {
            const auto& pDepth = pData->getTexture(kDepth);
            mpFbo->attachDepthStencilTarget(pDepth);

            mpState->setFbo(mpFbo);
            pContext->pushGraphicsState(mpState);
            pContext->pushGraphicsVars(mpVars);
            pContext->clearDsv(pDepth->getDSV().get(), 1, 0);
            mpSceneRenderer->renderScene(pContext);
            pContext->popGraphicsState();
            pContext->popGraphicsVars();
        }
    }

    DepthPass& DepthPass::setDepthBufferFormat(ResourceFormat format)
    {
        if (isDepthStencilFormat(format) == false)
        {
            logWarning("DepthPass buffer format must be a depth-stencil format");
        }
        else
        {
            mDepthFormat = format;
            mPassChangedCB();
        }
        return *this;
    }

    DepthPass& DepthPass::setDepthStencilState(const DepthStencilState::SharedPtr& pDsState)
    {
        mpState->setDepthStencilState(pDsState);
        return *this;
    }

    static const Gui::DropdownList kDepthFormats =
    {
        { (uint32_t)ResourceFormat::D16Unorm, "D16Unorm"},
        { (uint32_t)ResourceFormat::D32Float, "D32Float" },
        { (uint32_t)ResourceFormat::D24UnormS8, "D24UnormS8" },
        { (uint32_t)ResourceFormat::D32FloatS8X24, "D32FloatS8X24" },
    };

    void DepthPass::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup || pGui->beginGroup(uiGroup))
        {
            uint32_t depthFormat = (uint32_t)mDepthFormat;
            if (pGui->addDropdown("Buffer Format", kDepthFormats, depthFormat)) setDepthBufferFormat(ResourceFormat(depthFormat));

            if (uiGroup) pGui->endGroup();
        }
    }
}