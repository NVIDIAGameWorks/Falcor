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
#include "RenderGraphExe.h"

namespace Falcor
{
    void RenderGraphExe::execute(const Context& ctx)
    {
        PROFILE("RenderGraphExe::execute()");

        for (const auto& pass : mExecutionList)
        {
            PROFILE(pass.name);

            RenderData renderData(pass.name, mpResourceCache, ctx.pGraphDictionary, ctx.defaultTexDims, ctx.defaultTexFormat);
            pass.pPass->execute(ctx.pRenderContext, renderData);
        }
    }

    void RenderGraphExe::renderUI(Gui::Widgets& widget)
    {
        for (const auto& p : mExecutionList)
        {
            const auto& pPass = p.pPass;

            if (auto passGroup = widget.group(p.name))
            {
                // If you are thinking about displaying the profiler results next to the group label, it won't work. Since the times change every frame, IMGUI thinks it's a different group and will not expand it
                const auto& desc = pPass->getDesc();
                if (desc.size()) passGroup.tooltip(desc.c_str());
                pPass->renderUI(passGroup);
            }
        }
    }

    bool RenderGraphExe::onMouseEvent(const MouseEvent& mouseEvent)
    {
        bool b = false;
        for (const auto& p : mExecutionList)
        {
            const auto& pPass = p.pPass;
            b = b || pPass->onMouseEvent(mouseEvent);
        }
        return b;
    }

    bool RenderGraphExe::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        bool b = false;
        for (const auto& p : mExecutionList)
        {
            const auto& pPass = p.pPass;
            b = b || pPass->onKeyEvent(keyEvent);
        }
        return b;
    }

    void RenderGraphExe::onHotReload(HotReloadFlags reloaded)
    {
        for (const auto& p : mExecutionList)
        {
            const auto& pPass = p.pPass;
            pPass->onHotReload(reloaded);
        }
    }

    void RenderGraphExe::insertPass(const std::string& name, const RenderPass::SharedPtr& pPass)
    {
        mExecutionList.push_back(Pass(name, pPass));
    }

    Resource::SharedPtr RenderGraphExe::getResource(const std::string& name) const
    {
        assert(mpResourceCache);
        return mpResourceCache->getResource(name);
    }

    void RenderGraphExe::setInput(const std::string& name, const Resource::SharedPtr& pResource)
    {
        mpResourceCache->registerExternalResource(name, pResource);
    }
}
