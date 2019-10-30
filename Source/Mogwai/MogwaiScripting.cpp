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
#include "stdafx.h"

namespace Mogwai
{
    namespace
    {
        const std::string kRunScript = "script";
        const std::string kLoadScene = "loadScene";
        const std::string kSaveConfig = "saveConfig";
        const std::string kAddGraph = "addGraph";
        const std::string kRemoveGraph = "removeGraph";
        const std::string kToggleUI = "ui";
        const std::string kResizeSwapChain = "resizeSwapChain";
        const std::string kActiveGraph = "activeGraph";
        const std::string kGetGraph = "graph";
        const std::string kGetScene = "scene";

        template<typename T>
        std::string prepareHelpMessage(const T& g)
        {
            std::string s = Renderer::getVersionString() + "\nGlobal utility objects:\n";
            static const size_t kMaxSpace = 8;
            for (auto n : g)
            {
                s += "\t`" + n.first + "`";
                s += (n.first.size() >= kMaxSpace) ? " " : std::string(kMaxSpace - n.first.size(), ' ');
                s += n.second;
                s += "\n";
            }

            s += "\nGlobal functions\n";
            s += "\trenderFrame()      Render a frame. If the clock is not paused, it will advance by one tick. You can use it inside `For loops`, for example to loop over a specific time-range\n";
            s += "\texit()             Exit Mogwai\n";
            return s;
        }

        std::string windowConfig()
        {
            std::string s;
            SampleConfig c = gpFramework->getConfig();
            s += "# Window Configuration\n";
            s += Scripting::makeMemberFunc(kRendererVar, kResizeSwapChain, c.windowDesc.width, c.windowDesc.height);
            s += Scripting::makeMemberFunc(kRendererVar, kToggleUI, c.showUI);
            return s;
        }
    }

    void Renderer::dumpConfig(std::string filename) const
    {
        if(filename.empty())
        {
            if (!saveFileDialog(Scripting::kFileExtensionFilters, filename)) return;
        }

        std::string s;

        if (mpScene)
        {
            s += "# Scene\n";
            s += Scripting::makeMemberFunc(kRendererVar, kLoadScene, filenameString(mpScene->getFilename()));
        }

        if(mGraphs.size()) s += "\n# Graphs\n";
        for (auto& g : mGraphs)
        {
            s += RenderGraphExporter::getIR(g.pGraph);
            s += std::string(kRendererVar) + "." + kAddGraph + "(" + RenderGraphIR::getFuncName(g.pGraph->getName()) + "())\n";
        }

        s += "\n" + windowConfig() + "\n";

        for (auto& pe : mpExtensions)
        {
            auto eStr = pe->getScript();
            if (eStr.size()) s += eStr + "\n";
        }

        std::ofstream(filename) << s;
    }

    void Renderer::registerScriptBindings(ScriptBindings::Module& m)
    {
        auto c = m.class_<Renderer>("Renderer");

        c.func_(kRunScript.c_str(), &Renderer::loadScript, "filename"_a = std::string());
        c.func_(kLoadScene.c_str(), &Renderer::loadScene);
        c.func_(kSaveConfig.c_str(), &Renderer::dumpConfig, "filename"_a = std::string());
        c.func_(kAddGraph.c_str(), &Renderer::addGraph);
        c.func_(kRemoveGraph.c_str(), ScriptBindings::overload_cast<const std::string&>(&Renderer::removeGraph));
        c.func_(kRemoveGraph.c_str(), ScriptBindings::overload_cast<const RenderGraph::SharedPtr&>(&Renderer::removeGraph));
        c.func_(kGetGraph.c_str(), &Renderer::getGraph);
        c.func_(kGetScene.c_str(), &Renderer::getScene);

        Extension::Bindings b(m, c);
        b.addGlobalObject(kRendererVar, this, "The engine");
        for (auto& pe : mpExtensions) pe->scriptBindings(b);
        mGlobalHelpMessage = prepareHelpMessage(b.mGlobalObjects);

        // Replace the `help` function
        auto globalHelp = [this]() { pybind11::print(mGlobalHelpMessage);};
        m.func_("help", globalHelp);

        auto objectHelp = [](pybind11::object o)
        {
            auto b = pybind11::module::import("builtins");
            auto h = b.attr("help");
            h(o);
        };
        m.func_("help", objectHelp);

        auto resize = [](Renderer* pRenderer, uint32_t width, uint32_t height) {gpFramework->resizeSwapChain(width, height); };
        c.func_(kResizeSwapChain.c_str(), resize);

        auto toggleUI = [](Renderer* pRenderer, bool show) {gpFramework->toggleUI(show); };
        c.func_(kToggleUI.c_str(), toggleUI, "show"_a = true);
        c.func_(kActiveGraph.c_str(), &Renderer::getActiveGraph);
    }
}
