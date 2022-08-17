/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include "Mogwai.h"
#include "RenderGraph/RenderGraphIR.h"
#include "RenderGraph/RenderGraphImportExport.h"
#include "Utils/Scripting/ScriptWriter.h"
#include "Utils/Settings.h"
#include <fstream>

namespace Mogwai
{
    namespace
    {
        const std::string kRunScript = "script";
        const std::string kLoadScene = "loadScene";
        const std::string kUnloadScene = "unloadScene";
        const std::string kSaveConfig = "saveConfig";
        const std::string kAddGraph = "addGraph";
        const std::string kSetActiveGraph = "setActiveGraph";
        const std::string kRemoveGraph = "removeGraph";
        const std::string kGetGraph = "getGraph";
        const std::string kUI = "ui";
        const std::string kKeyCallback = "keyCallback";
        const std::string kResizeSwapChain = "resizeSwapChain";
        const std::string kRenderFrame = "renderFrame";
        const std::string kActiveGraph = "activeGraph";
        const std::string kScene = "scene";
        const std::string kClock = "clock";
        const std::string kProfiler = "profiler";

        const std::string kRendererVar = "m";

        const std::string kGlobalHelp =
            Renderer::getVersionString() +
            "\nGlobal variables:\n" +
            "\tm                  Mogwai instance.\n"
            "\nGlobal functions\n" +
            "\trenderFrame()      Render a frame. If the clock is not paused, it will advance by one tick. You can use it inside for loops, for example to loop over a specific time-range.\n" +
            "\texit()             Terminate.\n";

        std::string windowConfig()
        {
            std::string s;
            SampleConfig c = gpFramework->getConfig();
            s += "# Window Configuration\n";
            s += ScriptWriter::makeMemberFunc(kRendererVar, kResizeSwapChain, c.windowDesc.width, c.windowDesc.height);
            s += ScriptWriter::makeSetProperty(kRendererVar, kUI, c.showUI);
            return s;
        }
    }

    void Renderer::saveConfig(const std::filesystem::path& path) const
    {
        std::string s;

        if (!mGraphs.empty())
        {
            s += "# Graphs\n";
            for (const auto& g : mGraphs)
            {
                s += RenderGraphExporter::getIR(g.pGraph);
                s += kRendererVar + "." + kAddGraph + "(" + RenderGraphIR::getFuncName(g.pGraph->getName()) + "())\n";
            }
            s += "\n";
        }

        if (mpScene)
        {
            s += "# Scene\n";
            s += ScriptWriter::makeMemberFunc(kRendererVar, kLoadScene, ScriptWriter::getPathString(mpScene->getPath()));
            const std::string sceneVar = kRendererVar + "." + kScene;
            s += mpScene->getScript(sceneVar);
            s += "\n";
        }

        s += windowConfig() + "\n";

        {
            s += "# Clock Settings\n";
            const std::string clockVar = kRendererVar + "." + kClock;
            s += gpFramework->getGlobalClock().getScript(clockVar) + "\n";
        }

        for (auto& pe : mpExtensions)
        {
            if (auto var = pe->getScriptVar(); !var.empty())
            {
                var = kRendererVar + "." + var;
                auto eStr = pe->getScript(var);
                if (eStr.size()) s += eStr + "\n";
            }
        }

        std::ofstream(path) << s;
    }

    void Renderer::registerScriptBindings(pybind11::module& m)
    {
        using namespace pybind11::literals;

        pybind11::class_<Renderer> renderer(m, "Renderer");
        renderer.def(kRunScript.c_str(), &Renderer::loadScript, "path"_a);
        renderer.def(kLoadScene.c_str(), &Renderer::loadScene, "path"_a, "buildFlags"_a = SceneBuilder::Flags::Default);
        renderer.def(kUnloadScene.c_str(), &Renderer::unloadScene);
        renderer.def(kSaveConfig.c_str(), &Renderer::saveConfig, "path"_a);
        renderer.def(kAddGraph.c_str(), &Renderer::addGraph, "graph"_a);
        renderer.def(kSetActiveGraph.c_str(),
            [](Renderer* pRenderer, const RenderGraph::SharedPtr& pGraph)
            {
                pRenderer->setActiveGraph(pGraph);
            }, "graph"_a);
        renderer.def(kRemoveGraph.c_str(), pybind11::overload_cast<const std::string&>(&Renderer::removeGraph), "name"_a);
        renderer.def(kRemoveGraph.c_str(), pybind11::overload_cast<const RenderGraph::SharedPtr&>(&Renderer::removeGraph), "graph"_a);
        renderer.def(kGetGraph.c_str(), &Renderer::getGraph, "name"_a);

        auto resizeSwapChain = [](Renderer* pRenderer, uint32_t width, uint32_t height) { gpFramework->resizeSwapChain(width, height); };
        renderer.def(kResizeSwapChain.c_str(), resizeSwapChain);

        auto renderFrame = [](Renderer* pRenderer) { ProgressBar::close(); gpFramework->renderFrame(); };
        renderer.def(kRenderFrame.c_str(), renderFrame);

        renderer.def_property_readonly(kScene.c_str(), &Renderer::getScene);
        renderer.def_property_readonly(kActiveGraph.c_str(), &Renderer::getActiveGraph);
        renderer.def_property_readonly(kClock.c_str(), [] (Renderer* pRenderer) { return &gpFramework->getGlobalClock(); });
        renderer.def_property_readonly(kProfiler.c_str(), [] (Renderer* pRenderer) { return Profiler::instancePtr(); });

        auto getUI = [](Renderer* pRenderer) { return gpFramework->isUiEnabled(); };
        auto setUI = [](Renderer* pRenderer, bool show) { gpFramework->toggleUI(show); };
        renderer.def_property(kUI.c_str(), getUI, setUI);

        renderer.def_property(kKeyCallback.c_str(), &Renderer::getKeyCallback, &Renderer::setKeyCallback);

        for (auto& pe : mpExtensions)
        {
            pe->registerScriptBindings(m);
            if (auto var = pe->getScriptVar(); !var.empty())
            {
                renderer.def_property_readonly(var.c_str(), [&pe] (Renderer* pRenderer) { return pe.get(); });
            }
        }

        // Replace the `help` function
        auto globalHelp = [this]() { pybind11::print(kGlobalHelp); };
        m.def("help", globalHelp);

        auto objectHelp = [](pybind11::object o)
        {
            auto b = pybind11::module::import("builtins");
            auto h = b.attr("help");
            h(o);
        };
        m.def("help", objectHelp, "object"_a);

        // Register global renderer variable.
        Scripting::getDefaultContext().setObject(kRendererVar, this);

        // Register deprecated global variables.
        Scripting::getDefaultContext().setObject("t", &gpFramework->getGlobalClock()); // PYTHONDEPRECATED

        auto findExtension = [this](const std::string& name)
        {
            for (auto& pe : mpExtensions)
            {
                if (pe->getName() == name) return pe.get();
            }
            FALCOR_ASSERT(false);
            return static_cast<Extension*>(nullptr);
        };

        Scripting::getDefaultContext().setObject("fc", findExtension("Frame Capture")); // PYTHONDEPRECATED
        Scripting::getDefaultContext().setObject("vc", findExtension("Video Capture")); // PYTHONDEPRECATED
        Scripting::getDefaultContext().setObject("tc", findExtension("Timing Capture")); // PYTHONDEPRECATED

        renderer.def("addOptions", [](Renderer* r, pybind11::dict d = {})
        {
            gpFramework->getSettings().addOptions(Dictionary(d));
            r->onOptionsChange();
        }, "dict"_a = pybind11::dict());
        renderer.def("addFilteredAttributes", [](Renderer* r, pybind11::dict d = {})
        {
            gpFramework->getSettings().addFilteredAttributes(Dictionary(d));
        }, "dict"_a = pybind11::dict());
        renderer.def("clearOptions", [](Renderer* r)
        {
            gpFramework->getSettings().clearOptions();
            r->onOptionsChange();
        });
        renderer.def("clearFilteredAttributes", [](Renderer* r)
        {
            gpFramework->getSettings().clearFilteredAttributes();
        });
    }
}
