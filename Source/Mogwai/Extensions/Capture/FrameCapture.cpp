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
#include "FrameCapture.h"
#include <filesystem>

namespace Mogwai
{
    namespace
    {
        const std::string kScriptVar = "fc";
        const std::string kPrintFrames = "print";
        const std::string kFrames = "frames";
        const std::string kAddFrames = "addFrames";
        const std::string kUI = "ui";
        const std::string kOutputs = "outputs";
        const std::string kCapture = "capture";

        template<typename T>
        std::vector<typename T::value_type::first_type> getFirstOfPair(const T& pair)
        {
            std::vector<typename T::value_type::first_type> v;
            v.reserve(pair.size());
            for (auto p : pair) v.push_back(p.first);
            return v;
        }
    }

    MOGWAI_EXTENSION(FrameCapture);

    FrameCapture::UniquePtr FrameCapture::create(Renderer* pRenderer)
    {
        return UniquePtr(new FrameCapture(pRenderer));
    }

    void FrameCapture::renderUI(Gui* pGui)
    {
        if (mShowUI)
        {
            auto w = Gui::Window(pGui, mName.c_str(), mShowUI, {}, { 800, 400 });

            CaptureTrigger::renderUI(w);

            w.checkbox("Capture All Outputs", mCaptureAllOutputs);
            w.tooltip("Capture all available outputs instead of the marked ones only.");

            if (w.button("Capture Current Frame")) capture();
        }
    }

    void FrameCapture::scriptBindings(Bindings& bindings)
    {
        CaptureTrigger::scriptBindings(bindings);
        auto& m = bindings.getModule();

        pybind11::class_<FrameCapture, CaptureTrigger> frameCapture(m, "FrameCapture");

        bindings.addGlobalObject(kScriptVar, this, "Frame Capture Helpers");

        // Members
        frameCapture.def(kFrames.c_str(), pybind11::overload_cast<const RenderGraph*, const uint64_vec&>(&FrameCapture::addFrames)); // PYTHONDEPRECATED
        frameCapture.def(kFrames.c_str(), pybind11::overload_cast<const std::string&, const uint64_vec&>(&FrameCapture::addFrames)); // PYTHONDEPRECATED
        frameCapture.def(kAddFrames.c_str(), pybind11::overload_cast<const RenderGraph*, const uint64_vec&>(&FrameCapture::addFrames), "graph"_a, "frames"_a);
        frameCapture.def(kAddFrames.c_str(), pybind11::overload_cast<const std::string&, const uint64_vec&>(&FrameCapture::addFrames), "name"_a, "frames"_a);

        auto printGraph = [](FrameCapture* pFC, RenderGraph* pGraph) { pybind11::print(pFC->graphFramesStr(pGraph)); };
        frameCapture.def(kPrintFrames.c_str(), printGraph, "graph"_a);
        frameCapture.def(kCapture.c_str(), &FrameCapture::capture);
        auto printAllGraphs = [](FrameCapture* pFC)
        {
            std::string s;
            for (const auto& g : pFC->mGraphRanges) {s += "'" + g.first->getName() + "':\n" + pFC->graphFramesStr(g.first) + "\n";}
            pybind11::print(s.empty() ? "Empty" : s);
        };
        frameCapture.def(kPrintFrames.c_str(), printAllGraphs);

        // Settings
        auto getUI = [](FrameCapture* pFC) { return pFC->mShowUI; };
        auto setUI = [](FrameCapture* pFC, bool show) { pFC->mShowUI = show; };
        frameCapture.def_property(kUI.c_str(), getUI, setUI);
    }

    std::string FrameCapture::getScript()
    {
        std::string s;

        s += "# Frame Capture\n";
        s += CaptureTrigger::getScript(kScriptVar);

        for (const auto& g : mGraphRanges)
        {
            s += Scripting::makeMemberFunc(kScriptVar, kAddFrames, g.first->getName(), getFirstOfPair(g.second));
        }
        return s;
    }

    void FrameCapture::triggerFrame(RenderContext* pCtx, RenderGraph* pGraph, uint64_t frameID)
    {
        std::vector<std::string> unmarkedOutputs;

        if (mCaptureAllOutputs)
        {
            // Mark all outputs and (re)execute the graph.
            unmarkedOutputs = pGraph->getUnmarkedOutputs();
            for (const auto& output : unmarkedOutputs) pGraph->markOutput(output);
            pGraph->execute(pCtx);
        }

        for (uint32_t i = 0 ; i < pGraph->getOutputCount() ; i++)
        {
            Texture* pTex = pGraph->getOutput(i)->asTexture().get();
            assert(pTex);
            auto ext = Bitmap::getFileExtFromResourceFormat(pTex->getFormat());
            auto format = Bitmap::getFormatFromFileExtension(ext);
            std::string filename = getOutputNamePrefix(pGraph->getOutputName(i)) + std::to_string(gpFramework->getGlobalClock().getFrame()) + "." + ext;
            pTex->captureToFile(0, 0, filename, format);
        }

        if (mCaptureAllOutputs && !unmarkedOutputs.empty())
        {
            for (const auto& output : unmarkedOutputs) pGraph->unmarkOutput(output);
            pGraph->compile(pCtx);
        }
    }

    void FrameCapture::addFrames(const RenderGraph* pGraph, const uint64_vec& frames)
    {
        for (auto f : frames) addRange(pGraph, f, 1);
    }

    void FrameCapture::addFrames(const std::string& graphName, const uint64_vec& frames)
    {
        auto pGraph = mpRenderer->getGraph(graphName).get();
        if (!pGraph) throw std::runtime_error("Can't find a graph named '" + graphName + "'");
        this->addFrames(pGraph, frames);
    }

    std::string FrameCapture::graphFramesStr(const RenderGraph* pGraph)
    {
        const auto& ranges = mGraphRanges[pGraph];
        std::string s("\t");
        s += kFrames + " = " + ScriptBindings::repr(getFirstOfPair(ranges));
        return s;
    }

    void FrameCapture::capture()
    {
        auto pGraph = mpRenderer->getActiveGraph();
        if (!pGraph) return;
        uint64_t frameID = gpFramework->getGlobalClock().getFrame();
        triggerFrame(gpDevice->getRenderContext(), pGraph, frameID);
    }
}
