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
#include "FrameCapture.h"
#include <filesystem>

namespace Mogwai
{
    namespace
    {
        const std::string kPrintFrames = "print";
        const std::string kFrames = "frames";
        const std::string kScriptVar = "fc";
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
            auto w = Gui::Window(pGui, "Frame Capture", mShowUI, {}, { 400, 400 });
            CaptureTrigger::renderUI(w);
        }
    }

    void FrameCapture::scriptBindings(Bindings& bindings)
    {
        CaptureTrigger::scriptBindings(bindings);
        auto& m = bindings.getModule();
        auto fc = m.class_<FrameCapture, CaptureTrigger>("FrameCapture");
        bindings.addGlobalObject(kScriptVar, this, "Frame Capture Helpers");

        // Members
        fc.func_(kFrames.c_str(), ScriptBindings::overload_cast<const RenderGraph*, const uint64_vec&>(&FrameCapture::frames));
        fc.func_(kFrames.c_str(), ScriptBindings::overload_cast<const std::string&, const uint64_vec&>(&FrameCapture::frames));

        auto printGraph = [](FrameCapture* pFC, RenderGraph* pGraph) { pybind11::print(pFC->graphFramesStr(pGraph)); };
        fc.func_(kPrintFrames.c_str(), printGraph);
        fc.func_(kCapture.c_str(), &FrameCapture::capture);
        auto printAllGraphs = [](FrameCapture* pFC)
        {
            std::string s;
            for (const auto& g : pFC->mGraphRanges) {s += "`" + g.first->getName() + "`:\n" + pFC->graphFramesStr(g.first) + "\n";}
            pybind11::print(s.empty() ? "Empty" : s);
        };
        fc.func_(kPrintFrames.c_str(), printAllGraphs);

        // Settings
        auto showUI = [](FrameCapture* pFC, bool show) { pFC->mShowUI = show; };
        fc.func_(kUI.c_str(), showUI, "show"_a = true);
    }

    std::string FrameCapture::getScript()
    {
        std::string s;

        s += "# Frame Capture\n";
        s += CaptureTrigger::getScript(kScriptVar);

        for (const auto& g : mGraphRanges)
        {
            s += Scripting::makeMemberFunc(kScriptVar, kFrames, g.first->getName(), getFirstOfPair(g.second));
        }
        return s;
    }

    void FrameCapture::triggerFrame(RenderContext* pCtx, RenderGraph* pGraph, uint64_t frameID)
    {
        for (uint32_t i = 0 ; i < pGraph->getOutputCount() ; i++)
        {
            Texture* pTex = pGraph->getOutput(i)->asTexture().get();
            assert(pTex);
            std::string filename = getOutputNamePrefix(pGraph->getOutputName(i)) + to_string(gpFramework->getGlobalClock().frame()) + ".";;
            auto ext = Bitmap::getFileExtFromResourceFormat(pTex->getFormat());
            filename += ext;
            auto format = Bitmap::getFormatFromFileExtension(ext);
            pTex->captureToFile(0, 0, filename, format);
        }
    }

    void FrameCapture::frames(const RenderGraph* pGraph, const uint64_vec& frames)
    {
        for (auto f : frames) addRange(pGraph, f, 1);
    }

    void FrameCapture::frames(const std::string& graphName, const uint64_vec& frames)
    {
        auto pGraph = mpRenderer->getGraph(graphName).get();
        if (!pGraph) throw std::runtime_error("Can't find a graph named `" + graphName + "`");
        this->frames(pGraph, frames);
    }

    std::string FrameCapture::graphFramesStr(const RenderGraph* pGraph)
    {
        const auto& ranges = mGraphRanges[pGraph];
        std::string s("\t");
        s += kFrames + " = " + to_string(getFirstOfPair(ranges));
        return s;
    }

    void FrameCapture::capture()
    {
        auto pGraph = mpRenderer->getActiveGraph();
        if (!pGraph) return;
        uint64_t frameID = gpFramework->getGlobalClock().frame();
        triggerFrame(gpDevice->getRenderContext(), pGraph, frameID);
    }
}
