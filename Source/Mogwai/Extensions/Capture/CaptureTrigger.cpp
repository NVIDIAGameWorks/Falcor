/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "CaptureTrigger.h"
#include "Utils/Scripting/ScriptWriter.h"
#include <filesystem>

namespace Mogwai
{
    namespace
    {
        const std::string kReset = "reset";
        const std::string kRemoveGraph = "removeGraph";
        const std::string kOutputDir = "outputDir";
        const std::string kBaseFilename = "baseFilename";

        void throwIfOverlapping(uint64_t xStart, uint64_t xCount, uint64_t yStart, uint64_t yCount)
        {
            uint64_t xEnd = xStart + xCount - 1;
            uint64_t yEnd = yStart + yCount - 1;
            if (xStart <= yEnd && yStart <= xEnd)
            {
                FALCOR_THROW("This range overlaps an existing range!");
            }
        }

        template<typename T>
        std::optional<typename T::value_type> findRange(const T& frames, uint64_t startFrame)
        {
            for (auto r : frames)
            {
                if (r.first == startFrame) return r;
            }
            return std::nullopt;
        }
    }

    void CaptureTrigger::addRange(const RenderGraph* pGraph, uint64_t startFrame, uint64_t count)
    {
        auto& ranges = mGraphRanges[pGraph];

        if (count == 0)
        {
            for (auto r = ranges.begin(); r != ranges.end(); r++)
            {
                if (r->first == startFrame)
                {
                    ranges.erase(r);
                    return;
                }
            }
        }

        for (auto& range : ranges)
        {
            if (startFrame == range.first && count == range.second) continue; // Silently ignore existing ranges
            throwIfOverlapping(startFrame, count, range.first, range.second);
        }

        ranges.push_back({ startFrame, count });
    }

    void CaptureTrigger::reset(const RenderGraph* pGraph)
    {
        if (pGraph) mGraphRanges.erase(pGraph);
        else        mGraphRanges.clear();
    }

    void CaptureTrigger::beginFrame(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
    {
        RenderGraph* pGraph = mpRenderer->getActiveGraph();
        if (!pGraph) return;
        uint64_t frameId = mpRenderer->getGlobalClock().getFrame();
        if (mGraphRanges.find(pGraph) == mGraphRanges.end()) return;
        const auto& ranges = mGraphRanges.at(pGraph);

        if (mCurrent.pGraph)
        {
            FALCOR_ASSERT(pGraph == mCurrent.pGraph);
            return;
        }

        // Check if we need to start a range
        for (auto& r : ranges)
        {
            if (r.first == frameId)
            {
                mCurrent.pGraph = pGraph;
                mCurrent.range = r;
                beginRange(pGraph, r);
                break;
            }
        }
    }

    void CaptureTrigger::endFrame(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
    {
        if (!mCurrent.pGraph) return;
        uint64_t frameId = mpRenderer->getGlobalClock().getFrame();
        const auto& ranges = mGraphRanges.at(mCurrent.pGraph);

        triggerFrame(pRenderContext, mCurrent.pGraph, frameId);

        uint64_t end = mCurrent.range.first + mCurrent.range.second;
        if (frameId + 1 == end)
        {
            endRange(mCurrent.pGraph, mCurrent.range);
            mCurrent = {};
        }
    }

    void CaptureTrigger::activeGraphChanged(RenderGraph* pNewGraph, RenderGraph* pPrevGraph)
    {
        if (mCurrent.pGraph)
        {
            endRange(mCurrent.pGraph, mCurrent.range);
            mCurrent = {};
        }
    }

    void CaptureTrigger::renderBaseUI(Gui::Window& w)
    {
        w.textbox("Base Filename", mBaseFilename);
        w.text("Output Directory\n" + mOutputDir.string());
        w.tooltip("Relative paths are treated as relative to the runtime directory (" + getRuntimeDirectory().string() + ").");
        if (w.button("Change Folder"))
        {
            std::filesystem::path path;
            if (chooseFolderDialog(path)) setOutputDirectory(path);
        }
    }

    void CaptureTrigger::setOutputDirectory(const std::filesystem::path& path_)
    {
        std::filesystem::path path = path_;
        if (path.is_absolute())
        {
            // Use relative path to executable directory if possible.
            auto relativePath = path.lexically_relative(getRuntimeDirectory());
            if (!relativePath.empty() && relativePath.string().find("..") == std::string::npos) path = relativePath;
        }
        mOutputDir = path;
    }

    void CaptureTrigger::setBaseFilename(const std::string& baseFilename)
    {
        mBaseFilename = baseFilename;
    }

    void CaptureTrigger::registerScriptBindings(pybind11::module& m)
    {
        using namespace pybind11::literals;

        if (pybind11::hasattr(m, "CaptureTrigger")) return;

        pybind11::class_<CaptureTrigger> captureTrigger(m, "CaptureTrigger");

        // Members
        captureTrigger.def(kReset.c_str(), &CaptureTrigger::reset, "graph"_a = nullptr);

        // Properties
        captureTrigger.def_property(kOutputDir.c_str(), &CaptureTrigger::getOutputDirectory, &CaptureTrigger::setOutputDirectory);
        captureTrigger.def_property(kBaseFilename.c_str(), &CaptureTrigger::getBaseFilename, &CaptureTrigger::setBaseFilename);
    }

    std::string CaptureTrigger::getScript(const std::string& var) const
    {
        std::string s;
        s += ScriptWriter::makeSetProperty(var, kOutputDir, ScriptWriter::getPathString(mOutputDir));
        s += ScriptWriter::makeSetProperty(var, kBaseFilename, mBaseFilename);
        return s;
    }

    std::filesystem::path CaptureTrigger::getOutputPath() const
    {
        auto path = mOutputDir;
        if (!path.is_absolute()) path = std::filesystem::absolute(getRuntimeDirectory() / path);
        return path;
    }

    std::string CaptureTrigger::getOutputNamePrefix(const std::string& output) const
    {
        auto path = getOutputPath();
        path /= mBaseFilename + "." + output + ".";
        return path.string();
    }
}
