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
#pragma once
#include "../../Mogwai.h"

namespace Mogwai
{
    class CaptureTrigger : public Extension
    {
    public:
        virtual ~CaptureTrigger() {};

        virtual void beginFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override final;
        virtual void endFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override final;
        virtual bool hasWindow() const override { return true; }
        virtual bool isWindowShown() const override { return mShowUI; }
        virtual void toggleWindow() override { mShowUI = !mShowUI; }
        virtual void registerScriptBindings(pybind11::module& m) override;
        virtual void activeGraphChanged(RenderGraph* pNewGraph, RenderGraph* pPrevGraph) override;
    protected:
        CaptureTrigger(Renderer* pRenderer, const std::string& name) : Extension(pRenderer, name) {}

        using Range = std::pair<uint64_t, uint64_t>; // Start frame and count

        virtual void beginRange(RenderGraph* pGraph, const Range& r) {};
        virtual void triggerFrame(RenderContext* pCtx, RenderGraph* pGraph, uint64_t frameID) {};
        virtual void endRange(RenderGraph* pGraph, const Range& r) {};

        void addRange(const RenderGraph* pGraph, uint64_t startFrame, uint64_t count);
        void reset(const RenderGraph* pGraph = nullptr);
        void renderUI(Gui::Window& w);

        void setOutputDirectory(const std::filesystem::path& path);
        const std::filesystem::path& getOutputDirectory() const { return mOutputDir; }

        void setBaseFilename(const std::string& baseFilename);
        const std::string& getBaseFilename() const { return mBaseFilename; }

        std::string getScript(const std::string& var) const override;
        std::filesystem::path getOutputPath() const;
        std::string getOutputNamePrefix(const std::string& output) const;

        using range_vec = std::vector<Range>;
        std::unordered_map<const RenderGraph*, range_vec> mGraphRanges;

        std::string mBaseFilename = "Mogwai";
        std::filesystem::path mOutputDir = ".";
        bool mShowUI = false;

        struct
        {
            RenderGraph* pGraph = nullptr;
            Range range;
        } mCurrent;
    };
}
