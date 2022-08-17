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
#include "Falcor.h"
#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderGraphUI.h"

using namespace Falcor;

class RenderGraphEditor : public IRenderer
{
public:
    struct Options
    {
        std::string graphFile;
        std::string graphName;
        bool runFromMogwai = false;
    };

    RenderGraphEditor(const Options& options);
    ~RenderGraphEditor();

    void onLoad(RenderContext* pRenderContext) override;
    void onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(uint32_t width, uint32_t height) override;
    void onGuiRender(Gui* pGui) override;
    void onDroppedFile(const std::filesystem::path& path) override;

private:
    void createNewGraph(const std::string& renderGraphName);
    void loadGraphsFromFile(const std::filesystem::path& path, const std::string& graphName = "");
    void serializeRenderGraph(const std::filesystem::path& path);
    void deserializeRenderGraph(const std::filesystem::path& path);
    void renderLogWindow(Gui::Widgets& widget);
    void loadAllPassLibraries();

    Options mOptions;

    std::vector<RenderGraph::SharedPtr> mpGraphs;
    std::vector<RenderGraphUI> mRenderGraphUIs;
    std::unordered_map<std::string, uint32_t> mGraphNamesToIndex;
    size_t mCurrentGraphIndex;
    uint2 mWindowSize;
    std::string mCurrentLog;
    std::string mNextGraphString;
    std::string mCurrentGraphOutput;
    std::string mGraphOutputEditString;
    std::filesystem::path mUpdateFilePath;
    Texture::SharedPtr mpDefaultIconTex;

    Gui::DropdownList mOpenGraphNames;
    bool mShowCreateGraphWindow = false;
    bool mShowDebugWindow = false;
    bool mViewerRunning = false;
    size_t mViewerProcess = 0;
    bool mResetGuiWindows = false;
};
