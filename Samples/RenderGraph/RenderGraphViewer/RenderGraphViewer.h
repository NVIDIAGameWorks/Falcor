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
#pragma once
#include "Falcor.h"

using namespace Falcor;

class RenderGraphViewer : public Renderer
{
public:    
    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;
    void onDataReload(SampleCallbacks* pSample) override;
    void onShutdown(SampleCallbacks* pSample) override;
    void onInitializeTesting(SampleCallbacks* pSample) override;
    void onBeginTestFrame(SampleTest* pSampleTest) override;
    
private:
    Scene::SharedPtr mpDefaultScene;
    FirstPersonCameraController mCamController;
    void addGraphDialog(SampleCallbacks* pCallbacks);
    void addGraphsFromFile(const std::string& filename, SampleCallbacks* pCallbacks);
    void removeActiveGraph();
    void loadScene(SampleCallbacks* pCallbacks);
    void loadSceneFromFile(const std::string& filename, SampleCallbacks* pCallbacks);

    struct DebugWindow
    {
        std::string windowName;
        std::string currentOutput;
        static size_t index;
    };

    struct GraphData
    {
        time_t fileModifiedTime;
        std::string filename;
        std::string name;
        RenderGraph::SharedPtr pGraph;
        std::string mainOutput;
        bool showAllOutputs = false;
        std::vector<std::string> originalOutputs;
        std::vector<DebugWindow> debugWindows;
    };

    void initGraph(const RenderGraph::SharedPtr& pGraph, const std::string& name, const std::string& filename, SampleCallbacks* pCallbacks, GraphData& data);
    std::vector<std::string> getGraphOutputs(const RenderGraph::SharedPtr& pGraph);
    void graphOutputsGui(Gui* pGui, SampleCallbacks* pSample);
    bool renderDebugWindow(Gui* pGui, const Gui::DropdownList& dropdown, DebugWindow& data, const uvec2& winSize); // Returns true if we need to close the window
    void renderOutputUI(Gui* pGui, const Gui::DropdownList& dropdown, std::string& selectedOutput);
    void addDebugWindow();

    std::vector<GraphData> mGraphs;
    uint32_t mActiveGraph = 0;

    // Editor stuff
    void openEditor();
    void resetEditor();
    void editorFileChangeCB();
    void applyEditorChanges();

    static const size_t kInvalidProcessId = -1; // We use this to know that the editor was launching the viewer
    size_t mEditorProcess = 0;
    std::string mEditorTempFile;
    std::string mEditorScript;
};
