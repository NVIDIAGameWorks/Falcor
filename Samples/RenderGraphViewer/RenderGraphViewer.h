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
    ~RenderGraphViewer();

    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;
    void RenderGraphViewer::onDataReload(SampleCallbacks* pSample) override;

private:
    FirstPersonCameraController mCamControl;
    void loadScene(const std::string& filename, bool showProgressBar, SampleCallbacks* pSample);
    RenderGraph::SharedPtr createGraph(SampleCallbacks* pSample);
    void fileWriteCallback(const std::string& filename);
    void loadGraphFromFile(SampleCallbacks* pSample, const std::string& filename);
    
    void createDefaultGraph(SampleCallbacks* pSample);
    void insertNewGraph(const RenderGraph::SharedPtr& pGraph, const std::string& fileName, const std::string& name);
    void resetGraphOutputs();
    void updateOutputDropdown(const std::string& passName);

    Scene::SharedPtr mpScene;
    std::string mSceneFilename;
    
    struct DebugWindowInfo
    {
        std::string mGraphName;
        std::string mOutputName;
        bool mRenderOutput = true;
        uint32_t mNextOutputIndex = 0;
    };

    std::unordered_map<std::string, DebugWindowInfo> mDebugWindowInfos;
    bool mShowAllOutputs = true;

    bool mEditorRunning = false;
    bool mApplyGraphChanges = false;
    size_t mEditorProcess = 0;
    std::string mTempFilePath;
    std::string mFocusedRenderGraphName;
    std::string mEditingRenderGraphName;

    struct GraphViewerInfo
    {
        RenderGraph::SharedPtr mpGraph;
        std::string mOutputString = "BlitPass.dst";
        uint32_t mGraphOutputIndex = 0;
        bool mEnableDepthPrePass = true;
        std::string mFileName;
        std::string mName;
        std::vector< RenderGraph::OutputInfo > mCurrentOutputs;
        std::unordered_set<std::string> mOriginalOutputNames;
        std::vector<std::string> mOutputNames;
        std::vector<std::string> mScriptBacklog;
        Gui::DropdownList mOutputDropdown;
    };

    Gui::DropdownList mRenderGraphsList;
    uint32_t mActiveGraphIndex;
    std::unordered_map<std::string, GraphViewerInfo> mpRenderGraphs;
    std::unordered_set<std::string> mActiveGraphNames; // all graphs that have previewing graph outputs
};
