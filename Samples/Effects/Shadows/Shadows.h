/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "SampleTest.h"

using namespace Falcor;

class Shadows : public Renderer
{
public:
    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;

private:
    void displayShadowMap(RenderContext* pContext);
    void displayVisibilityBuffer(RenderContext* pContext);
    void runMainPass(RenderContext* pContext);
    void createVisualizationProgram();
    void createScene(const std::string& filename);
    void displayLoadSceneDialog();
    void setLightIndex(int32_t index);

    std::vector<CascadedShadowMaps::SharedPtr> mpCsmTech;
    Scene::SharedPtr mpScene;

    struct
    {
        FullScreenPass::UniquePtr pShadowMapProgram;
        GraphicsVars::SharedPtr pShadowMapProgramVars;
        FullScreenPass::UniquePtr pVisibilityBufferProgram;
        GraphicsVars::SharedPtr pVisibilityBufferProgramVars;
    } mShadowVisualizer;

    struct
    {
        GraphicsProgram::SharedPtr pProgram;
        GraphicsVars::SharedPtr pProgramVars;
    } mLightingPass;

    Sampler::SharedPtr mpLinearSampler = nullptr;

    SceneRenderer::SharedPtr mpRenderer;

    enum class DebugMode { None = 0, ShadowMap = 1, VisibilityBuffer = 2, Count = 3 };
    static const Gui::DropdownList skDebugModeList;
    struct Controls
    {
        bool updateShadowMap = true;
        uint32_t debugMode = (uint32_t)DebugMode::None;
        int32_t displayedCascade = 0;
        int32_t cascadeCount = 4;
        int32_t lightIndex = 0;
    };
    Controls mControls;


    struct ShadowOffsets
    {
        uint32_t displayedCascade;
    } mOffsets;  

    //non csm data in this cb so it can be sent as a single blob
    struct PerFrameCBData
    {
        //This is effectively a bool, but bool only takes up 1 byte which messes up setBlob
        glm::mat4 camVpAtLastCsmUpdate = glm::mat4();
        uint32_t visualizeCascades = 0u;
    } mPerFrameCBData;

    static const std::string skDefaultScene;
    glm::uvec2 mWindowDimensions;
    std::vector<Texture::SharedPtr> mpVisibilityBuffers;

    //Testing 
    void onInitializeTesting(SampleCallbacks* pSample) override;
    void onEndTestFrame(SampleCallbacks* pSample, SampleTest* pSampleTest) override;
    std::vector<uint32_t> mFilterFrames;
    std::vector<uint32_t>::iterator mFilterFramesIt;
};
