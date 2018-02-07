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

class Shadows : public SampleTest
{
public:
    void onLoad() override;
    void onFrameRender() override;
    void onShutdown() override;
    void onResizeSwapChain() override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void onGuiRender() override;
    void displayShadowMap();
    void runMainPass();
    void createVisualizationProgram();
    void createScene(const std::string& filename);
    void displayLoadSceneDialog();
    void setLightIndex(int32_t index);

    std::vector<CascadedShadowMaps::UniquePtr> mpCsmTech;
    Scene::SharedPtr mpScene;

    struct
    {
        FullScreenPass::UniquePtr pProgram;
        GraphicsVars::SharedPtr pProgramVars;
    } mShadowVisualizer;

    struct
    {
        GraphicsProgram::SharedPtr pProgram;
        GraphicsVars::SharedPtr pProgramVars;
    } mLightingPass;

    Sampler::SharedPtr mpLinearSampler = nullptr;

    SceneRenderer::SharedPtr mpRenderer;

    struct Controls
    {
        bool updateShadowMap = true;
        bool showShadowMap = false;
        int32_t displayedCascade = 0;
        int32_t cascadeCount = 4;
        int32_t lightIndex = 0;
    };
    Controls mControls;

    struct ShadowOffsets
    {
        uint32_t visualizeCascades;
        uint32_t displayedCascade;
    } mOffsets;  

    //non csm data in this cb so it can be sent as a single blob
    struct PerFrameCBData
    {
        //This is effectively a bool, but bool only takes up 1 byte which messes up setBlob
        uint32_t visualizeCascades = 0u;
        uint32_t padding1 = 0;
        uint64_t padding2 = 0;
        glm::mat4 camVpAtLastCsmUpdate = glm::mat4();
    } mPerFrameCBData;

	const std::string mkDefaultScene = "Arcade\\Arcade.fscene";

    //Testing 
    void onInitializeTesting() override;
    void onEndTestFrame() override;
    std::vector<uint32_t> mFilterFrames;
    std::vector<uint32_t>::iterator mFilterFramesIt;
};
