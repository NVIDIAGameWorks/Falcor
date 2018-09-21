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
#include "SampleTest.h"

using namespace Falcor;

class LightProbeViewer : public Renderer
{
public:
    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;
    void onDataReload(SampleCallbacks* pSample) override;
    void onDroppedFile(SampleCallbacks* pSample, const std::string& filename) override;

private:
    void resetCamera();
    void updateLightProbe(LightProbe::SharedPtr pLightProbe);
    void renderInfoText(SampleCallbacks* pSample);

    static const std::string kEnvMapName;

    uint32_t mDiffuseSamples = LightProbe::kDefaultDiffSamples;
    uint32_t mSpecSamples = LightProbe::kDefaultSpecSamples;
    int32_t mSpecMip = 0;

    enum class Viewport
    {
        Scene,
        Orig,
        Diffuse,
        Specular,
        Count
    };

    // Viewport coordinates
    uvec4 mMainRect;
    std::array<uvec4, (uint32_t)Viewport::Count> mRects;
    Viewport mSelectedView;

    Camera::SharedPtr mpCamera;
    Model::SharedPtr mpModel = nullptr;
    SixDoFCameraController mCameraController;

    SkyBox::SharedPtr mpSkyBox;

    Scene::SharedPtr mpScene;
    SceneRenderer::SharedPtr mpSceneRenderer;
    LightProbe::SharedPtr mpLightProbe;

    GraphicsProgram::SharedPtr mpProgram = nullptr;
    GraphicsVars::SharedPtr mpVars = nullptr;
    GraphicsState::SharedPtr mpState = nullptr;

    Sampler::SharedPtr mpLinearSampler = nullptr;
    DepthStencilState::SharedPtr mpDepthState = nullptr;
    RasterizerState::SharedPtr mpRasterizerState = nullptr;
};