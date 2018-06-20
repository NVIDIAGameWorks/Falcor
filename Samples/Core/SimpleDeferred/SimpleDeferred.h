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

class SimpleDeferred : public Renderer
{
public:
    ~SimpleDeferred();

    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onShutdown(SampleCallbacks* pSample) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;
private:
    void loadModel(Fbo* pTargetFbo);
    void loadModelFromFile(const std::string& filename, Fbo* pTargetFbo);
    void resetCamera();
    void renderModelUiElements(Gui* pGui);

    Model::SharedPtr mpModel = nullptr;
    ModelViewCameraController mModelViewCameraController;
    FirstPersonCameraController mFirstPersonCameraController;
    Sampler::SharedPtr mpLinearSampler;

    GraphicsProgram::SharedPtr mpDeferredPassProgram;
    GraphicsVars::SharedPtr mpDeferredVars;

    GraphicsVars::SharedPtr mpLightingVars;
    FullScreenPass::UniquePtr mpLightingPass;

    float mAspectRatio = 0;

    enum
    {
        ModelViewCamera,
        FirstPersonCamera
    } mCameraType = ModelViewCamera;

    CameraController& getActiveCameraController();

    Camera::SharedPtr mpCamera;

    bool mAnimate = false;
    bool mGenerateTangentSpace = true;
    glm::vec3 mAmbientIntensity = glm::vec3(0.1f, 0.1f, 0.1f);

    uint32_t mActiveAnimationID = sBindPoseAnimationID;
    static const uint32_t sBindPoseAnimationID = (uint32_t)-1;

    RasterizerState::SharedPtr mpCullRastState[3]; // 0 = no culling, 1 = backface culling, 2 = frontface culling
    uint32_t mCullMode = 1;

    enum : uint32_t
    {
        Disabled = 0,
        ShowPositions,
        ShowNormals,
        ShowAlbedo,
        ShowLighting
    } mDebugMode = Disabled;

    DepthStencilState::SharedPtr mpNoDepthDS;
    DepthStencilState::SharedPtr mpDepthTestDS;
    BlendState::SharedPtr mpOpaqueBS;

    // G-Buffer
    Fbo::SharedPtr mpGBufferFbo;

    DirectionalLight::SharedPtr mpDirLight;
    PointLight::SharedPtr mpPointLight;

    float mNearZ = 1e-2f;
    float mFarZ = 1e3f;

    static const std::string skDefaultModel;

    //testing
    void onInitializeTesting(SampleCallbacks* pSample) override;
    void onEndTestFrame(SampleCallbacks* pSample, SampleTest* pSampleTest) override;
    std::vector<uint32_t> mChangeModeFrames;
    std::vector<uint32_t>::iterator mChangeModeIt;
};