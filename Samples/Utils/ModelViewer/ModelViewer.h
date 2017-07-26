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
#include "Utils/Picking/Picking.h"

using namespace Falcor;

class ModelViewer : public Sample
{
public:
    void onLoad() override;
    void onFrameRender() override;
    void onShutdown() override;
    void onResizeSwapChain() override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onGuiRender() override;

private:
    void loadModel();
    void saveModel();
    void deleteCulledMeshes();

    void loadModelFromFile(const std::string& Filename);
    void resetCamera();
    void renderModelUI();
    void setModelString(bool isAfterCull, float LoadTime);

    Model::SharedPtr mpModel = nullptr;
    ModelViewCameraController mModelViewCameraController;
    FirstPersonCameraController mFirstPersonCameraController;
    SixDoFCameraController m6DoFCameraController;

    bool mUseTriLinearFiltering = true;
    Sampler::SharedPtr mpPointSampler = nullptr;
    Sampler::SharedPtr mpLinearSampler = nullptr;

    GraphicsProgram::SharedPtr mpProgram = nullptr;
    GraphicsVars::SharedPtr mpProgramVars = nullptr;
    GraphicsState::SharedPtr mpGraphicsState = nullptr;

    enum
    {
        ModelViewCamera,
        FirstPersonCamera,
        SixDoFCamera
    } mCameraType = ModelViewCamera;

    CameraController& getActiveCameraController();

    Camera::SharedPtr mpCamera;

    bool mDrawWireframe = false;
    bool mAnimate = false;
    bool mGenerateTangentSpace = true;
    glm::vec3 mAmbientIntensity = glm::vec3(0.1f, 0.1f, 0.1f);

    uint32_t mActiveAnimationID = kBindPoseAnimationID;
    static const uint32_t kBindPoseAnimationID = AnimationController::kBindPoseAnimationId;

    RasterizerState::SharedPtr mpWireframeRS = nullptr;
    RasterizerState::SharedPtr mpCullRastState[3]; // 0 = no culling, 1 = backface culling, 2 = frontface culling
    uint32_t mCullMode = 1;

    DepthStencilState::SharedPtr mpNoDepthDS = nullptr;
    DepthStencilState::SharedPtr mpDepthTestDS = nullptr;

    DirectionalLight::SharedPtr mpDirLight;
    PointLight::SharedPtr mpPointLight;

    std::string mModelString;

    float mNearZ;
    float mFarZ;
};