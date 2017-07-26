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
#include "Raytracing/RTContext.h"

using namespace Falcor;
using namespace Falcor::RT;

class SimpleOptix : public Sample
{
public:
    SimpleOptix();
    ~SimpleOptix();

    void onLoad() override;
    void onFrameRender() override;
    void onShutdown() override;
    void onResizeSwapChain() override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void initUI();
    void loadScene();
    void loadSceneFromFile(const std::string filename);

    void onDataReload() override;

    // Shared 
    Scene::SharedPtr mpScene;
    Sampler::SharedPtr mpLinearSampler;

    DirectionalLight::SharedPtr mpDirLight;
    PointLight::SharedPtr mpPointLight;

    // Ray tracing routines
    RTContext::SharedPtr    mRTContext;

    RoutineHandle   mRTRaygenRtn;
    RoutineHandle   mRTShadingRtn;

    Fbo::SharedPtr  mpTargetFBO;

    bool        mRaytrace = true;
    bool        mSupersample = true;
    int         mBounces = 0;

    // Raster routines
    Program::SharedPtr              mpRasterShader;
    UniformBuffer::SharedPtr        mpPerFrameCB;
    RasterizerState::SharedPtr      mpCullRastState;
    DepthStencilState::SharedPtr    mpDepthTestDS;
    SceneRenderer::UniquePtr        mpRenderer;

    uint32_t                        mFrameCount = 0;

    float                           mExposure = 1.f;

    float mNearZ = 1e-2f;
    float mFarZ = 1e3f;
};