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

class ShaderBuffersSample : public Renderer
{
public:
    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;
    void onDataReload(SampleCallbacks* pSample) override;

private:
    GraphicsState::SharedPtr mpGraphicsState;
    ComputeProgram::SharedPtr mpComputeProgram;
    ComputeState::SharedPtr mpComputeState;
    ComputeVars::SharedPtr mpComputeVars;
    StructuredBuffer::SharedPtr mpLightBuffer;

    GraphicsProgram::SharedPtr mpProgram;
    GraphicsVars::SharedPtr mpProgramVars;
    Model::SharedPtr mpModel;
    Vao::SharedConstPtr mpVao;
    uint32_t mIndexCount = 0;
    Buffer::SharedPtr mpInvocationsBuffer;
    StructuredBuffer::SharedPtr mpRWBuffer;
    StructuredBuffer::SharedPtr mpAppendLightData;
    TypedBuffer<vec3>::SharedPtr mpSurfaceColorBuffer;

    bool mCountPixelShaderInvocations = false;

    Camera::SharedPtr mpCamera;
    ModelViewCameraController mCameraController;

    struct Light
    {
        glm::vec3 worldDir = glm::vec3(0, -1, 0);
        glm::vec3 intensity = glm::vec3(0.6f, 0.8f, 0.8f);
    };

    Light mLightData;

    glm::vec3 mSurfaceColor = glm::vec3(0.36f,0.87f,0.52f);
    Vao::SharedConstPtr getVao();

    static const std::string skDefaultModel;
};