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
#include <Falcor.h>
#include <Cuda/CudaModule.h>
#include <vector>

using namespace Falcor;


class SimpleCuda : public Sample
{
public:
    void onLoad() override;
    void onFrameRender() override;
    void onShutdown() override;
    void onResizeSwapChain() override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onDataReload() override;

private:
    struct VertexBufferInfo 
    {
        uint32_t			numVertices = 0;
        uint32_t			stride = 0;
        ResourceFormat		attribFormat;
        size_t				attribSize = 0;
        uint32_t			attribOffset = 0;
    };

    Fbo::SharedPtr	            mDisplayFbo = nullptr;
    Texture::SharedConstPtr     mDisplayTexture;

    std::vector< Buffer::SharedConstPtr > mVertexBuffers;
    std::vector< VertexBufferInfo >		  mVertexBuffersInfo;

    Camera::SharedPtr           mpCamera;
    Model::SharedPtr            mpModel;

    Program::SharedPtr          mpProgram = nullptr;
    Sampler::SharedPtr          mpLinearSampler = nullptr;
    ModelRenderer               mModelRenderer;

    ModelViewCameraController   mModelViewCameraController;

    float mNearZ;
    float mFarZ;

    std::shared_ptr<Cuda::CudaModule>		mCudaModule = nullptr;
    std::shared_ptr<Cuda::CudaModule>		mCudaModuleVertex = nullptr;

    void initUI();
    void loadModelFromFile(const std::string& filename);
    void resetCamera();


};
