/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "CudaInterop.h"
#include "CopySurface.h"

void CudaInterop::onLoad(RenderContext* pRenderContext)
{
    // Create our input and output textures
    mpInputTex = Texture::createFromFile("smoke-puff.png", false, false, ResourceBindFlags::Shared);
    mWidth = mpInputTex->getWidth();
    mHeight = mpInputTex->getHeight();
    mpOutputTex = Texture::create2D(mWidth, mHeight, mpInputTex->getFormat(), 1, 1, nullptr, ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource);

    // Define our usage flags and then map the textures to CUDA surfaces. Surface values of 0
    // indicate an error during mapping. We need to cache mInputSurf and mOutputSurf as
    // mapTextureToSurface() can only be called once per resource.
    uint32_t usageFlags = cudaArrayColorAttachment;
    mInputSurf = FalcorCUDA::mapTextureToSurface(mpInputTex, usageFlags);
    if (mInputSurf == 0)
    {
        logError("Input texture to surface mapping failed");
        return;
    }
    mOutputSurf = FalcorCUDA::mapTextureToSurface(mpOutputTex, usageFlags);
    if (mOutputSurf == 0)
    {
        logError("Output texture to surface mapping failed");
        return;
    }
}

void CudaInterop::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    // Call the CUDA kernel
    uint32_t format = (getFormatType(mpInputTex->getFormat()) == FormatType::Float) ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    launchCopySurface(mInputSurf, mOutputSurf, mWidth, mHeight, format);
    pRenderContext->blit(mpOutputTex->getSRV(), pTargetFbo->getRenderTargetView(0));
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    // Initializes the CUDA driver API, which is required prior to any API calls.
    if (!FalcorCUDA::initCUDA())
    {
        logError("CUDA driver API initialization failed");
        return -1;
    }
    CudaInterop::UniquePtr pRenderer = std::make_unique<CudaInterop>();
    SampleConfig config;
    config.windowDesc.title = "Falcor-Cuda Interop";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pRenderer);
    return 0;
}
