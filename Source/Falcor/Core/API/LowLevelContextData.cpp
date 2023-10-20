/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "LowLevelContextData.h"
#include "Device.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"

#if FALCOR_HAS_CUDA
#include "Utils/CudaUtils.h"
#endif

#include <slang-gfx.h>

namespace Falcor
{
LowLevelContextData::LowLevelContextData(Device* pDevice, gfx::ICommandQueue* pQueue) : mpDevice(pDevice), mpGfxCommandQueue(pQueue)
{
    mpFence = mpDevice->createFence();
    mpFence->breakStrongReferenceToDevice();

#if FALCOR_HAS_CUDA
    // GFX currently doesn't support shared fences on Vulkan.
    if (mpDevice->getType() == Device::Type::D3D12)
    {
        mpDevice->initCudaDevice();
        mpCudaFence = mpDevice->createFence(true);
        mpCudaFence->breakStrongReferenceToDevice();
        mpCudaSemaphore = make_ref<cuda_utils::ExternalSemaphore>(mpCudaFence);
    }
#endif

    openCommandBuffer();
}

LowLevelContextData::~LowLevelContextData()
{
    if (mIsCommandBufferOpen)
    {
        closeCommandBuffer();
    }
}

NativeHandle LowLevelContextData::getCommandQueueNativeHandle() const
{
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(mpGfxCommandQueue->getNativeHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(reinterpret_cast<ID3D12CommandQueue*>(gfxNativeHandle.handleValue));
#endif
#if FALCOR_HAS_VULKAN
    if (mpDevice->getType() == Device::Type::Vulkan)
        return NativeHandle(reinterpret_cast<VkQueue>(gfxNativeHandle.handleValue));
#endif
    return {};
}

NativeHandle LowLevelContextData::getCommandBufferNativeHandle() const
{
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(mGfxCommandBuffer->getNativeHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(reinterpret_cast<ID3D12GraphicsCommandList*>(gfxNativeHandle.handleValue));
#endif
#if FALCOR_HAS_VULKAN
    if (mpDevice->getType() == Device::Type::Vulkan)
        return NativeHandle(reinterpret_cast<VkCommandBuffer>(gfxNativeHandle.handleValue));
#endif
    return {};
}

void LowLevelContextData::closeCommandBuffer()
{
    mIsCommandBufferOpen = false;
    closeEncoders();
    mGfxCommandBuffer->close();
}

void LowLevelContextData::openCommandBuffer()
{
    mIsCommandBufferOpen = true;
    FALCOR_GFX_CALL(mpDevice->getCurrentTransientResourceHeap()->createCommandBuffer(mGfxCommandBuffer.writeRef()));
    mpCommandBuffer = mGfxCommandBuffer.get();
}

void LowLevelContextData::submitCommandBuffer()
{
    closeCommandBuffer();
    mpGfxCommandQueue->executeCommandBuffers(1, mGfxCommandBuffer.readRef(), mpFence->getGfxFence(), mpFence->updateSignaledValue());
    openCommandBuffer();
}

gfx::IResourceCommandEncoder* LowLevelContextData::getResourceCommandEncoder()
{
    if (mpResourceCommandEncoder)
    {
        return mpResourceCommandEncoder;
    }
    if (mpComputeCommandEncoder)
    {
        return mpComputeCommandEncoder;
    }
    if (mpRayTracingCommandEncoder)
    {
        return mpRayTracingCommandEncoder;
    }
    closeEncoders();
    mpResourceCommandEncoder = mpCommandBuffer->encodeResourceCommands();
    return mpResourceCommandEncoder;
}

gfx::IComputeCommandEncoder* LowLevelContextData::getComputeCommandEncoder()
{
    if (mpComputeCommandEncoder)
    {
        return mpComputeCommandEncoder;
    }
    closeEncoders();
    mpComputeCommandEncoder = mpCommandBuffer->encodeComputeCommands();
    return mpComputeCommandEncoder;
}

gfx::IRenderCommandEncoder* LowLevelContextData::getRenderCommandEncoder(
    gfx::IRenderPassLayout* renderPassLayout,
    gfx::IFramebuffer* framebuffer,
    bool& newEncoder
)
{
    if (mpRenderCommandEncoder && mpRenderPassLayout == renderPassLayout && mpFramebuffer == framebuffer)
    {
        newEncoder = false;
        return mpRenderCommandEncoder;
    }
    closeEncoders();
    mpRenderCommandEncoder = mpCommandBuffer->encodeRenderCommands(renderPassLayout, framebuffer);
    mpRenderPassLayout = renderPassLayout;
    mpFramebuffer = framebuffer;
    newEncoder = true;
    return mpRenderCommandEncoder;
}

gfx::IRayTracingCommandEncoder* LowLevelContextData::getRayTracingCommandEncoder()
{
    if (mpRayTracingCommandEncoder)
    {
        return mpRayTracingCommandEncoder;
    }
    closeEncoders();
    mpRayTracingCommandEncoder = mpCommandBuffer->encodeRayTracingCommands();
    return mpRayTracingCommandEncoder;
}

void LowLevelContextData::closeEncoders()
{
    if (mpResourceCommandEncoder)
    {
        mpResourceCommandEncoder->endEncoding();
        mpResourceCommandEncoder = nullptr;
    }
    if (mpRenderCommandEncoder)
    {
        mpRenderCommandEncoder->endEncoding();
        mpRenderCommandEncoder = nullptr;
    }
    if (mpComputeCommandEncoder)
    {
        mpComputeCommandEncoder->endEncoding();
        mpComputeCommandEncoder = nullptr;
    }
    if (mpRayTracingCommandEncoder)
    {
        mpRayTracingCommandEncoder->endEncoding();
        mpRayTracingCommandEncoder = nullptr;
    }
}

void LowLevelContextData::beginDebugEvent(const char* name)
{
    float blackColor[3] = {0.0f, 0.0f, 0.0f};
    getResourceCommandEncoder()->beginDebugEvent(name, blackColor);
}

void LowLevelContextData::endDebugEvent()
{
    getResourceCommandEncoder()->endDebugEvent();
}
} // namespace Falcor
