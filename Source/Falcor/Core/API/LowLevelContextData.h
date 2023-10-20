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
#pragma once
#include "Fence.h"
#include "Handles.h"
#include "NativeHandle.h"
#include "Core/Macros.h"

namespace Falcor
{

namespace cuda_utils
{
class ExternalSemaphore;
}

class FALCOR_API LowLevelContextData
{
public:
    /**
     * Constructor. Throws an exception if creation failed.
     * @param[in] pDevice Device.
     * @param[in] pQueue Command queue.
     */
    LowLevelContextData(Device* pDevice, gfx::ICommandQueue* pQueue);
    ~LowLevelContextData();

    gfx::ICommandQueue* getGfxCommandQueue() const { return mpGfxCommandQueue; }
    gfx::ICommandBuffer* getGfxCommandBuffer() const { return mGfxCommandBuffer; }

    /**
     * Returns the native API handle for the command queue:
     * - D3D12: ID3D12CommandQueue*
     * - Vulkan: VkQueue (Vulkan)
     */
    NativeHandle getCommandQueueNativeHandle() const;

    /**
     * Returns the native API handle for the command buffer:
     * - D3D12: ID3D12GraphicsCommandList*
     * - Vulkan: VkCommandBuffer
     */
    NativeHandle getCommandBufferNativeHandle() const;

    const ref<Fence>& getFence() const { return mpFence; }

#if FALCOR_HAS_CUDA
    const ref<Fence>& getCudaFence() const { return mpCudaFence; }
    const ref<cuda_utils::ExternalSemaphore>& getCudaSemaphore() const { return mpCudaSemaphore; }
#endif

    void closeCommandBuffer();
    void openCommandBuffer();
    void submitCommandBuffer();

    gfx::IResourceCommandEncoder* getResourceCommandEncoder();
    gfx::IComputeCommandEncoder* getComputeCommandEncoder();
    gfx::IRenderCommandEncoder* getRenderCommandEncoder(
        gfx::IRenderPassLayout* renderPassLayout,
        gfx::IFramebuffer* framebuffer,
        bool& newEncoder
    );
    gfx::IRayTracingCommandEncoder* getRayTracingCommandEncoder();
    void closeEncoders();

    void beginDebugEvent(const char* name);
    void endDebugEvent();

private:
    Device* mpDevice;
    gfx::ICommandQueue* mpGfxCommandQueue;
    Slang::ComPtr<gfx::ICommandBuffer> mGfxCommandBuffer;
    ref<Fence> mpFence;

#if FALCOR_HAS_CUDA
    ref<Fence> mpCudaFence;
    ref<cuda_utils::ExternalSemaphore> mpCudaSemaphore;
#endif

    gfx::ICommandBuffer* mpCommandBuffer = nullptr;
    bool mIsCommandBufferOpen = false;

    gfx::IFramebuffer* mpFramebuffer = nullptr;
    gfx::IRenderPassLayout* mpRenderPassLayout = nullptr;
    gfx::IResourceCommandEncoder* mpResourceCommandEncoder = nullptr;
    gfx::IComputeCommandEncoder* mpComputeCommandEncoder = nullptr;
    gfx::IRenderCommandEncoder* mpRenderCommandEncoder = nullptr;
    gfx::IRayTracingCommandEncoder* mpRayTracingCommandEncoder = nullptr;
};
} // namespace Falcor
