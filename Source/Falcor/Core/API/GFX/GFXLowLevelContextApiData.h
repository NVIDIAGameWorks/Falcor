/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

#if FALCOR_HAS_D3D12
#include "Core/API/Shared/D3D12DescriptorPool.h"
#endif

#include <slang-gfx.h>

namespace Falcor
{
    struct LowLevelContextApiData
    {
#if FALCOR_HAS_D3D12
        D3D12CommandListHandle mpD3D12CommandListHandle;
        D3D12CommandQueueHandle mpD3D12CommandQueueHandle;
#endif
        Slang::ComPtr<gfx::ICommandBuffer> pCommandBuffer;
        bool mIsCommandBufferOpen = false;

        gfx::IResourceCommandEncoder* getResourceCommandEncoder();
        gfx::IComputeCommandEncoder* getComputeCommandEncoder();
        gfx::IRenderCommandEncoder* getRenderCommandEncoder(gfx::IRenderPassLayout* renderPassLayout, gfx::IFramebuffer* framebuffer, bool& newEncoder);
        gfx::IRayTracingCommandEncoder* getRayTracingCommandEncoder();
        void closeEncoders();
    private:
        gfx::IFramebuffer* mpFramebuffer = nullptr;
        gfx::IRenderPassLayout* mpRenderPassLayout = nullptr;
        gfx::IResourceCommandEncoder* mpResourceCommandEncoder = nullptr;
        gfx::IComputeCommandEncoder* mpComputeCommandEncoder = nullptr;
        gfx::IRenderCommandEncoder* mpRenderCommandEncoder = nullptr;
        gfx::IRayTracingCommandEncoder* mpRayTracingCommandEncoder = nullptr;
    };
}
