/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "Core/API/Device.h"
#include "Core/API/LowLevelContextData.h"
#include "GFXDeviceApiData.h"
#include "GFXLowLevelContextApiData.h"

namespace Falcor
{
    LowLevelContextData::SharedPtr LowLevelContextData::create(CommandQueueType type, CommandQueueHandle queue)
    {
        return SharedPtr(new LowLevelContextData(type, queue));
    }

    LowLevelContextData::LowLevelContextData(CommandQueueType type, CommandQueueHandle queue)
        : mType(type)
        , mpQueue(queue)
    {
        mpFence = GpuFence::create();
        mpApiData = new LowLevelContextApiData;
        assert(mpFence && mpApiData);

        auto transientHeap = gpDevice->getApiData()->pTransientResourceHeaps[gpDevice->getCurrentBackBufferIndex()].get();
        mpApiData->pCommandBuffer = transientHeap->createCommandBuffer();
    }

    LowLevelContextData::~LowLevelContextData()
    {
        safe_delete(mpApiData);
    }

    void LowLevelContextData::flush()
    {
        mpApiData->closeEncoders();
        mpApiData->pCommandBuffer->close();
        mpQueue->executeCommandBuffers(1, mpApiData->pCommandBuffer.readRef(), mpFence->getApiHandle(), mpFence->externalSignal());
        auto transientHeap = gpDevice->getApiData()->pTransientResourceHeaps[gpDevice->getCurrentBackBufferIndex()].get();
        mpApiData->pCommandBuffer = transientHeap->createCommandBuffer();
    }

    void LowLevelContextApiData::closeEncoders()
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

    gfx::IResourceCommandEncoder* LowLevelContextApiData::getResourceCommandEncoder()
    {
        if (mpResourceCommandEncoder)
        {
            return mpResourceCommandEncoder;
        }
        closeEncoders();
        mpResourceCommandEncoder = pCommandBuffer->encodeResourceCommands();
        return mpResourceCommandEncoder;
    }

    gfx::IComputeCommandEncoder* LowLevelContextApiData::getComputeCommandEncoder()
    {
        if (mpComputeCommandEncoder)
        {
            return mpComputeCommandEncoder;
        }
        closeEncoders();
        mpComputeCommandEncoder = pCommandBuffer->encodeComputeCommands();
        return mpComputeCommandEncoder;
    }

    gfx::IRenderCommandEncoder* LowLevelContextApiData::getRenderCommandEncoder(gfx::IRenderPassLayout* renderPassLayout, gfx::IFramebuffer* framebuffer, bool& newEncoder)
    {
        if (mpRenderCommandEncoder && mpRenderPassLayout == renderPassLayout && mpFramebuffer == framebuffer)
        {
            newEncoder = false;
            return mpRenderCommandEncoder;
        }
        closeEncoders();
        mpRenderCommandEncoder = pCommandBuffer->encodeRenderCommands(renderPassLayout, framebuffer);
        mpRenderPassLayout = renderPassLayout;
        mpFramebuffer = framebuffer;
        newEncoder = true;
        return mpRenderCommandEncoder;
    }

    gfx::IRayTracingCommandEncoder* LowLevelContextApiData::getRayTracingCommandEncoder()
    {
        if (mpRayTracingCommandEncoder)
        {
            return mpRayTracingCommandEncoder;
        }
        closeEncoders();
        mpRayTracingCommandEncoder = pCommandBuffer->encodeRayTracingCommands();
        return mpRayTracingCommandEncoder;
    }
}
