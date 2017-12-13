/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "API/LowLevel/LowLevelContextData.h"
#include "API/Vulkan/FalcorVK.h"
#include "API/Device.h"

namespace Falcor
{
    struct LowLevelContextApiData
    {
        FencedPool<VkCommandBuffer>::SharedPtr pCmdBufferAllocator;
        bool recordingCmds = false;
    };

    VkCommandBuffer createCommandBuffer(void* pUserData)
    {
        LowLevelContextData* pThis = (LowLevelContextData*)pUserData;
        VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
        cmdBufAllocateInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufAllocateInfo.commandPool        = pThis->getCommandAllocator();
        cmdBufAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufAllocateInfo.commandBufferCount = 1;
        VkCommandBuffer cmdBuf;
        vk_call(vkAllocateCommandBuffers(gpDevice->getApiHandle(), &cmdBufAllocateInfo, &cmdBuf));
        return cmdBuf;
    }

    LowLevelContextData::SharedPtr LowLevelContextData::create(LowLevelContextData::CommandQueueType type, CommandQueueHandle queue)
    {
        SharedPtr pThis = SharedPtr(new LowLevelContextData);
        pThis->mType = type;
        pThis->mpFence = GpuFence::create();
        pThis->mpQueue = queue;

        VkCommandPoolCreateInfo commandPoolCreateInfo{};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = gpDevice->getApiCommandQueueType(type);
        VkCommandPool pool;
        if (VK_FAILED(vkCreateCommandPool(gpDevice->getApiHandle(), &commandPoolCreateInfo, nullptr, &pool)))
        {
            logError("Could not create command pool");
            return nullptr;
        }
        pThis->mpAllocator = CommandAllocatorHandle::create(pool);
        pThis->mpApiData = new LowLevelContextApiData;
        pThis->mpApiData->pCmdBufferAllocator = FencedPool<VkCommandBuffer>::create(pThis->mpFence, createCommandBuffer, pThis.get());
        pThis->mpList = pThis->mpApiData->pCmdBufferAllocator->newObject();

        return pThis;
    }

    LowLevelContextData::~LowLevelContextData()
    {
        safe_delete(mpApiData);
    }

    void LowLevelContextData::reset()
    {
        if(mpApiData->recordingCmds == false)
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            beginInfo.pInheritanceInfo = nullptr;
            mpList = mpApiData->pCmdBufferAllocator->newObject();
            vk_call(vkBeginCommandBuffer(mpList, &beginInfo));
            mpApiData->recordingCmds = true;
        }
    }

    // Submit the recorded command buffers here. 
    void LowLevelContextData::flush()
    {
        mpApiData->recordingCmds = false;
        vk_call(vkEndCommandBuffer(mpList));
        VkSubmitInfo submitInfo = {};

        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mpList;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = nullptr;
        vk_call(vkQueueSubmit(mpQueue, 1, &submitInfo, nullptr));
        mpFence->gpuSignal(mpQueue);
        reset();    // Need to call vkBeginCommandBuffer()
    }
}
