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
#include "API/LowLevel/GpuFence.h"
#include "API/Device.h"
#include "API/Vulkan/FalcorVK.h"

namespace Falcor
{
    // #VKTODO This entire class seems overly complicated. Need to make sure that there are no performance issues
    VkFence createFence()
    {
        VkFenceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        vkCreateFence(gpDevice->getApiHandle(), &info, nullptr, &fence);
        return fence;
    }

    void destroyFence(VkFence fence)
    {
        vkDestroyFence(gpDevice->getApiHandle(), fence, nullptr);
    }

    void resetFence(VkFence fence)
    {
        vkResetFences(gpDevice->getApiHandle(), 1, &fence);
    }

    VkSemaphore createSemaphore()
    {
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore sem;
        vkCreateSemaphore(gpDevice->getApiHandle(), &semaphoreInfo, nullptr, &sem);
        return sem;
    };

    void destroySemaphore(VkSemaphore semaphore)
    {
        vkDestroySemaphore(gpDevice->getApiHandle(), semaphore, nullptr);
    }

    struct FenceApiData
    {
        template<typename VkType, VkType(*createFunc)(void), void(*destroyFunc)(VkType), void(*resetFunc)(VkType)>
        class SmartQueue
        {
        public:
            ~SmartQueue()
            {
                popAllObjects();
                for (auto& o : freeObjects)
                {
                    destroyFunc(o);
                }
            }

            VkType getObject()
            {
                VkType object;
                if (freeObjects.size())
                {
                    object = freeObjects.back();
                    freeObjects.pop_back();
                    if(resetFunc) resetFunc(object);
                }
                else
                {
                    object = createFunc();
                }
                activeObjects.push_back(object);
                return object;
            }

            void popAllObjects()
            {
                freeObjects.insert(freeObjects.end(), activeObjects.begin(), activeObjects.end());
                activeObjects.clear();
            }

            void popFront()
            {
                freeObjects.push_back(activeObjects.front());
                activeObjects.pop_front();
            }

            void popFront(size_t count)
            {
                freeObjects.insert(freeObjects.end(), activeObjects.begin(), activeObjects.begin() + count);
                activeObjects.erase(activeObjects.begin(), activeObjects.begin() + count);
            }

            VkType front()
            {
                return activeObjects.front();
            }

            bool hasActiveObjects() const { return activeObjects.size() != 0; }

            std::deque<VkType>& getActiveObjects() { return activeObjects; }
            std::vector<VkType>& getFreeObjects() { return freeObjects; }
        private:
            std::deque<VkType> activeObjects;
            std::vector<VkType> freeObjects;
        };

        SmartQueue<VkFence, createFence, destroyFence, resetFence> fenceQueue;
        SmartQueue<VkSemaphore, createSemaphore, destroySemaphore, nullptr> semaphoreQueue;
        std::vector<VkSemaphore> semaphoreWaitList;
        uint64_t gpuValue = 0;
    };

    GpuFence::~GpuFence()
    {
        safe_delete(mpApiData);
    }

    GpuFence::SharedPtr GpuFence::create()
    {
        SharedPtr pFence = SharedPtr(new GpuFence());
        pFence->mpApiData = new FenceApiData;
        pFence->mCpuValue = 1;
        return pFence;
    }
    
    uint64_t GpuFence::gpuSignal(CommandQueueHandle pQueue)
    {
        mCpuValue++;
        VkFence fence = mpApiData->fenceQueue.getObject();
        VkSemaphore sem = mpApiData->semaphoreQueue.getObject();
        mpApiData->semaphoreWaitList.push_back(sem);

        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &sem;

        static const uint32_t waitThreshold = 20;
        static const uint32_t waitCount = 10;
        static const std::vector<VkPipelineStageFlags> waitStages(waitCount, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        if (mpApiData->semaphoreWaitList.size() > waitThreshold)
        {
            // #VKTODO syncGpu() is never actually called, we need to do some cleanup to make sure we don't allocate semaphors until we exhast the memory
            // We insert a wait here. We should make sure it doesn't actually stall the queue
            submit.pWaitDstStageMask = waitStages.data();
            submit.waitSemaphoreCount = waitCount;
            submit.pWaitSemaphores = mpApiData->semaphoreWaitList.data();
        }

        vk_call(vkQueueSubmit(pQueue, 1, &submit , fence));

        if (mpApiData->semaphoreWaitList.size() > waitThreshold)
        {
            mpApiData->semaphoreWaitList.erase(mpApiData->semaphoreWaitList.begin(), mpApiData->semaphoreWaitList.begin() + waitCount);
        }
        return mCpuValue - 1;
    }

    FenceHandle GpuFence::getApiHandle() const
    {
        auto fence = mpApiData->semaphoreQueue.getObject(); // #VKTODO Figure this out. It is implemented like this based on the internal usage in VkDevice.cpp, but might not be what the user expects
        mpApiData->semaphoreWaitList.push_back(fence);
        return fence;
    }

    void GpuFence::syncGpu(CommandQueueHandle pQueue)
    {
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = (uint32_t)mpApiData->semaphoreWaitList.size();
        submit.pWaitSemaphores = mpApiData->semaphoreWaitList.data();
        const std::vector<VkPipelineStageFlags> waitStages(submit.waitSemaphoreCount, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        submit.pWaitDstStageMask = waitStages.data();

        vk_call(vkQueueSubmit(pQueue, 1, &submit, nullptr));
        mpApiData->semaphoreWaitList.clear();
    }
    
    void releaseSemaphores(FenceApiData* pApiData)
    {
        size_t sems = pApiData->semaphoreQueue.getActiveObjects().size();
        size_t fences = pApiData->fenceQueue.getActiveObjects().size();
        assert(fences <= sems);
        size_t fenceDelta = sems - fences;
        // Make sure we don't release anything that's on the wait list
        size_t wait = pApiData->semaphoreWaitList.size();
        assert(wait <= sems);
        size_t waitDelta = sems - wait;
        size_t count = min(waitDelta, fenceDelta);
        pApiData->semaphoreQueue.popFront(count);
    }

    void GpuFence::syncCpu()
    {
        if (mpApiData->fenceQueue.hasActiveObjects() == false) return;

        auto& activeFences = mpApiData->fenceQueue.getActiveObjects();
        std::vector<VkFence> fenceVec(activeFences.begin(), activeFences.end());
        vk_call(vkWaitForFences(gpDevice->getApiHandle(), (uint32_t)fenceVec.size(), fenceVec.data(), true, UINT64_MAX));
        mpApiData->gpuValue += fenceVec.size();
        mpApiData->fenceQueue.popAllObjects();
        releaseSemaphores(mpApiData);  // Call this after popping the fences
    }

    uint64_t GpuFence::getGpuValue() const
    {
        auto& activeFences = mpApiData->fenceQueue.getActiveObjects();
        while (activeFences.size())
        {
            VkFence fence = activeFences.front();
            if (vkGetFenceStatus(gpDevice->getApiHandle(), fence) == VK_SUCCESS)
            {
                mpApiData->fenceQueue.popFront();
                mpApiData->gpuValue++;
            }
            else
            {
                break;
            }
        }
        releaseSemaphores(mpApiData);
        return mpApiData->gpuValue;
    }
}
