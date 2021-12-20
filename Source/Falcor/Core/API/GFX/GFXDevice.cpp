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

#include "GFXDeviceApiData.h"
#include "GFXFormats.h"
#include "Core/Program/Program.h"

using namespace gfx;

namespace Falcor
{
    CommandQueueHandle Device::getCommandQueueHandle(LowLevelContextData::CommandQueueType type, uint32_t index) const
    {
        return mCmdQueues[(uint32_t)type][index];
    }

    ApiCommandQueueType Device::getApiCommandQueueType(LowLevelContextData::CommandQueueType type) const
    {
        switch (type)
        {
        case LowLevelContextData::CommandQueueType::Copy:
            return ApiCommandQueueType::Graphics;
        case LowLevelContextData::CommandQueueType::Compute:
            return ApiCommandQueueType::Graphics;
        case LowLevelContextData::CommandQueueType::Direct:
            return ApiCommandQueueType::Graphics;
        default:
            throw ArgumentError("Unknown command queue type");
        }
    }

    bool Device::getApiFboData(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat, ResourceHandle apiHandles[kSwapChainBuffersCount], uint32_t& currentBackBufferIndex)
    {
        return false;
    }

    void Device::toggleFullScreen(bool fullscreen)
    {
    }

    void Device::destroyApiObjects()
    {
        safe_delete(mpApiData);
    }

    void Device::apiPresent()
    {
        mpApiData->pSwapChain->present();
        mCurrentBackBufferIndex = (mCurrentBackBufferIndex + 1) % kSwapChainBuffersCount;
        mpApiData->pTransientResourceHeaps[mCurrentBackBufferIndex]->synchronizeAndReset();
    }

    bool Device::apiInit()
    {
        const uint32_t kTransientHeapConstantBufferSize = 16 * 1024 * 1024;

        DeviceApiData* pData = new DeviceApiData;
        mpApiData = pData;

        IDevice::Desc desc = {};
        desc.deviceType = DeviceType::DirectX12;
        desc.slang.slangGlobalSession = getSlangGlobalSession();
        if (SLANG_FAILED(gfxCreateDevice(&desc, pData->pDevice.writeRef()))) return false;

        for (uint32_t i = 0; i < kSwapChainBuffersCount; ++i)
        {
            ITransientResourceHeap::Desc transientHeapDesc = {};
            transientHeapDesc.constantBufferSize = kTransientHeapConstantBufferSize;
            if (SLANG_FAILED(pData->pDevice->createTransientResourceHeap(transientHeapDesc, pData->pTransientResourceHeaps[i].writeRef())))
            {
                return false;
            }
        }

        ICommandQueue::Desc queueDesc = {};
        queueDesc.type = ICommandQueue::QueueType::Graphics;
        if (SLANG_FAILED(pData->pDevice->createCommandQueue(queueDesc, pData->pQueue.writeRef()))) return false;
        return createSwapChain(mDesc.colorFormat);
    }

    bool Device::createSwapChain(ResourceFormat colorFormat)
    {
        ISwapchain::Desc desc = { };
        desc.format = getGFXFormat(colorFormat);
        desc.imageCount = kSwapChainBuffersCount;
        auto clientSize = mpWindow->getClientAreaSize();
        desc.width = clientSize.x;
        desc.height = clientSize.y;
        desc.enableVSync = mDesc.enableVsync;
        desc.queue = mpApiData->pQueue.get();
        return !SLANG_FAILED(mpApiData->pDevice->createSwapchain(desc, gfx::WindowHandle::FromHwnd(mpWindow->getApiHandle()), mpApiData->pSwapChain.writeRef()));
    }

    void Device::apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat)
    {
        assert(mpApiData->pSwapChain && getGFXFormat(colorFormat) == mpApiData->pSwapChain->getDesc().format);
        mpApiData->pSwapChain->resize(width, height);
    }

    bool Device::isWindowOccluded() const
    {
        return false;
    }
}
