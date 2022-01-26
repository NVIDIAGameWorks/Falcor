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
        for (uint32_t i = 0; i < kSwapChainBuffersCount; i++)
        {
            Slang::ComPtr<gfx::ITextureResource> imageHandle;
            HRESULT hr = mpApiData->pSwapChain->getImage(i, imageHandle.writeRef());
            apiHandles[i] = imageHandle.get();
            if (FAILED(hr))
            {
                logError("Failed to get back-buffer " + std::to_string(i) + " from the swap-chain", hr);
                return false;
            }
        }
        currentBackBufferIndex = this->mCurrentBackBufferIndex;
        return true;
    }

    void Device::toggleFullScreen(bool fullscreen)
    {
    }

    void Device::destroyApiObjects()
    {
        safe_delete(mpApiData);
    }

    gfx::ITransientResourceHeap* Device::getCurrentTransientResourceHeap()
    {
        return mpApiData->pTransientResourceHeaps[mCurrentBackBufferIndex].get();
    }

    void Device::apiPresent()
    {
        mpApiData->pSwapChain->present();
        mCurrentBackBufferIndex = mpApiData->pSwapChain->acquireNextImage();
        this->mpRenderContext->getLowLevelData()->closeCommandBuffer();
        getCurrentTransientResourceHeap()->synchronizeAndReset();
        this->mpRenderContext->getLowLevelData()->openCommandBuffer();
    }

    Device::SupportedFeatures querySupportedFeatures(DeviceHandle pDevice)
    {
        Device::SupportedFeatures result = Device::SupportedFeatures::None;
        if (pDevice->hasFeature("ray-tracing"))
        {
            result |= Device::SupportedFeatures::Raytracing;
        }
        if (pDevice->hasFeature("ray-query"))
        {
            result |= Device::SupportedFeatures::RaytracingTier1_1;
        }
        if (pDevice->hasFeature("conservative-rasterization-3"))
        {
            result |= Device::SupportedFeatures::ConservativeRasterizationTier3;
        }
        if (pDevice->hasFeature("conservative-rasterization-2"))
        {
            result |= Device::SupportedFeatures::ConservativeRasterizationTier2;
        }
        if (pDevice->hasFeature("conservative-rasterization-1"))
        {
            result |= Device::SupportedFeatures::ConservativeRasterizationTier1;
        }
        if (pDevice->hasFeature("rasterizer-ordered-views"))
        {
            result |= Device::SupportedFeatures::RasterizerOrderedViews;
        }

        if (pDevice->hasFeature("programmable-sample-positions-2"))
        {
            result |= Device::SupportedFeatures::ProgrammableSamplePositionsFull;
        }
        else if (pDevice->hasFeature("programmable-sample-positions-1"))
        {
            result |= Device::SupportedFeatures::ProgrammableSamplePositionsPartialOnly;
        }

        if (pDevice->hasFeature("barycentrics"))
        {
            result |= Device::SupportedFeatures::Barycentrics;
        }
        return result;
    }

    Device::ShaderModel querySupportedShaderModel(DeviceHandle pDevice)
    {
        struct SMLevel { const char* name; Device::ShaderModel level; };
        const SMLevel levels[] =
        {
            {"sm_6_7", Device::ShaderModel::SM6_7},
            {"sm_6_6", Device::ShaderModel::SM6_6},
            {"sm_6_5", Device::ShaderModel::SM6_5},
            {"sm_6_4", Device::ShaderModel::SM6_4},
            {"sm_6_3", Device::ShaderModel::SM6_3},
            {"sm_6_2", Device::ShaderModel::SM6_2},
            {"sm_6_1", Device::ShaderModel::SM6_1},
            {"sm_6_0", Device::ShaderModel::SM6_0}
        };
        for (auto level : levels)
        {
            if (pDevice->hasFeature(level.name))
            {
                return level.level;
            }
        }
        return Device::ShaderModel::Unknown;
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

        mApiHandle = pData->pDevice;

        mSupportedFeatures = querySupportedFeatures(mApiHandle);
        mSupportedShaderModel = querySupportedShaderModel(mApiHandle);

        for (uint32_t i = 0; i < kSwapChainBuffersCount; ++i)
        {
            ITransientResourceHeap::Desc transientHeapDesc = {};
            transientHeapDesc.flags = ITransientResourceHeap::Flags::AllowResizing;
            transientHeapDesc.constantBufferSize = kTransientHeapConstantBufferSize;
            transientHeapDesc.samplerDescriptorCount = 2048;
            transientHeapDesc.uavDescriptorCount = 1000000;
            transientHeapDesc.srvDescriptorCount = 1000000;
            transientHeapDesc.constantBufferDescriptorCount = 1000000;
            transientHeapDesc.accelerationStructureDescriptorCount = 1000000;
            if (SLANG_FAILED(pData->pDevice->createTransientResourceHeap(transientHeapDesc, pData->pTransientResourceHeaps[i].writeRef())))
            {
                return false;
            }
        }

        ICommandQueue::Desc queueDesc = {};
        queueDesc.type = ICommandQueue::QueueType::Graphics;
        if (SLANG_FAILED(pData->pDevice->createCommandQueue(queueDesc, pData->pQueue.writeRef()))) return false;
        for (auto& queue : mCmdQueues)
        {
            queue.push_back(pData->pQueue);
        }
        return createSwapChain(mDesc.colorFormat);
    }

    bool Device::createSwapChain(ResourceFormat colorFormat)
    {
        ISwapchain::Desc desc = { };
        desc.format = getGFXFormat(srgbToLinearFormat(colorFormat));
        desc.imageCount = kSwapChainBuffersCount;
        auto clientSize = mpWindow->getClientAreaSize();
        desc.width = clientSize.x;
        desc.height = clientSize.y;
        desc.enableVSync = mDesc.enableVsync;
        desc.queue = mpApiData->pQueue.get();
        if (SLANG_FAILED(mpApiData->pDevice->createSwapchain(desc, gfx::WindowHandle::FromHwnd(mpWindow->getApiHandle()), mpApiData->pSwapChain.writeRef())))
        {
            return false;
        }
        mCurrentBackBufferIndex = mpApiData->pSwapChain->acquireNextImage();
        return true;
    }

    void Device::apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat)
    {
        FALCOR_ASSERT(mpApiData->pSwapChain && getGFXFormat(srgbToLinearFormat(colorFormat)) == mpApiData->pSwapChain->getDesc().format);
        FALCOR_ASSERT(SLANG_SUCCEEDED(mpApiData->pSwapChain->resize(width, height)));
        mCurrentBackBufferIndex = mpApiData->pSwapChain->acquireNextImage();
    }

    bool Device::isWindowOccluded() const
    {
        return false;
    }
}
