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
#include "Framework.h"
#include "Sample.h"
#include "API/Device.h"
#include "API/LowLevel/GpuFence.h"

namespace Falcor
{
    struct DeviceApiData
    {
        IDXGIFactory4Ptr pDxgiFactory = nullptr;
        IDXGISwapChain3Ptr pSwapChain = nullptr;
        bool isWindowOccluded = false;
    };

    void d3dTraceHR(const std::string& msg, HRESULT hr)
    {
        char hr_msg[512];
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, hr, 0, hr_msg, ARRAYSIZE(hr_msg), nullptr);

        std::string error_msg = msg + ".\nError! " + hr_msg;
        logError(error_msg);
    }

    D3D_FEATURE_LEVEL getD3DFeatureLevel(uint32_t majorVersion, uint32_t minorVersion)
    {
        if (majorVersion == 12)
        {
            switch (minorVersion)
            {
            case 0:
                return D3D_FEATURE_LEVEL_12_0;
            case 1:
                return D3D_FEATURE_LEVEL_12_1;
            }
        }
        else if (majorVersion == 11)
        {
            switch (minorVersion)
            {
            case 0:
                return D3D_FEATURE_LEVEL_11_1;
            case 1:
                return D3D_FEATURE_LEVEL_11_0;
            }
        }
        else if (majorVersion == 10)
        {
            switch (minorVersion)
            {
            case 0:
                return D3D_FEATURE_LEVEL_10_0;
            case 1:
                return D3D_FEATURE_LEVEL_10_1;
            }
        }
        else if (majorVersion == 9)
        {
            switch (minorVersion)
            {
            case 1:
                return D3D_FEATURE_LEVEL_9_1;
            case 2:
                return D3D_FEATURE_LEVEL_9_2;
            case 3:
                return D3D_FEATURE_LEVEL_9_3;
            }
        }
        return (D3D_FEATURE_LEVEL)0;
    }

    IDXGISwapChain3Ptr createDxgiSwapChain(IDXGIFactory4* pFactory, const Window* pWindow, ID3D12CommandQueue* pCommandQueue, ResourceFormat colorFormat)
    {
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
        swapChainDesc.BufferCount = kDefaultSwapChainBuffers;
        swapChainDesc.Width = pWindow->getClientAreaWidth();
        swapChainDesc.Height = pWindow->getClientAreaHeight();
        // Flip mode doesn't support SRGB formats, so we strip them down when creating the resource. We will create the RTV as SRGB instead.
        // More details at the end of https://msdn.microsoft.com/en-us/library/windows/desktop/bb173064.aspx
        swapChainDesc.Format = getDxgiFormat(srgbToLinearFormat(colorFormat));
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.SampleDesc.Count = 1;

        // CreateSwapChainForHwnd() doesn't accept IDXGISwapChain3 (Why MS? Why?)
        MAKE_SMART_COM_PTR(IDXGISwapChain1);
        IDXGISwapChain1Ptr pSwapChain;

        HRESULT hr = pFactory->CreateSwapChainForHwnd(pCommandQueue, pWindow->getApiHandle(), &swapChainDesc, nullptr, nullptr, &pSwapChain);
        if (FAILED(hr))
        {
            d3dTraceHR("Failed to create the swap-chain", hr);
            return false;
        }

        IDXGISwapChain3Ptr pSwapChain3;
        d3d_call(pSwapChain->QueryInterface(IID_PPV_ARGS(&pSwapChain3)));
        return pSwapChain3;
    }

    ID3D12DevicePtr createDevice(IDXGIFactory4* pFactory, D3D_FEATURE_LEVEL featureLevel, Device::Desc::CreateDeviceFunc createFunc, bool& rgb32FSupported)
    {
        // Find the HW adapter
        IDXGIAdapter1Ptr pAdapter;
        ID3D12DevicePtr pDevice;

        for (uint32_t i = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(i, &pAdapter); i++)
        {
            DXGI_ADAPTER_DESC1 desc;
            pAdapter->GetDesc1(&desc);

            // Skip SW adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                continue;
            }

            // Try and create a D3D12 device
            if (createFunc)
            {
                pDevice = createFunc(pAdapter, featureLevel);
                if (pDevice) return pDevice;
            }
            else if (D3D12CreateDevice(pAdapter, featureLevel, IID_PPV_ARGS(&pDevice)) == S_OK)
            {
				rgb32FSupported = (desc.VendorId != 0x1002); // The AMD cards I tried can't handle 96-bits textures correctly
                return pDevice;
            }
        }

        logErrorAndExit("Could not find a GPU that supports D3D12 device");
        return nullptr;
    }

    CommandQueueHandle Device::getCommandQueueHandle(LowLevelContextData::CommandQueueType type, uint32_t index) const
    {
        return mCmdQueues[(uint32_t)type][index];
    }

    ApiCommandQueueType Device::getApiCommandQueueType(LowLevelContextData::CommandQueueType type) const
    {
        switch (type)
        {
        case LowLevelContextData::CommandQueueType::Copy:
            return D3D12_COMMAND_LIST_TYPE_COPY;
        case LowLevelContextData::CommandQueueType::Compute:
            return D3D12_COMMAND_LIST_TYPE_COMPUTE;
        case LowLevelContextData::CommandQueueType::Direct:
            return D3D12_COMMAND_LIST_TYPE_DIRECT;
        default:
            should_not_get_here();
            return D3D12_COMMAND_LIST_TYPE_DIRECT;
        }
    }

    bool Device::getApiFboData(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat, std::vector<ResourceHandle>& apiHandles, uint32_t& currentBackBufferIndex)
    {
        for (uint32_t i = 0; i < mSwapChainBufferCount; i++)
        {
            HRESULT hr = mpApiData->pSwapChain->GetBuffer(i, IID_PPV_ARGS(&apiHandles[i]));
            if (FAILED(hr))
            {
                d3dTraceHR("Failed to get back-buffer " + std::to_string(i) + " from the swap-chain", hr);
                return false;
            }
        }
        currentBackBufferIndex = mpApiData->pSwapChain->GetCurrentBackBufferIndex();
        return true;
    }

    void Device::destroyApiObjects()
    {
        safe_delete(mpApiData);
        mpWindow.reset();
    }

    void Device::apiPresent()
    {
        mpApiData->pSwapChain->Present(mVsyncOn ? 1 : 0, 0);
        mCurrentBackBufferIndex = (mCurrentBackBufferIndex + 1) % mSwapChainBufferCount;
    }

    bool Device::apiInit(const Desc& desc)
    {
        DeviceApiData* pData = new DeviceApiData;
        mpApiData = pData;
        UINT dxgiFlags = 0;
        if (desc.enableDebugLayer)
        {
            ID3D12DebugPtr pDx12Debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDx12Debug))))
            {
                pDx12Debug->EnableDebugLayer();
            }
            dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }

        // Create the DXGI factory
        d3d_call(CreateDXGIFactory2(dxgiFlags, IID_PPV_ARGS(&mpApiData->pDxgiFactory)));

        mApiHandle = createDevice(mpApiData->pDxgiFactory, getD3DFeatureLevel(desc.apiMajorVersion, desc.apiMinorVersion), desc.createDeviceFunc, mRgb32FloatSupported);
        if (mApiHandle == nullptr)
        {
            return false;
        }

        for (uint32_t i = 0; i < kQueueTypeCount; i++)
        {
            for (uint32_t j = 0; j < desc.cmdQueues[i]; j++)
            {
                // Create the command queue
                D3D12_COMMAND_QUEUE_DESC cqDesc = {};
                cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
                cqDesc.Type = getApiCommandQueueType((LowLevelContextData::CommandQueueType)i);

                ID3D12CommandQueuePtr pQueue;
                if (FAILED(mApiHandle->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&pQueue))))
                {
                    logError("Failed to create command queue");
                    return nullptr;
                }

                mCmdQueues[i].push_back(pQueue);
            }
        }

        uint64_t freq;
        d3d_call(getCommandQueueHandle(LowLevelContextData::CommandQueueType::Direct, 0)->GetTimestampFrequency(&freq));
        mGpuTimestampFrequency = 1000.0 / (double)freq;

        mpRenderContext = RenderContext::create(mCmdQueues[(uint32_t)LowLevelContextData::CommandQueueType::Direct][0]);
        return createSwapChain(desc.colorFormat);
    }

    bool Device::createSwapChain(ResourceFormat colorFormat)
    {
        mpApiData->pSwapChain = createDxgiSwapChain(mpApiData->pDxgiFactory, mpWindow.get(), mpRenderContext->getLowLevelData()->getCommandQueue(), colorFormat);
        if (mpApiData->pSwapChain == nullptr)
        {
            return false;
        }

        mpSwapChainFbos.resize(mSwapChainBufferCount);
        return true;
    }

    void Device::apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat)
    {
        DXGI_SWAP_CHAIN_DESC desc;
        d3d_call(mpApiData->pSwapChain->GetDesc(&desc));
        d3d_call(mpApiData->pSwapChain->ResizeBuffers(mSwapChainBufferCount, width, height, desc.BufferDesc.Format, desc.Flags));
    }

    bool Device::isWindowOccluded() const
    {
        if (mpApiData->isWindowOccluded)
        {
            mpApiData->isWindowOccluded = (mpApiData->pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED);
        }
        return mpApiData->isWindowOccluded;
    }

    bool Device::isExtensionSupported(const std::string& name) const
    {
        return _ENABLE_NVAPI;
    }
}