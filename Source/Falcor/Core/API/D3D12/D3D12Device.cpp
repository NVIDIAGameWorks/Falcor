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
#include "stdafx.h"
#include "Core/API/Device.h"

namespace Falcor
{
    namespace
    {
        const uint32_t kDefaultVendorId = 0x10DE; ///< NVIDIA GPUs
    }

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
                return D3D_FEATURE_LEVEL_11_0;
            case 1:
                return D3D_FEATURE_LEVEL_11_1;
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

    IDXGISwapChain3Ptr createDxgiSwapChain(IDXGIFactory4* pFactory, const Window* pWindow, ID3D12CommandQueue* pCommandQueue, ResourceFormat colorFormat, uint32_t bufferCount)
    {
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
        swapChainDesc.BufferCount = bufferCount;
        swapChainDesc.Width = pWindow->getClientAreaSize().x;
        swapChainDesc.Height = pWindow->getClientAreaSize().y;
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

    DeviceHandle createDevice(IDXGIFactory4* pFactory, D3D_FEATURE_LEVEL requestedFeatureLevel, const std::vector<UUID>& experimentalFeatures)
    {
        // Feature levels to try creating devices. Listed in descending order so the highest supported level is used.
        const static D3D_FEATURE_LEVEL kFeatureLevels[] =
        {
            D3D_FEATURE_LEVEL_12_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3,
            D3D_FEATURE_LEVEL_9_2,
            D3D_FEATURE_LEVEL_9_1
        };

        const static uint32_t kUnspecified = uint32_t(-1);

        // Read FALCOR_GPU_VENDOR_ID environment variable.
        const uint32_t preferredGpuVendorId = ([]()
        {
            std::string str;
            // Use base = 0 in stoi to autodetect octal/hex/decimal strings
            return getEnvironmentVariable("FALCOR_GPU_VENDOR_ID", str) ? std::stoi(str, nullptr, 0) : kUnspecified;
        })();

        // Read FALCOR_GPU_DEVICE_ID environment variable.
        const uint32_t preferredGpuIndex = ([] ()
        {
            std::string str;
            return getEnvironmentVariable("FALCOR_GPU_DEVICE_ID", str) ? std::stoi(str) : kUnspecified;
        })();

        IDXGIAdapter1Ptr pAdapter;
        DeviceHandle pDevice;
        D3D_FEATURE_LEVEL selectedFeatureLevel;

        auto createMaxFeatureLevel = [&](const D3D_FEATURE_LEVEL* pFeatureLevels, uint32_t featureLevelCount) -> bool
        {
            for (uint32_t i = 0; i < featureLevelCount; i++)
            {
                if (SUCCEEDED(D3D12CreateDevice(pAdapter, pFeatureLevels[i], IID_PPV_ARGS(&pDevice))))
                {
                    selectedFeatureLevel = pFeatureLevels[i];
                    return true;
                }
            }

            return false;
        };

        // Properties to search for
        const uint32_t vendorId = (preferredGpuVendorId != kUnspecified) ? preferredGpuVendorId : kDefaultVendorId;
        const uint32_t gpuIdx = (preferredGpuIndex != kUnspecified) ? preferredGpuIndex : 0;

        // Select adapter
        uint32_t vendorDeviceIndex = 0; // Tracks device index within adapters matching a specific vendor ID
        uint32_t selectedAdapterIndex = uint32_t(-1); // The final adapter chosen to create the device object from
        for (uint32_t i = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(i, &pAdapter); i++)
        {
            DXGI_ADAPTER_DESC1 desc;
            pAdapter->GetDesc1(&desc);

            // Skip SW adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

            // Skip if vendorId doesn't match requested
            if (desc.VendorId != vendorId) continue;

            // When a vendor match is found above, count to the specified device index of that vendor (e.g. the i-th NVIDIA GPU)
            if(vendorDeviceIndex++ < gpuIdx) continue;

            // Select the first adapter satisfying the conditions
            selectedAdapterIndex = i;
            break;
        }

        if (selectedAdapterIndex == uint32_t(-1))
        {
            // If no GPU was found, just select the first
            selectedAdapterIndex = 0;

            // Log a warning if an adapter matching user specifications wasn't found.
            // Selection could have failed based on the default settings, but that isn't an error.
            if (preferredGpuVendorId != kUnspecified || preferredGpuIndex != kUnspecified)
            {
                logWarning("Could not find a GPU matching conditions specified in environment variables.");
            }
        }

        // Retrieve the adapter that's been selected
        HRESULT result = pFactory->EnumAdapters1(selectedAdapterIndex, &pAdapter);
        assert(SUCCEEDED(result));

        if (requestedFeatureLevel == 0) createMaxFeatureLevel(kFeatureLevels, arraysize(kFeatureLevels));
        else createMaxFeatureLevel(&requestedFeatureLevel, 1);

        if (pDevice != nullptr)
        {
            logInfo("Successfully created device with feature level: " + to_string(selectedFeatureLevel));
            return pDevice;
        }

        logFatal("Could not find a GPU that supports D3D12 device");
        return nullptr;
    }

    Device::SupportedFeatures getSupportedFeatures(DeviceHandle pDevice)
    {
        Device::SupportedFeatures supported = Device::SupportedFeatures::None;

        D3D12_FEATURE_DATA_D3D12_OPTIONS2 features2;
        HRESULT hr = pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &features2, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS2));
        if (FAILED(hr) || features2.ProgrammableSamplePositionsTier == D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_NOT_SUPPORTED)
        {
            logInfo("Programmable sample positions is not supported on this device.");
        }
        else
        {
            if (features2.ProgrammableSamplePositionsTier == D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_1) supported |= Device::SupportedFeatures::ProgrammableSamplePositionsPartialOnly;
            else if (features2.ProgrammableSamplePositionsTier == D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_2) supported |= Device::SupportedFeatures::ProgrammableSamplePositionsFull;
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS5 features5;
        hr = pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features5, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS5));
        if (FAILED(hr) || features5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
        {
            logInfo("Raytracing is not supported on this device.");
        }
        else
        {
            supported |= Device::SupportedFeatures::Raytracing;
        }

        return supported;
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
            throw std::exception("Unknown command queue type");
        }
    }

    bool Device::getApiFboData(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat, ResourceHandle apiHandles[kSwapChainBuffersCount], uint32_t& currentBackBufferIndex)
    {
        for (uint32_t i = 0; i < kSwapChainBuffersCount; i++)
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

    void Device::toggleFullScreen(bool fullscreen)
    {
        mpApiData->pSwapChain->SetFullscreenState(fullscreen, nullptr);
    }

    void Device::destroyApiObjects()
    {
        safe_delete(mpApiData);
        mpWindow.reset();
    }

    void Device::apiPresent()
    {
        mpApiData->pSwapChain->Present(mDesc.enableVsync ? 1 : 0, 0);
        mCurrentBackBufferIndex = (mCurrentBackBufferIndex + 1) % kSwapChainBuffersCount;
    }

    bool Device::apiInit()
    {
        DeviceApiData* pData = new DeviceApiData;
        mpApiData = pData;
        UINT dxgiFlags = 0;
        if (mDesc.enableDebugLayer)
        {
            ID3D12DebugPtr pDx12Debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDx12Debug))))
            {
                pDx12Debug->EnableDebugLayer();
                dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
            }
            else
            {
                logWarning("The D3D12 debug layer is not available. Please install Graphics Tools.");
                mDesc.enableDebugLayer = false;
            }
        }

        // Create the DXGI factory
        d3d_call(CreateDXGIFactory2(dxgiFlags, IID_PPV_ARGS(&mpApiData->pDxgiFactory)));

        // Create the device
        mApiHandle = createDevice(mpApiData->pDxgiFactory, getD3DFeatureLevel(mDesc.apiMajorVersion, mDesc.apiMinorVersion), mDesc.experimentalFeatures);
        if (mApiHandle == nullptr) return false;

        mSupportedFeatures = getSupportedFeatures(mApiHandle);

        if (mDesc.enableDebugLayer)
        {
            MAKE_SMART_COM_PTR(ID3D12InfoQueue);
            ID3D12InfoQueuePtr pInfoQueue;
            mApiHandle->QueryInterface(IID_PPV_ARGS(&pInfoQueue));
            D3D12_MESSAGE_ID hideMessages[] =
            {
                D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_ID_COPY_DESCRIPTORS_INVALID_RANGES
            };
            D3D12_INFO_QUEUE_FILTER f = {};
            f.DenyList.NumIDs = arraysize(hideMessages);
            f.DenyList.pIDList = hideMessages;
            pInfoQueue->AddStorageFilterEntries(&f);

            // Break on DEVICE_REMOVAL_PROCESS_AT_FAULT
            pInfoQueue->SetBreakOnID(D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_AT_FAULT, true);
        }

        for (uint32_t i = 0; i < kQueueTypeCount; i++)
        {
            for (uint32_t j = 0; j < mDesc.cmdQueues[i]; j++)
            {
                // Create the command queue
                D3D12_COMMAND_QUEUE_DESC cqDesc = {};
                cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
                cqDesc.Type = getApiCommandQueueType((LowLevelContextData::CommandQueueType)i);

                ID3D12CommandQueuePtr pQueue;
                if (FAILED(mApiHandle->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&pQueue))))
                {
                    logError("Failed to create command queue");
                    return false;
                }

                mCmdQueues[i].push_back(pQueue);
            }
        }

        uint64_t freq;
        d3d_call(getCommandQueueHandle(LowLevelContextData::CommandQueueType::Direct, 0)->GetTimestampFrequency(&freq));
        mGpuTimestampFrequency = 1000.0 / (double)freq;
        return createSwapChain(mDesc.colorFormat);
    }

    bool Device::createSwapChain(ResourceFormat colorFormat)
    {
        mpApiData->pSwapChain = createDxgiSwapChain(mpApiData->pDxgiFactory, mpWindow.get(), getCommandQueueHandle(LowLevelContextData::CommandQueueType::Direct, 0), colorFormat, kSwapChainBuffersCount);
        if (mpApiData->pSwapChain == nullptr) return false;
        return true;
    }

    void Device::apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat)
    {
        DXGI_SWAP_CHAIN_DESC desc;
        d3d_call(mpApiData->pSwapChain->GetDesc(&desc));
        d3d_call(mpApiData->pSwapChain->ResizeBuffers(kSwapChainBuffersCount, width, height, desc.BufferDesc.Format, desc.Flags));
    }

    bool Device::isWindowOccluded() const
    {
        if (mpApiData->isWindowOccluded)
        {
            mpApiData->isWindowOccluded = (mpApiData->pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED);
        }
        return mpApiData->isWindowOccluded;
    }
}
