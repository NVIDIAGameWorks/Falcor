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
#include "GFXDeviceApiData.h"
#include "GFXFormats.h"
#include "Core/Macros.h"
#include "Core/API/Raytracing.h"
#include "Core/API/GFX/GFXAPI.h"
#include "Core/Program/Program.h"
#include "Utils/Logger.h"

#if FALCOR_NVAPI_AVAILABLE
#include "Core/API/D3D12/D3D12NvApiExDesc.h"
#endif

using namespace gfx;

namespace Falcor
{
    static_assert((uint32_t)RayFlags::None == 0);
    static_assert((uint32_t)RayFlags::ForceOpaque == 0x1);
    static_assert((uint32_t)RayFlags::ForceNonOpaque == 0x2);
    static_assert((uint32_t)RayFlags::AcceptFirstHitAndEndSearch == 0x4);
    static_assert((uint32_t)RayFlags::SkipClosestHitShader == 0x8);
    static_assert((uint32_t)RayFlags::CullBackFacingTriangles == 0x10);
    static_assert((uint32_t)RayFlags::CullFrontFacingTriangles == 0x20);
    static_assert((uint32_t)RayFlags::CullOpaque == 0x40);
    static_assert((uint32_t)RayFlags::CullNonOpaque == 0x80);
    static_assert((uint32_t)RayFlags::SkipTriangles == 0x100);
    static_assert((uint32_t)RayFlags::SkipProceduralPrimitives == 0x200);

    static_assert(getMaxViewportCount() <= 8);

#if FALCOR_NVAPI_AVAILABLE
    // To use NVAPI, we intercept the API calls in the gfx layer and dispatch into the NVAPI_Create*PipelineState
    // functions instead if the shader uses NVAPI functionalities.
    // We use the gfx API dispatcher mechanism to intercept and redirect the API call.
    // This is done by defining an implementation of `IPipelineCreationAPIDispatcher` and passing an instance of this
    // implementation to `gfxCreateDevice`.
    class PipelineCreationAPIDispatcher : public gfx::IPipelineCreationAPIDispatcher
    {
    private:

        bool findNvApiShaderParameter(slang::IComponentType* program, uint32_t& space, uint32_t& registerId)
        {
            auto globalTypeLayout = program->getLayout()->getGlobalParamsVarLayout()->getTypeLayout();
            auto index = globalTypeLayout->findFieldIndexByName("g_NvidiaExt");
            if (index != -1)
            {
                auto field = globalTypeLayout->getFieldByIndex((unsigned int)index);
                space = field->getBindingSpace();
                registerId = field->getBindingIndex();
                return true;
            }
            return false;
        }

        void createNvApiUavSlotExDesc(NvApiPsoExDesc& ret, uint32_t space, uint32_t uavSlot)
        {
            ret.psoExtension = NV_PSO_SET_SHADER_EXTNENSION_SLOT_AND_SPACE;

            auto& desc = ret.mExtSlotDesc;
            std::memset(&desc, 0, sizeof(desc));

            desc.psoExtension = NV_PSO_SET_SHADER_EXTNENSION_SLOT_AND_SPACE;
            desc.version = NV_SET_SHADER_EXTENSION_SLOT_DESC_VER;
            desc.baseVersion = NV_PSO_EXTENSION_DESC_VER;
            desc.uavSlot = uavSlot;
            desc.registerSpace = space;
        }

    public:
        PipelineCreationAPIDispatcher()
        {
            if (NvAPI_Initialize() != NVAPI_OK)
            {
                throw RuntimeError("Failed to initialize NVAPI.");
            }
        }

        ~PipelineCreationAPIDispatcher()
        {
            NvAPI_Unload();
        }

        virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) override
        {
            if (uuid == SlangUUID SLANG_UUID_IPipelineCreationAPIDispatcher)
            {
                *outObject = static_cast<gfx::IPipelineCreationAPIDispatcher*>(this);
                return SLANG_OK;
            }
            return SLANG_E_NO_INTERFACE;
        }

        // The lifetime of this dispatcher object will be managed by `Falcor::Device` so we don't need
        // to actually implement reference counting here.
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override
        {
            return 1;
        }

        virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override
        {
            return 1;
        }

        // This method will be called by the gfx layer to create an API object for a compute pipeline state.
        virtual gfx::Result createComputePipelineState(gfx::IDevice* device, slang::IComponentType* program, void* pipelineDesc, void** outPipelineState)
        {
            gfx::IDevice::InteropHandles nativeHandle;
            device->getNativeDeviceHandles(&nativeHandle);
            ID3D12Device* pD3D12Device = reinterpret_cast<ID3D12Device*>(nativeHandle.handles[0].handleValue);

            uint32_t space, registerId;
            if (findNvApiShaderParameter(program, space, registerId))
            {
                NvApiPsoExDesc psoDesc = {};
                createNvApiUavSlotExDesc(psoDesc, space, registerId);
                const NVAPI_D3D12_PSO_EXTENSION_DESC* ppPSOExtensionsDesc[1] = { &psoDesc.mExtSlotDesc };
                auto result = NvAPI_D3D12_CreateComputePipelineState(
                    pD3D12Device,
                    reinterpret_cast<D3D12_COMPUTE_PIPELINE_STATE_DESC*>(pipelineDesc),
                    1,
                    ppPSOExtensionsDesc,
                    (ID3D12PipelineState**)outPipelineState);
                return (result == NVAPI_OK) ? SLANG_OK : SLANG_FAIL;
            }
            else
            {
                ID3D12PipelineState* pState = nullptr;
                SLANG_RETURN_ON_FAIL(pD3D12Device->CreateComputePipelineState(
                    reinterpret_cast<D3D12_COMPUTE_PIPELINE_STATE_DESC*>(pipelineDesc), IID_PPV_ARGS(&pState)));
                *outPipelineState = pState;
            }
            return SLANG_OK;
        }

        // This method will be called by the gfx layer to create an API object for a graphics pipeline state.
        virtual gfx::Result createGraphicsPipelineState(gfx::IDevice* device, slang::IComponentType* program, void* pipelineDesc, void** outPipelineState)
        {
            gfx::IDevice::InteropHandles nativeHandle;
            device->getNativeDeviceHandles(&nativeHandle);
            ID3D12Device* pD3D12Device = reinterpret_cast<ID3D12Device*>(nativeHandle.handles[0].handleValue);

            uint32_t space, registerId;
            if (findNvApiShaderParameter(program, space, registerId))
            {
                NvApiPsoExDesc psoDesc = {};
                createNvApiUavSlotExDesc(psoDesc, space, registerId);
                const NVAPI_D3D12_PSO_EXTENSION_DESC* ppPSOExtensionsDesc[1] = { &psoDesc.mExtSlotDesc };

                auto result = NvAPI_D3D12_CreateGraphicsPipelineState(
                    pD3D12Device,
                    reinterpret_cast<D3D12_GRAPHICS_PIPELINE_STATE_DESC*>(pipelineDesc),
                    1,
                    ppPSOExtensionsDesc,
                    (ID3D12PipelineState**)outPipelineState);
                return (result == NVAPI_OK) ? SLANG_OK : SLANG_FAIL;
            }
            else
            {
                ID3D12PipelineState* pState = nullptr;
                SLANG_RETURN_ON_FAIL(pD3D12Device->CreateGraphicsPipelineState(
                    reinterpret_cast<D3D12_GRAPHICS_PIPELINE_STATE_DESC*>(pipelineDesc), IID_PPV_ARGS(&pState)));
                *outPipelineState = pState;
            }
            return SLANG_OK;
        }

        // This method will be called by the gfx layer right before creating a ray tracing state object.
        virtual gfx::Result beforeCreateRayTracingState(gfx::IDevice* device, slang::IComponentType* program)
        {
            gfx::IDevice::InteropHandles nativeHandle;
            device->getNativeDeviceHandles(&nativeHandle);
            ID3D12Device* pD3D12Device = reinterpret_cast<ID3D12Device*>(nativeHandle.handles[0].handleValue);

            uint32_t space, registerId;
            if (findNvApiShaderParameter(program, space, registerId))
            {
                if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(pD3D12Device, registerId, space) != NVAPI_OK)
                {
                    throw RuntimeError("Failed to set NvApi extension");
                }
            }

            return SLANG_OK;
        }

        // This method will be called by the gfx layer right after creating a ray tracing state object.
        virtual gfx::Result afterCreateRayTracingState(gfx::IDevice* device, slang::IComponentType* program)
        {
            gfx::IDevice::InteropHandles nativeHandle;
            device->getNativeDeviceHandles(&nativeHandle);
            ID3D12Device* pD3D12Device = reinterpret_cast<ID3D12Device*>(nativeHandle.handles[0].handleValue);

            uint32_t space, registerId;
            if (findNvApiShaderParameter(program, space, registerId))
            {
                if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(pD3D12Device, 0xFFFFFFFF, 0) != NVAPI_OK)
                {
                    throw RuntimeError("Failed to set NvApi extension");
                }
            }
            return SLANG_OK;
        }

    };
#endif // FALCOR_NVAPI_AVAILABLE

    class GFXDebugCallBack : public IDebugCallback
    {
        virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(DebugMessageType type, DebugMessageSource source, const char* message) override
        {
            if (type == DebugMessageType::Error)
            {
                logError("GFX Error: {}", message);
            }
            else if (type == DebugMessageType::Warning)
            {
                logWarning("GFX Warning: {}", message);
            }
            else
            {
                logDebug("GFX Info: {}", message);
            }
        }
    };

    GFXDebugCallBack gGFXDebugCallBack;

    Device::Device(Window::SharedPtr pWindow, const Desc& desc) : mpWindow(pWindow), mDesc(desc) {}
    Device::~Device() = default;

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
            gfx::Result hr = mpApiData->pSwapChain->getImage(i, imageHandle.writeRef());
            apiHandles[i] = imageHandle.get();
            if (SLANG_FAILED(hr))
            {
                logError("Failed to get back-buffer {} from swap-chain, error: {}", i, hr);
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
#if FALCOR_NVAPI_AVAILABLE
        mpApiData->pApiDispatcher.reset();
#endif
        mpApiData.reset();
    }

    gfx::ITransientResourceHeap* Device::getCurrentTransientResourceHeap()
    {
        return mpApiData->pTransientResourceHeaps[mCurrentBackBufferIndex].get();
    }

    void Device::present()
    {
        mpRenderContext->resourceBarrier(mpSwapChainFbos[mCurrentBackBufferIndex]->getColorTexture(0).get(), Resource::State::Present);
        mpRenderContext->flush();
        mpApiData->pSwapChain->present();
        // Call to acquireNextImage will block until the next image in the swapchain is ready for present and all the
        // GPU tasks associated with rendering the next image in the swapchain has already completed.
        mCurrentBackBufferIndex = mpApiData->pSwapChain->acquireNextImage();
        if (mCurrentBackBufferIndex != -1)
        {
            mpRenderContext->getLowLevelData()->closeCommandBuffer();
            getCurrentTransientResourceHeap()->synchronizeAndReset();
            mpRenderContext->getLowLevelData()->openCommandBuffer();
        }
        // Since call to `acquireNextImage` already included a fence wait inside GFX, we don't need to wait again.
        // Instead we just signal `mpFrameFence` from the host.
        mpFrameFence->externalSignal();
        if (mpFrameFence->getCpuValue() >= kSwapChainBuffersCount) mpFrameFence->setGpuValue(mpFrameFence->getCpuValue() - kSwapChainBuffersCount);
        executeDeferredReleases();
        mFrameID++;
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

        mpApiData.reset(new DeviceApiData);
        DeviceApiData* pData = mpApiData.get();

        IDevice::Desc desc = {};
#if FALCOR_GFX_VK
        desc.deviceType = DeviceType::Vulkan;
#elif FALCOR_GFX_D3D12
        desc.deviceType = DeviceType::DirectX12;
#endif
        desc.slang.slangGlobalSession = getSlangGlobalSession();

        gfx::D3D12DeviceExtendedDesc extDesc = {};
        extDesc.rootParameterShaderAttributeName = "root";
        void* pExtDesc = &extDesc;
        desc.extendedDescCount = 1;
        desc.extendedDescs = &pExtDesc;

#if FALCOR_NVAPI_AVAILABLE
        mpApiData->pApiDispatcher.reset(new PipelineCreationAPIDispatcher());
        desc.apiCommandDispatcher = static_cast<ISlangUnknown*>(mpApiData->pApiDispatcher.get());
#endif
        gfxSetDebugCallback(&gGFXDebugCallBack);

        if (SLANG_FAILED(gfxCreateDevice(&desc, mpApiData->pDevice.writeRef()))) return false;

        mApiHandle = mpApiData->pDevice;

        mGpuTimestampFrequency = 1000.0 / (double)mApiHandle->getDeviceInfo().timestampFrequency;
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

#if FALCOR_GFX_VK
        if (mpWindow->getClientAreaSize().x == 0 || mpWindow->getClientAreaSize().y == 0)
        {
            logWarning("Attempting to initialize Vulkan device on a 0-sized window. The swapchain will be invalid and using it could lead to error.");
        }
#endif

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
#if FALCOR_WINDOWS
        if (SLANG_FAILED(mpApiData->pDevice->createSwapchain(desc, gfx::WindowHandle::FromHwnd(mpWindow->getApiHandle()), mpApiData->pSwapChain.writeRef())))
#elif FALCOR_LINUX
        if (SLANG_FAILED(mpApiData->pDevice->createSwapchain(desc, gfx::WindowHandle::FromXWindow(mpWindow->getApiHandle().pDisplay, mpWindow->getApiHandle().window), mpApiData->pSwapChain.writeRef())))
#endif
        {
            return false;
        }
        mCurrentBackBufferIndex = mpApiData->pSwapChain->acquireNextImage();
        return true;
    }

    void Device::apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat)
    {
        FALCOR_ASSERT(mpApiData->pSwapChain);
        FALCOR_GFX_CALL(mpApiData->pSwapChain->resize(width, height));
        mCurrentBackBufferIndex = mpApiData->pSwapChain->acquireNextImage();
        if (mCurrentBackBufferIndex != -1)
        {
            mpRenderContext->getLowLevelData()->closeCommandBuffer();
            getCurrentTransientResourceHeap()->synchronizeAndReset();
            mpRenderContext->getLowLevelData()->openCommandBuffer();
        }
    }

    bool Device::isWindowOccluded() const
    {
        return mCurrentBackBufferIndex == -1;
    }

#if FALCOR_HAS_D3D12
    const D3D12DeviceHandle Device::getD3D12Handle()
    {
        gfx::IDevice::InteropHandles interopHandles = {};
        mApiHandle->getNativeDeviceHandles(&interopHandles);
        FALCOR_ASSERT(interopHandles.handles[0].api == gfx::InteropHandleAPI::D3D12);
        return reinterpret_cast<ID3D12Device*>(interopHandles.handles[0].handleValue);
    }
#else
    const D3D12DeviceHandle Device::getD3D12Handle()
    {
        return nullptr;
    }
#endif // FALCOR_HAS_D3D12
}
