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
#include "Device.h"
#include "Raytracing.h"
#include "GFXHelpers.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/Macros.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramManager.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Timing/Profiler.h"

#if FALCOR_HAS_D3D12
#include "Core/API/Shared/D3D12DescriptorPool.h"
#endif

#if FALCOR_NVAPI_AVAILABLE
#include "Core/API/NvApiExDesc.h"
#include <nvShaderExtnEnums.h> // Required for checking SER support.
#endif

#include <algorithm>
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

static const uint32_t kTransientHeapConstantBufferSize = 16 * 1024 * 1024;

std::shared_ptr<Device> gpDevice;

std::shared_ptr<Device>& getGlobalDevice()
{
    return gpDevice;
}

class GFXDebugCallBack : public gfx::IDebugCallback
{
    virtual SLANG_NO_THROW void SLANG_MCALL
    handleMessage(gfx::DebugMessageType type, gfx::DebugMessageSource source, const char* message) override
    {
        if (type == gfx::DebugMessageType::Error)
        {
            logError("GFX Error: {}", message);
        }
        else if (type == gfx::DebugMessageType::Warning)
        {
            logWarning("GFX Warning: {}", message);
        }
        else
        {
            logDebug("GFX Info: {}", message);
        }
    }
};

GFXDebugCallBack gGFXDebugCallBack; // TODO: REMOVEGLOBAL

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

    ~PipelineCreationAPIDispatcher() { NvAPI_Unload(); }

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
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 2; }

    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override
    {
        // Returning 2 is important here, because when releasing a COM pointer, it checks
        // if the ref count **was 1 before releasing** in order to free the object.
        return 2;
    }

    // This method will be called by the gfx layer to create an API object for a compute pipeline state.
    virtual gfx::Result createComputePipelineState(
        gfx::IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState
    )
    {
        gfx::IDevice::InteropHandles nativeHandle;
        FALCOR_GFX_CALL(device->getNativeDeviceHandles(&nativeHandle));
        ID3D12Device* pD3D12Device = reinterpret_cast<ID3D12Device*>(nativeHandle.handles[0].handleValue);

        uint32_t space, registerId;
        if (findNvApiShaderParameter(program, space, registerId))
        {
            NvApiPsoExDesc psoDesc = {};
            createNvApiUavSlotExDesc(psoDesc, space, registerId);
            const NVAPI_D3D12_PSO_EXTENSION_DESC* ppPSOExtensionsDesc[1] = {&psoDesc.mExtSlotDesc};
            auto result = NvAPI_D3D12_CreateComputePipelineState(
                pD3D12Device, reinterpret_cast<D3D12_COMPUTE_PIPELINE_STATE_DESC*>(pipelineDesc), 1, ppPSOExtensionsDesc,
                (ID3D12PipelineState**)outPipelineState
            );
            return (result == NVAPI_OK) ? SLANG_OK : SLANG_FAIL;
        }
        else
        {
            ID3D12PipelineState* pState = nullptr;
            SLANG_RETURN_ON_FAIL(pD3D12Device->CreateComputePipelineState(
                reinterpret_cast<D3D12_COMPUTE_PIPELINE_STATE_DESC*>(pipelineDesc), IID_PPV_ARGS(&pState)
            ));
            *outPipelineState = pState;
        }
        return SLANG_OK;
    }

    // This method will be called by the gfx layer to create an API object for a graphics pipeline state.
    virtual gfx::Result createGraphicsPipelineState(
        gfx::IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState
    )
    {
        gfx::IDevice::InteropHandles nativeHandle;
        FALCOR_GFX_CALL(device->getNativeDeviceHandles(&nativeHandle));
        ID3D12Device* pD3D12Device = reinterpret_cast<ID3D12Device*>(nativeHandle.handles[0].handleValue);

        uint32_t space, registerId;
        if (findNvApiShaderParameter(program, space, registerId))
        {
            NvApiPsoExDesc psoDesc = {};
            createNvApiUavSlotExDesc(psoDesc, space, registerId);
            const NVAPI_D3D12_PSO_EXTENSION_DESC* ppPSOExtensionsDesc[1] = {&psoDesc.mExtSlotDesc};

            auto result = NvAPI_D3D12_CreateGraphicsPipelineState(
                pD3D12Device, reinterpret_cast<D3D12_GRAPHICS_PIPELINE_STATE_DESC*>(pipelineDesc), 1, ppPSOExtensionsDesc,
                (ID3D12PipelineState**)outPipelineState
            );
            return (result == NVAPI_OK) ? SLANG_OK : SLANG_FAIL;
        }
        else
        {
            ID3D12PipelineState* pState = nullptr;
            SLANG_RETURN_ON_FAIL(pD3D12Device->CreateGraphicsPipelineState(
                reinterpret_cast<D3D12_GRAPHICS_PIPELINE_STATE_DESC*>(pipelineDesc), IID_PPV_ARGS(&pState)
            ));
            *outPipelineState = pState;
        }
        return SLANG_OK;
    }

    // This method will be called by the gfx layer right before creating a ray tracing state object.
    virtual gfx::Result beforeCreateRayTracingState(gfx::IDevice* device, slang::IComponentType* program)
    {
        gfx::IDevice::InteropHandles nativeHandle;
        FALCOR_GFX_CALL(device->getNativeDeviceHandles(&nativeHandle));
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
        FALCOR_GFX_CALL(device->getNativeDeviceHandles(&nativeHandle));
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

inline Device::Type getDefaultDeviceType()
{
#if FALCOR_HAS_D3D12
    return Device::Type::D3D12;
#elif FALCOR_HAS_VULKAN
    return Device::Type::Vulkan;
#else
    throw RuntimeError("No default device type");
#endif
}

inline gfx::DeviceType getGfxDeviceType(Device::Type deviceType)
{
    switch (deviceType)
    {
    case Device::Type::Default:
        return gfx::DeviceType::Default;
    case Device::Type::D3D12:
        return gfx::DeviceType::DirectX12;
    case Device::Type::Vulkan:
        return gfx::DeviceType::Vulkan;
    default:
        throw RuntimeError("Unknown device type");
    }
}

inline Device::Limits queryLimits(gfx::IDevice* pDevice)
{
    const auto& deviceLimits = pDevice->getDeviceInfo().limits;

    auto toUint3 = [](const uint32_t value[]) { return uint3(value[0], value[1], value[2]); };

    Device::Limits limits = {};
    limits.maxComputeDispatchThreadGroups = toUint3(deviceLimits.maxComputeDispatchThreadGroups);
    limits.maxShaderVisibleSamplers = deviceLimits.maxShaderVisibleSamplers;
    return limits;
}

inline Device::SupportedFeatures querySupportedFeatures(gfx::IDevice* pDevice)
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

    if (pDevice->hasFeature("wave-ops"))
    {
        result |= Device::SupportedFeatures::WaveOperations;
    }

    return result;
}

inline Device::ShaderModel querySupportedShaderModel(gfx::IDevice* pDevice)
{
    struct SMLevel
    {
        const char* name;
        Device::ShaderModel level;
    };
    const SMLevel levels[] = {{"sm_6_7", Device::ShaderModel::SM6_7}, {"sm_6_6", Device::ShaderModel::SM6_6},
                              {"sm_6_5", Device::ShaderModel::SM6_5}, {"sm_6_4", Device::ShaderModel::SM6_4},
                              {"sm_6_3", Device::ShaderModel::SM6_3}, {"sm_6_2", Device::ShaderModel::SM6_2},
                              {"sm_6_1", Device::ShaderModel::SM6_1}, {"sm_6_0", Device::ShaderModel::SM6_0}};
    for (auto level : levels)
    {
        if (pDevice->hasFeature(level.name))
        {
            return level.level;
        }
    }
    return Device::ShaderModel::Unknown;
}

Device::Device(const Desc& desc) : mDesc(desc) {}
Device::~Device()
{
    mGfxDevice.setNull();

#if FALCOR_NVAPI_AVAILABLE
    mpAPIDispatcher.reset();
#endif
}

std::shared_ptr<Device> Device::create(const Device::Desc& desc)
{
    if (gpDevice)
    {
        throw RuntimeError("Falcor only supports a single device.");
    }

    gpDevice = std::make_shared<Device>(desc);
    if (!gpDevice->init())
    {
        throw RuntimeError("Failed to create device.");
    }

    return gpDevice;
}

bool Device::init()
{
    // Create a global slang session passed to GFX and used for compiling programs in ProgramManager.
    slang::createGlobalSession(mSlangGlobalSession.writeRef());

    if (mDesc.type == Type::Default)
        mDesc.type = getDefaultDeviceType();

#if !FALCOR_HAS_D3D12
    if (mDesc.type == Type::D3D12)
        throw RuntimeError("D3D12 device not supported.");
#endif
#if !FALCOR_HAS_VULKAN
    if (mDesc.type == Type::Vulkan)
        throw RuntimeError("Vulkan device not supported.");
#endif

    gfx::IDevice::Desc desc = {};
    desc.deviceType = getGfxDeviceType(mDesc.type);
    desc.slang.slangGlobalSession = mSlangGlobalSession;

    // Setup shader cache.
    desc.shaderCache.maxEntryCount = mDesc.maxShaderCacheEntryCount;
    if (mDesc.shaderCachePath == "")
    {
        desc.shaderCache.shaderCachePath = nullptr;
    }
    else
    {
        desc.shaderCache.shaderCachePath = mDesc.shaderCachePath.c_str();
        // If the supplied shader cache path does not exist, we will need to create it before creating the device.
        if (std::filesystem::exists(mDesc.shaderCachePath))
        {
            if (!std::filesystem::is_directory(mDesc.shaderCachePath))
                throw RuntimeError("Shader cache path {} exists and is not a directory", mDesc.shaderCachePath);
        }
        else
        {
            std::filesystem::create_directories(mDesc.shaderCachePath);
        }
    }

    std::vector<void*> extendedDescs;
    // Add extended desc for root parameter attribute.
    gfx::D3D12DeviceExtendedDesc extDesc = {};
    extDesc.rootParameterShaderAttributeName = "root";
    extendedDescs.push_back(&extDesc);
#if FALCOR_HAS_D3D12
    // Add extended descs for experimental API features.
    gfx::D3D12ExperimentalFeaturesDesc experimentalFeaturesDesc = {};
    experimentalFeaturesDesc.numFeatures = (uint32_t)mDesc.experimentalFeatures.size();
    experimentalFeaturesDesc.featureIIDs = mDesc.experimentalFeatures.data();
    if (desc.deviceType == gfx::DeviceType::DirectX12)
        extendedDescs.push_back(&experimentalFeaturesDesc);
#endif
    desc.extendedDescCount = extendedDescs.size();
    desc.extendedDescs = extendedDescs.data();

#if FALCOR_NVAPI_AVAILABLE
    mpAPIDispatcher.reset(new PipelineCreationAPIDispatcher());
    desc.apiCommandDispatcher = static_cast<ISlangUnknown*>(mpAPIDispatcher.get());
#endif

    // Setup debug layer.
    FALCOR_GFX_CALL(gfxSetDebugCallback(&gGFXDebugCallBack));
    if (mDesc.enableDebugLayer)
        gfx::gfxEnableDebugLayer();

    // Get list of available GPUs.
    const auto gpus = getGPUs(mDesc.type);

    if (mDesc.gpu >= gpus.size())
    {
        logWarning("GPU index {} is out of range, using first GPU instead.", mDesc.gpu);
        mDesc.gpu = 0;
    }

    // Try to create device on specific GPU.
    {
        desc.adapterLUID = &gpus[mDesc.gpu].luid;
        if (SLANG_FAILED(gfxCreateDevice(&desc, mGfxDevice.writeRef())))
            logWarning("Failed to create device on GPU {} ({}).", mDesc.gpu, gpus[mDesc.gpu].name);
    }

    // Otherwise try create device on any available GPU.
    if (!mGfxDevice)
    {
        desc.adapterLUID = nullptr;
        if (SLANG_FAILED(gfxCreateDevice(&desc, mGfxDevice.writeRef())))
            return false;
    }

    const auto& deviceInfo = mGfxDevice->getDeviceInfo();
    logInfo("Created GPU device '{}' using '{}' API.", deviceInfo.adapterName, deviceInfo.apiName);

    mLimits = queryLimits(mGfxDevice);
    mSupportedFeatures = querySupportedFeatures(mGfxDevice);

#if FALCOR_NVAPI_AVAILABLE
    // Explicitly check for SER support via NVAPI.
    // Slang currently relies on NVAPI to implement the SER API but cannot check it's availibility
    // due to not being shipped with NVAPI for licensing reasons.
    if (getType() == Type::D3D12)
    {
        ID3D12Device* pD3D12Device = getNativeHandle().as<ID3D12Device*>();
        // First check for avalibility of SER API (HitObject).
        bool supportSER = false;
        NvAPI_Status ret = NvAPI_D3D12_IsNvShaderExtnOpCodeSupported(pD3D12Device, NV_EXTN_OP_HIT_OBJECT_REORDER_THREAD, &supportSER);
        if (ret == NVAPI_OK && supportSER)
            mSupportedFeatures |= SupportedFeatures::ShaderExecutionReorderingAPI;

        // Then check for hardware support.
        NVAPI_D3D12_RAYTRACING_THREAD_REORDERING_CAPS reorderingCaps;
        ret = NvAPI_D3D12_GetRaytracingCaps(
            pD3D12Device, NVAPI_D3D12_RAYTRACING_CAPS_TYPE_THREAD_REORDERING, &reorderingCaps, sizeof(reorderingCaps)
        );
        if (ret == NVAPI_OK && reorderingCaps == NVAPI_D3D12_RAYTRACING_THREAD_REORDERING_CAP_STANDARD)
            mSupportedFeatures |= SupportedFeatures::RaytracingReordering;
    }
#endif

    mSupportedShaderModel = querySupportedShaderModel(mGfxDevice);
    mGpuTimestampFrequency = 1000.0 / (double)mGfxDevice->getDeviceInfo().timestampFrequency;

#if FALCOR_HAS_D3D12
    // Configure D3D12 validation layer.
    if (mDesc.type == Device::Type::D3D12 && mDesc.enableDebugLayer)
    {
        ID3D12Device* pD3D12Device = getNativeHandle().as<ID3D12Device*>();

        FALCOR_MAKE_SMART_COM_PTR(ID3D12InfoQueue);
        ID3D12InfoQueuePtr pInfoQueue;
        pD3D12Device->QueryInterface(IID_PPV_ARGS(&pInfoQueue));
        D3D12_MESSAGE_ID hideMessages[] = {
            D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
            D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
        };
        D3D12_INFO_QUEUE_FILTER f = {};
        f.DenyList.NumIDs = (UINT)std::size(hideMessages);
        f.DenyList.pIDList = hideMessages;
        pInfoQueue->AddStorageFilterEntries(&f);

        // Break on DEVICE_REMOVAL_PROCESS_AT_FAULT
        pInfoQueue->SetBreakOnID(D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_AT_FAULT, true);
    }
#endif

    for (uint32_t i = 0; i < kInFlightFrameCount; ++i)
    {
        gfx::ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.flags = gfx::ITransientResourceHeap::Flags::AllowResizing;
        transientHeapDesc.constantBufferSize = kTransientHeapConstantBufferSize;
        transientHeapDesc.samplerDescriptorCount = 2048;
        transientHeapDesc.uavDescriptorCount = 1000000;
        transientHeapDesc.srvDescriptorCount = 1000000;
        transientHeapDesc.constantBufferDescriptorCount = 1000000;
        transientHeapDesc.accelerationStructureDescriptorCount = 1000000;
        if (SLANG_FAILED(mGfxDevice->createTransientResourceHeap(transientHeapDesc, mpTransientResourceHeaps[i].writeRef())))
        {
            return false;
        }
    }

    gfx::ICommandQueue::Desc queueDesc = {};
    queueDesc.type = gfx::ICommandQueue::QueueType::Graphics;
    if (SLANG_FAILED(mGfxDevice->createCommandQueue(queueDesc, mGfxCommandQueue.writeRef())))
        return false;

    mpFrameFence = GpuFence::create(this);

#if FALCOR_HAS_D3D12
    if (getType() == Type::D3D12)
    {
        // Create the descriptor pools
        D3D12DescriptorPool::Desc poolDesc;
        poolDesc.setDescCount(ShaderResourceType::TextureSrv, 1000000)
            .setDescCount(ShaderResourceType::Sampler, 2048)
            .setShaderVisible(true);
        mpD3D12GpuDescPool = D3D12DescriptorPool::create(this, poolDesc, mpFrameFence);
        poolDesc.setShaderVisible(false).setDescCount(ShaderResourceType::Rtv, 16 * 1024).setDescCount(ShaderResourceType::Dsv, 1024);
        mpD3D12CpuDescPool = D3D12DescriptorPool::create(this, poolDesc, mpFrameFence);
    }
#endif // FALCOR_HAS_D3D12

    mpProgramManager = std::make_unique<ProgramManager>(shared_from_this());
    mpProfiler = std::make_unique<Profiler>(this);

    mpDefaultSampler = Sampler::create(this, Sampler::Desc());

    mpUploadHeap = GpuMemoryHeap::create(this, GpuMemoryHeap::Type::Upload, 1024 * 1024 * 2, mpFrameFence);
    mpRenderContext = std::make_unique<RenderContext>(this, mGfxCommandQueue);
    mpRenderContext->flush(); // This will bind the descriptor heaps.
    // TODO: Do we need to flush here or should RenderContext::create() bind the descriptor heaps automatically without flush? See #749.

    return true;
}

std::weak_ptr<QueryHeap> Device::createQueryHeap(QueryHeap::Type type, uint32_t count)
{
    QueryHeap::SharedPtr pHeap = QueryHeap::create(this, type, count);
    mTimestampQueryHeaps.push_back(pHeap);
    return pHeap;
}

void Device::releaseResource(ISlangUnknown* pResource)
{
    if (pResource)
    {
        // Some static objects get here when the application exits
        if (this)
        {
            mDeferredReleases.push({mpFrameFence ? mpFrameFence->getCpuValue() : 0, Slang::ComPtr<ISlangUnknown>(pResource)});
        }
    }
}

bool Device::isFeatureSupported(SupportedFeatures flags) const
{
    return is_set(mSupportedFeatures, flags);
}

bool Device::isShaderModelSupported(ShaderModel shaderModel) const
{
    return ((uint32_t)shaderModel <= (uint32_t)mSupportedShaderModel);
}

void Device::executeDeferredReleases()
{
    mpUploadHeap->executeDeferredReleases();
    uint64_t gpuValue = mpFrameFence->getGpuValue();
    while (mDeferredReleases.size() && mDeferredReleases.front().fenceValue <= gpuValue)
    {
        mDeferredReleases.pop();
    }

#if FALCOR_HAS_D3D12
    if (getType() == Type::D3D12)
    {
        mpD3D12CpuDescPool->executeDeferredReleases();
        mpD3D12GpuDescPool->executeDeferredReleases();
    }
#endif // FALCOR_HAS_D3D12
}

void Device::cleanup()
{
    mpRenderContext->flush(true);
    // Release all the bound resources. Need to do that before deleting the RenderContext
    mGfxCommandQueue.setNull();
    mDeferredReleases = decltype(mDeferredReleases)();
    mpRenderContext.reset();
    mpUploadHeap.reset();
    for (size_t i = 0; i < kInFlightFrameCount; ++i)
        mpTransientResourceHeaps[i].setNull();

    mpFrameFence.reset();
    for (auto& heap : mTimestampQueryHeaps)
        heap.reset();

#if FALCOR_HAS_D3D12
    mpD3D12CpuDescPool.reset();
    mpD3D12GpuDescPool.reset();
#endif // FALCOR_HAS_D3D12

    mpProfiler.reset();
    mpProgramManager.reset();

    mDeferredReleases = decltype(mDeferredReleases)();
}

void Device::flushAndSync()
{
    mpRenderContext->flush(true);
    mpFrameFence->gpuSignal(mpRenderContext->getLowLevelData()->getCommandQueue());
    executeDeferredReleases();
}

void Device::requireD3D12() const
{
    if (getType() != Type::D3D12)
        throw RuntimeError("D3D12 device is required.");
}

void Device::requireVulkan() const
{
    if (getType() != Type::Vulkan)
        throw RuntimeError("Vulkan device is required.");
}

ResourceBindFlags Device::getFormatBindFlags(ResourceFormat format)
{
    gfx::ResourceStateSet stateSet;
    FALCOR_GFX_CALL(mGfxDevice->getFormatSupportedResourceStates(getGFXFormat(format), &stateSet));

    ResourceBindFlags flags = ResourceBindFlags::None;
    if (stateSet.contains(gfx::ResourceState::ConstantBuffer))
    {
        flags |= ResourceBindFlags::Constant;
    }
    if (stateSet.contains(gfx::ResourceState::VertexBuffer))
    {
        flags |= ResourceBindFlags::Vertex;
    }
    if (stateSet.contains(gfx::ResourceState::IndexBuffer))
    {
        flags |= ResourceBindFlags::Index;
    }
    if (stateSet.contains(gfx::ResourceState::IndirectArgument))
    {
        flags |= ResourceBindFlags::IndirectArg;
    }
    if (stateSet.contains(gfx::ResourceState::StreamOutput))
    {
        flags |= ResourceBindFlags::StreamOutput;
    }
    if (stateSet.contains(gfx::ResourceState::ShaderResource))
    {
        flags |= ResourceBindFlags::ShaderResource;
    }
    if (stateSet.contains(gfx::ResourceState::RenderTarget))
    {
        flags |= ResourceBindFlags::RenderTarget;
    }
    if (stateSet.contains(gfx::ResourceState::DepthRead) || stateSet.contains(gfx::ResourceState::DepthWrite))
    {
        flags |= ResourceBindFlags::DepthStencil;
    }
    if (stateSet.contains(gfx::ResourceState::UnorderedAccess))
    {
        flags |= ResourceBindFlags::UnorderedAccess;
    }
    if (stateSet.contains(gfx::ResourceState::AccelerationStructure))
    {
        flags |= ResourceBindFlags::AccelerationStructure;
    }
    flags |= ResourceBindFlags::Shared;
    return flags;
}

void Device::reportLiveObjects()
{
    gfx::gfxReportLiveObjects();
}

bool Device::enableAgilitySDK()
{
#if FALCOR_WINDOWS && FALCOR_HAS_D3D12 && FALCOR_HAS_D3D12_AGILITY_SDK
    std::filesystem::path exeDir = getExecutableDirectory();
    std::filesystem::path sdkDir = getRuntimeDirectory() / FALCOR_D3D12_AGILITY_SDK_PATH;

    // Agility SDK can only be loaded from a relative path to the executable. Make sure both paths use the same driver letter.
    if (std::tolower(exeDir.string()[0]) != std::tolower(sdkDir.string()[0]))
    {
        logWarning(
            "Cannot enable D3D12 Agility SDK: Executable directory '{}' is not on the same drive as the SDK directory '{}'.", exeDir, sdkDir
        );
        return false;
    }

    // Get relative path and make sure there is the required trailing path delimiter.
    auto relPath = std::filesystem::relative(sdkDir, exeDir) / "";

    // Get the D3D12GetInterface procedure.
    typedef HRESULT(WINAPI * D3D12GetInterfaceFn)(REFCLSID rclsid, REFIID riid, void** ppvDebug);
    HMODULE handle = GetModuleHandleA("d3d12.dll");
    D3D12GetInterfaceFn pD3D12GetInterface = handle ? (D3D12GetInterfaceFn)GetProcAddress(handle, "D3D12GetInterface") : nullptr;
    if (!pD3D12GetInterface)
    {
        logWarning("Cannot enable D3D12 Agility SDK: Failed to get D3D12GetInterface.");
        return false;
    }

    // Local definition of CLSID_D3D12SDKConfiguration from d3d12.h
    const GUID CLSID_D3D12SDKConfiguration__ = {0x7cda6aca, 0xa03e, 0x49c8, {0x94, 0x58, 0x03, 0x34, 0xd2, 0x0e, 0x07, 0xce}};
    // Get the D3D12SDKConfiguration interface.
    FALCOR_MAKE_SMART_COM_PTR(ID3D12SDKConfiguration);
    ID3D12SDKConfigurationPtr pD3D12SDKConfiguration;
    if (!SUCCEEDED(pD3D12GetInterface(CLSID_D3D12SDKConfiguration__, IID_PPV_ARGS(&pD3D12SDKConfiguration))))
    {
        logWarning("Cannot enable D3D12 Agility SDK: Failed to get D3D12SDKConfiguration interface.");
        return false;
    }

    // Set the SDK version and path.
    if (!SUCCEEDED(pD3D12SDKConfiguration->SetSDKVersion(FALCOR_D3D12_AGILITY_SDK_VERSION, relPath.string().c_str())))
    {
        logWarning("Cannot enable D3D12 Agility SDK: Calling SetSDKVersion failed.");
        return false;
    }

    return true;
#endif
    return false;
}

std::vector<gfx::AdapterInfo> Device::getGPUs(Type deviceType)
{
    if (deviceType == Type::Default)
        deviceType = getDefaultDeviceType();
    auto adapters = gfx::gfxGetAdapters(getGfxDeviceType(deviceType));
    std::vector<gfx::AdapterInfo> result;
    for (gfx::GfxIndex i = 0; i < adapters.getCount(); ++i)
        result.push_back(adapters.getAdapters()[i]);
    // Move all NVIDIA adapters to the start of the list.
    std::stable_partition(
        result.begin(), result.end(),
        [](const gfx::AdapterInfo& info) { return toLowerCase(info.name).find("nvidia") != std::string::npos; }
    );
    return result;
}

gfx::ITransientResourceHeap* Device::getCurrentTransientResourceHeap()
{
    return mpTransientResourceHeaps[mCurrentTransientResourceHeapIndex].get();
}

void Device::endFrame()
{
    mpRenderContext->flush();

    // Wait on past frames.
    if (mpFrameFence->getCpuValue() >= kInFlightFrameCount)
        mpFrameFence->syncCpu(mpFrameFence->getCpuValue() - kInFlightFrameCount);

    // Switch to next transient resource heap.
    mCurrentTransientResourceHeapIndex = (mCurrentTransientResourceHeapIndex + 1) % kInFlightFrameCount;
    mpRenderContext->getLowLevelData()->closeCommandBuffer();
    getCurrentTransientResourceHeap()->synchronizeAndReset();
    mpRenderContext->getLowLevelData()->openCommandBuffer();

    // Signal frame fence for new frame.
    mpFrameFence->gpuSignal(mpRenderContext->getLowLevelData()->getCommandQueue());

    // Release resources from past frames.
    executeDeferredReleases();
}

NativeHandle Device::getNativeHandle(uint32_t index) const
{
    gfx::IDevice::InteropHandles gfxInteropHandles = {};
    FALCOR_GFX_CALL(mGfxDevice->getNativeDeviceHandles(&gfxInteropHandles));

#if FALCOR_HAS_D3D12
    if (getType() == Device::Type::D3D12)
    {
        if (index == 0)
            return NativeHandle(reinterpret_cast<ID3D12Device*>(gfxInteropHandles.handles[0].handleValue));
    }
#endif
#if FALCOR_HAS_VULKAN
    if (getType() == Device::Type::Vulkan)
    {
        if (index == 0)
            return NativeHandle(reinterpret_cast<VkInstance>(gfxInteropHandles.handles[0].handleValue));
        else if (index == 1)
            return NativeHandle(reinterpret_cast<VkPhysicalDevice>(gfxInteropHandles.handles[1].handleValue));
        else if (index == 2)
            return NativeHandle(reinterpret_cast<VkDevice>(gfxInteropHandles.handles[2].handleValue));
    }
#endif
    return {};
}

FALCOR_SCRIPT_BINDING(Device)
{
    pybind11::enum_<Device::Type> deviceType(m, "DeviceType");
    deviceType.value("Default", Device::Type::Default);
    deviceType.value("D3D12", Device::Type::D3D12);
    deviceType.value("Vulkan", Device::Type::Vulkan);

    ScriptBindings::SerializableStruct<Device::Desc> deviceDesc(m, "DeviceDesc");
#define field(f_) field(#f_, &Device::Desc::f_)
    deviceDesc.field(type);
    deviceDesc.field(gpu);
    deviceDesc.field(enableDebugLayer);
#undef field
}
} // namespace Falcor
