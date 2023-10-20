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
#include "ComputeStateObject.h"
#include "GraphicsStateObject.h"
#include "RtStateObject.h"
#include "NativeHandleTraits.h"
#include "Aftermath.h"
#include "PythonHelpers.h"
#include "Core/Macros.h"
#include "Core/Error.h"
#include "Core/ObjectPython.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramManager.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Timing/Profiler.h"

#if FALCOR_HAS_CUDA
#include "Utils/CudaUtils.h"
#endif

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
static_assert(sizeof(AdapterLUID) == sizeof(gfx::AdapterLUID));

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

static const size_t kConstantBufferDataPlacementAlignment = 256;
// This actually depends on the size of the index, but we can handle losing 2 bytes
static const size_t kIndexBufferDataPlacementAlignment = 4;

/// The default Shader Model to use when compiling programs.
/// If not supported, the highest supported shader model will be used instead.
static const ShaderModel kDefaultShaderModel = ShaderModel::SM6_6;

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
            FALCOR_THROW("Failed to initialize NVAPI.");
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
                pD3D12Device,
                reinterpret_cast<D3D12_COMPUTE_PIPELINE_STATE_DESC*>(pipelineDesc),
                1,
                ppPSOExtensionsDesc,
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
                pD3D12Device,
                reinterpret_cast<D3D12_GRAPHICS_PIPELINE_STATE_DESC*>(pipelineDesc),
                1,
                ppPSOExtensionsDesc,
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

    virtual gfx::Result createMeshPipelineState(
        gfx::IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState
    )
    {
        FALCOR_THROW("Mesh pipelines are not supported.");
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
                FALCOR_THROW("Failed to set NvApi extension");
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
                FALCOR_THROW("Failed to set NvApi extension");
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
    FALCOR_THROW("No default device type");
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
        FALCOR_THROW("Unknown device type");
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

inline ShaderModel querySupportedShaderModel(gfx::IDevice* pDevice)
{
    struct SMLevel
    {
        const char* name;
        ShaderModel level;
    };
    const SMLevel levels[] = {
        {"sm_6_7", ShaderModel::SM6_7},
        {"sm_6_6", ShaderModel::SM6_6},
        {"sm_6_5", ShaderModel::SM6_5},
        {"sm_6_4", ShaderModel::SM6_4},
        {"sm_6_3", ShaderModel::SM6_3},
        {"sm_6_2", ShaderModel::SM6_2},
        {"sm_6_1", ShaderModel::SM6_1},
        {"sm_6_0", ShaderModel::SM6_0},
    };
    for (auto level : levels)
    {
        if (pDevice->hasFeature(level.name))
        {
            return level.level;
        }
    }
    return ShaderModel::Unknown;
}

Device::Device(const Desc& desc) : mDesc(desc)
{
    if (mDesc.enableAftermath)
    {
#if FALCOR_HAS_AFTERMATH
        // Aftermath is incompatible with debug layers, so lets disable them.
        mDesc.enableDebugLayer = false;
        enableAftermath();
#else
        logWarning("Falcor was compiled without Aftermath support. Aftermath is disabled");
#endif
    }

    // Create a global slang session passed to GFX and used for compiling programs in ProgramManager.
    slang::createGlobalSession(mSlangGlobalSession.writeRef());

    if (mDesc.type == Type::Default)
        mDesc.type = getDefaultDeviceType();

#if !FALCOR_HAS_D3D12
    if (mDesc.type == Type::D3D12)
        FALCOR_THROW("D3D12 device not supported.");
#endif
#if !FALCOR_HAS_VULKAN
    if (mDesc.type == Type::Vulkan)
        FALCOR_THROW("Vulkan device not supported.");
#endif

    gfx::IDevice::Desc gfxDesc = {};
    gfxDesc.deviceType = getGfxDeviceType(mDesc.type);
    gfxDesc.slang.slangGlobalSession = mSlangGlobalSession;

    // Setup shader cache.
    gfxDesc.shaderCache.maxEntryCount = mDesc.maxShaderCacheEntryCount;
    if (mDesc.shaderCachePath == "")
    {
        gfxDesc.shaderCache.shaderCachePath = nullptr;
    }
    else
    {
        gfxDesc.shaderCache.shaderCachePath = mDesc.shaderCachePath.c_str();
        // If the supplied shader cache path does not exist, we will need to create it before creating the device.
        if (std::filesystem::exists(mDesc.shaderCachePath))
        {
            if (!std::filesystem::is_directory(mDesc.shaderCachePath))
                FALCOR_THROW("Shader cache path {} exists and is not a directory", mDesc.shaderCachePath);
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
    if (gfxDesc.deviceType == gfx::DeviceType::DirectX12)
        extendedDescs.push_back(&experimentalFeaturesDesc);
#endif
    gfxDesc.extendedDescCount = extendedDescs.size();
    gfxDesc.extendedDescs = extendedDescs.data();

#if FALCOR_NVAPI_AVAILABLE
    mpAPIDispatcher.reset(new PipelineCreationAPIDispatcher());
    gfxDesc.apiCommandDispatcher = static_cast<ISlangUnknown*>(mpAPIDispatcher.get());
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
        gfxDesc.adapterLUID = reinterpret_cast<const gfx::AdapterLUID*>(&gpus[mDesc.gpu].luid);
        if (SLANG_FAILED(gfxCreateDevice(&gfxDesc, mGfxDevice.writeRef())))
            logWarning("Failed to create device on GPU {} ({}).", mDesc.gpu, gpus[mDesc.gpu].name);
    }

    // Otherwise try create device on any available GPU.
    if (!mGfxDevice)
    {
        gfxDesc.adapterLUID = nullptr;
        if (SLANG_FAILED(gfxCreateDevice(&gfxDesc, mGfxDevice.writeRef())))
            FALCOR_THROW("Failed to create device");
    }

    const auto& deviceInfo = mGfxDevice->getDeviceInfo();
    mInfo.adapterName = deviceInfo.adapterName;
    mInfo.adapterLUID = gfxDesc.adapterLUID ? gpus[mDesc.gpu].luid : AdapterLUID();
    mInfo.apiName = deviceInfo.apiName;
    mLimits = queryLimits(mGfxDevice);
    mSupportedFeatures = querySupportedFeatures(mGfxDevice);

#if FALCOR_HAS_AFTERMATH
    if (mDesc.enableAftermath)
    {
        mpAftermathContext = std::make_unique<AftermathContext>(this);
        mpAftermathContext->initialize();
    }
#endif

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
    if (getType() == Type::Vulkan)
    {
        // Vulkan always supports SER.
        mSupportedFeatures |= SupportedFeatures::ShaderExecutionReorderingAPI;
    }

    mSupportedShaderModel = querySupportedShaderModel(mGfxDevice);
    mDefaultShaderModel = std::min(kDefaultShaderModel, mSupportedShaderModel);
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
            FALCOR_THROW("Failed to create transient resource heap");
    }

    gfx::ICommandQueue::Desc queueDesc = {};
    queueDesc.type = gfx::ICommandQueue::QueueType::Graphics;
    if (SLANG_FAILED(mGfxDevice->createCommandQueue(queueDesc, mGfxCommandQueue.writeRef())))
        FALCOR_THROW("Failed to create command queue");

    // The Device class contains a bunch of nested resource objects that have strong references to the device.
    // This is because we want a strong reference to the device when those objects are returned to the user.
    // However, here it immediately creates cyclic references device->resource->device upon creation of the device.
    // To break the cycles, we break the strong reference to the device for the resources that it owns.

    // Here, we temporarily increase the refcount of the device, so it won't be destroyed upon breaking the
    // nested strong references to it.
    this->incRef();

#if FALCOR_ENABLE_REF_TRACKING
    this->setEnableRefTracking(true);
#endif

    mpFrameFence = createFence();
    mpFrameFence->breakStrongReferenceToDevice();

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

    mpProgramManager = std::make_unique<ProgramManager>(this);

    mpProfiler = std::make_unique<Profiler>(ref<Device>(this));
    mpProfiler->breakStrongReferenceToDevice();

    mpDefaultSampler = createSampler(Sampler::Desc());
    mpDefaultSampler->breakStrongReferenceToDevice();

    mpUploadHeap = GpuMemoryHeap::create(ref<Device>(this), MemoryType::Upload, 1024 * 1024 * 2, mpFrameFence);
    mpUploadHeap->breakStrongReferenceToDevice();

    mpReadBackHeap = GpuMemoryHeap::create(ref<Device>(this), MemoryType::ReadBack, 1024 * 1024 * 2, mpFrameFence);
    mpReadBackHeap->breakStrongReferenceToDevice();

    mpTimestampQueryHeap = QueryHeap::create(ref<Device>(this), QueryHeap::Type::Timestamp, 1024 * 1024);
    mpTimestampQueryHeap->breakStrongReferenceToDevice();

    mpRenderContext = std::make_unique<RenderContext>(this, mGfxCommandQueue);

    // TODO: Do we need to flush here or should RenderContext::create() bind the descriptor heaps automatically without flush? See #749.
    mpRenderContext->submit(); // This will bind the descriptor heaps.

    this->decRef(false);

    logInfo(
        "Created GPU device '{}' using '{}' API (SM{}.{}).",
        mInfo.adapterName,
        mInfo.apiName,
        getShaderModelMajorVersion(mSupportedShaderModel),
        getShaderModelMinorVersion(mSupportedShaderModel)
    );
}

Device::~Device()
{
    mpRenderContext->submit(true);

    mpProfiler.reset();

    // Release all the bound resources. Need to do that before deleting the RenderContext
    mGfxCommandQueue.setNull();
    mDeferredReleases = decltype(mDeferredReleases)();
    mpRenderContext.reset();
    mpUploadHeap.reset();
    mpReadBackHeap.reset();
    mpTimestampQueryHeap.reset();
    for (size_t i = 0; i < kInFlightFrameCount; ++i)
        mpTransientResourceHeaps[i].setNull();

    mpDefaultSampler.reset();
    mpFrameFence.reset();

#if FALCOR_HAS_D3D12
    mpD3D12CpuDescPool.reset();
    mpD3D12GpuDescPool.reset();
#endif // FALCOR_HAS_D3D12

    mpProgramManager.reset();

    mDeferredReleases = decltype(mDeferredReleases)();


    mGfxDevice.setNull();

#if FALCOR_NVAPI_AVAILABLE
    mpAPIDispatcher.reset();
#endif
}

ref<Buffer> Device::createBuffer(size_t size, ResourceBindFlags bindFlags, MemoryType memoryType, const void* pInitData)
{
    return make_ref<Buffer>(ref<Device>(this), size, bindFlags, memoryType, pInitData);
}

ref<Buffer> Device::createTypedBuffer(
    ResourceFormat format,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    MemoryType memoryType,
    const void* pInitData
)
{
    return make_ref<Buffer>(ref<Device>(this), format, elementCount, bindFlags, memoryType, pInitData);
}

ref<Buffer> Device::createStructuredBuffer(
    uint32_t structSize,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    MemoryType memoryType,
    const void* pInitData,
    bool createCounter
)
{
    return make_ref<Buffer>(ref<Device>(this), structSize, elementCount, bindFlags, memoryType, pInitData, createCounter);
}

ref<Buffer> Device::createStructuredBuffer(
    const ReflectionType* pType,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    MemoryType memoryType,
    const void* pInitData,
    bool createCounter
)
{
    FALCOR_CHECK(pType != nullptr, "Can't create a structured buffer from a nullptr type.");
    const ReflectionResourceType* pResourceType = pType->unwrapArray()->asResourceType();
    if (!pResourceType || pResourceType->getType() != ReflectionResourceType::Type::StructuredBuffer)
    {
        FALCOR_THROW("Can't create a structured buffer from type '{}'.", pType->getClassName());
    }

    FALCOR_ASSERT(pResourceType->getSize() <= std::numeric_limits<uint32_t>::max());
    return make_ref<Buffer>(
        ref<Device>(this), (uint32_t)pResourceType->getSize(), elementCount, bindFlags, memoryType, pInitData, createCounter
    );
}

ref<Buffer> Device::createStructuredBuffer(
    const ShaderVar& shaderVar,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    MemoryType memoryType,
    const void* pInitData,
    bool createCounter
)
{
    return createStructuredBuffer(shaderVar.getType(), elementCount, bindFlags, memoryType, pInitData, createCounter);
}

ref<Buffer> Device::createBufferFromResource(
    gfx::IBufferResource* pResource,
    size_t size,
    ResourceBindFlags bindFlags,
    MemoryType memoryType
)
{
    return make_ref<Buffer>(ref<Device>(this), pResource, size, bindFlags, memoryType);
}

ref<Buffer> Device::createBufferFromNativeHandle(NativeHandle handle, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType)
{
    return make_ref<Buffer>(ref<Device>(this), handle, size, bindFlags, memoryType);
}

ref<Texture> Device::createTexture1D(
    uint32_t width,
    ResourceFormat format,
    uint32_t arraySize,
    uint32_t mipLevels,
    const void* pInitData,
    ResourceBindFlags bindFlags
)
{
    return make_ref<Texture>(
        ref<Device>(this), Resource::Type::Texture1D, format, width, 1, 1, arraySize, mipLevels, 1, bindFlags, pInitData
    );
}

ref<Texture> Device::createTexture2D(
    uint32_t width,
    uint32_t height,
    ResourceFormat format,
    uint32_t arraySize,
    uint32_t mipLevels,
    const void* pInitData,
    ResourceBindFlags bindFlags
)
{
    return make_ref<Texture>(
        ref<Device>(this), Resource::Type::Texture2D, format, width, height, 1, arraySize, mipLevels, 1, bindFlags, pInitData
    );
}

ref<Texture> Device::createTexture3D(
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    ResourceFormat format,
    uint32_t mipLevels,
    const void* pInitData,
    ResourceBindFlags bindFlags
)
{
    return make_ref<Texture>(
        ref<Device>(this), Resource::Type::Texture3D, format, width, height, depth, 1, mipLevels, 1, bindFlags, pInitData
    );
}

ref<Texture> Device::createTextureCube(
    uint32_t width,
    uint32_t height,
    ResourceFormat format,
    uint32_t arraySize,
    uint32_t mipLevels,
    const void* pInitData,
    ResourceBindFlags bindFlags
)
{
    return make_ref<Texture>(
        ref<Device>(this), Resource::Type::TextureCube, format, width, height, 1, arraySize, mipLevels, 1, bindFlags, pInitData
    );
}

ref<Texture> Device::createTexture2DMS(
    uint32_t width,
    uint32_t height,
    ResourceFormat format,
    uint32_t sampleCount,
    uint32_t arraySize,
    ResourceBindFlags bindFlags
)
{
    return make_ref<Texture>(
        ref<Device>(this), Resource::Type::Texture2DMultisample, format, width, height, 1, arraySize, 1, sampleCount, bindFlags, nullptr
    );
}

ref<Texture> Device::createTextureFromResource(
    gfx::ITextureResource* pResource,
    Texture::Type type,
    ResourceFormat format,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t arraySize,
    uint32_t mipLevels,
    uint32_t sampleCount,
    ResourceBindFlags bindFlags,
    Resource::State initState
)
{
    return make_ref<Texture>(
        ref<Device>(this), pResource, type, format, width, height, depth, arraySize, mipLevels, sampleCount, bindFlags, initState
    );
}

ref<Sampler> Device::createSampler(const Sampler::Desc& desc)
{
    return make_ref<Sampler>(ref<Device>(this), desc);
}

ref<Fence> Device::createFence(const FenceDesc& desc)
{
    return make_ref<Fence>(ref<Device>(this), desc);
}

ref<Fence> Device::createFence(bool shared)
{
    FenceDesc desc;
    desc.shared = shared;
    return createFence(desc);
}

ref<ComputeStateObject> Device::createComputeStateObject(const ComputeStateObjectDesc& desc)
{
    return make_ref<ComputeStateObject>(ref<Device>(this), desc);
}

ref<GraphicsStateObject> Device::createGraphicsStateObject(const GraphicsStateObjectDesc& desc)
{
    return make_ref<GraphicsStateObject>(ref<Device>(this), desc);
}

ref<RtStateObject> Device::createRtStateObject(const RtStateObjectDesc& desc)
{
    return make_ref<RtStateObject>(ref<Device>(this), desc);
}

size_t Device::getBufferDataAlignment(ResourceBindFlags bindFlags)
{
    if (is_set(bindFlags, ResourceBindFlags::Constant))
        return kConstantBufferDataPlacementAlignment;
    if (is_set(bindFlags, ResourceBindFlags::Index))
        return kIndexBufferDataPlacementAlignment;
    return 1;
}

void Device::releaseResource(ISlangUnknown* pResource)
{
    if (pResource)
    {
        // Some static objects get here when the application exits
        if (this)
        {
            mDeferredReleases.push({mpFrameFence ? mpFrameFence->getSignaledValue() : 0, Slang::ComPtr<ISlangUnknown>(pResource)});
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
    mpReadBackHeap->executeDeferredReleases();
    uint64_t currentValue = mpFrameFence->getCurrentValue();
    while (mDeferredReleases.size() && mDeferredReleases.front().fenceValue < currentValue)
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

void Device::wait()
{
    mpRenderContext->submit(true);
    mpRenderContext->signal(mpFrameFence.get());
    executeDeferredReleases();
}

void Device::requireD3D12() const
{
    if (getType() != Type::D3D12)
        FALCOR_THROW("D3D12 device is required.");
}

void Device::requireVulkan() const
{
    if (getType() != Type::Vulkan)
        FALCOR_THROW("Vulkan device is required.");
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

size_t Device::getTextureRowAlignment() const
{
    size_t alignment = 1;
    mGfxDevice->getTextureRowAlignment(&alignment);
    return alignment;
}

#if FALCOR_HAS_CUDA

bool Device::initCudaDevice()
{
    return getCudaDevice() != nullptr;
}

cuda_utils::CudaDevice* Device::getCudaDevice() const
{
    if (!mpCudaDevice)
        mpCudaDevice = make_ref<cuda_utils::CudaDevice>(this);
    return mpCudaDevice.get();
}

#endif

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

std::vector<AdapterInfo> Device::getGPUs(Type deviceType)
{
    if (deviceType == Type::Default)
        deviceType = getDefaultDeviceType();
    auto adapters = gfx::gfxGetAdapters(getGfxDeviceType(deviceType));
    std::vector<AdapterInfo> result;
    for (gfx::GfxIndex i = 0; i < adapters.getCount(); ++i)
    {
        const gfx::AdapterInfo& gfxInfo = adapters.getAdapters()[i];
        AdapterInfo info;
        info.name = gfxInfo.name;
        info.vendorID = gfxInfo.vendorID;
        info.deviceID = gfxInfo.deviceID;
        info.luid = *reinterpret_cast<const AdapterLUID*>(&gfxInfo.luid);
        result.push_back(info);
    }
    // Move all NVIDIA adapters to the start of the list.
    std::stable_partition(
        result.begin(), result.end(), [](const AdapterInfo& info) { return toLowerCase(info.name).find("nvidia") != std::string::npos; }
    );
    return result;
}

gfx::ITransientResourceHeap* Device::getCurrentTransientResourceHeap()
{
    return mpTransientResourceHeaps[mCurrentTransientResourceHeapIndex].get();
}

void Device::endFrame()
{
    mpRenderContext->submit();

    // Wait on past frames.
    if (mpFrameFence->getSignaledValue() > kInFlightFrameCount)
        mpFrameFence->wait(mpFrameFence->getSignaledValue() - kInFlightFrameCount);

    // Switch to next transient resource heap.
    getCurrentTransientResourceHeap()->finish();
    mCurrentTransientResourceHeapIndex = (mCurrentTransientResourceHeapIndex + 1) % kInFlightFrameCount;
    mpRenderContext->getLowLevelData()->closeCommandBuffer();
    getCurrentTransientResourceHeap()->synchronizeAndReset();
    mpRenderContext->getLowLevelData()->openCommandBuffer();

    // Signal frame fence for new frame.
    mpRenderContext->signal(mpFrameFence.get());

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
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Formats)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Buffer)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Texture)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Sampler)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Profiler)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(RenderContext)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Program)

    pybind11::class_<Device, ref<Device>> device(m, "Device");

    pybind11::enum_<Device::Type> deviceType(m, "DeviceType");
    deviceType.value("Default", Device::Type::Default);
    deviceType.value("D3D12", Device::Type::D3D12);
    deviceType.value("Vulkan", Device::Type::Vulkan);

    pybind11::class_<Device::Info> info(device, "Info");
    info.def_readonly("adapter_name", &Device::Info::adapterName);
    info.def_readonly("api_name", &Device::Info::apiName);

    pybind11::class_<Device::Limits> limits(device, "Limits");
    limits.def_readonly("max_compute_dispatch_thread_groups", &Device::Limits::maxComputeDispatchThreadGroups);
    limits.def_readonly("max_shader_visible_samplers", &Device::Limits::maxShaderVisibleSamplers);

    device.def(
        pybind11::init(
            [](Device::Type type, uint32_t gpu, bool enable_debug_layer, bool enable_aftermath)
            {
                Device::Desc desc;
                desc.type = type;
                desc.gpu = gpu;
                desc.enableDebugLayer = enable_debug_layer;
                desc.enableAftermath = enable_aftermath;
                return make_ref<Device>(desc);
            }
        ),
        "type"_a = Device::Type::Default,
        "gpu"_a = 0,
        "enable_debug_layer"_a = false,
        "enable_aftermath"_a = false
    );
    device.def(
        "create_buffer",
        [](Device& self, size_t size, ResourceBindFlags bind_flags, MemoryType memory_type)
        { return self.createBuffer(size, bind_flags, memory_type); },
        "size"_a,
        "bind_flags"_a = ResourceBindFlags::None,
        "memory_type"_a = MemoryType::DeviceLocal
    );
    device.def(
        "create_typed_buffer",
        [](Device& self, ResourceFormat format, size_t element_count, ResourceBindFlags bind_flags, MemoryType memory_type)
        { return self.createTypedBuffer(format, element_count, bind_flags, memory_type, nullptr); },
        "format"_a,
        "element_count"_a,
        "bind_flags"_a = ResourceBindFlags::None,
        "memory_type"_a = MemoryType::DeviceLocal
    );
    device.def(
        "create_structured_buffer",
        [](Device& self, size_t struct_size, size_t element_count, ResourceBindFlags bind_flags, MemoryType memory_type, bool create_counter
        ) { return self.createStructuredBuffer(struct_size, element_count, bind_flags, memory_type, nullptr, create_counter); },
        "struct_size"_a,
        "element_count"_a,
        "bind_flags"_a = ResourceBindFlags::None,
        "memory_type"_a = MemoryType::DeviceLocal,
        "create_counter"_a = false
    );

    device.def(
        "create_texture",
        [](Device& self,
           uint32_t width,
           uint32_t height,
           uint32_t depth,
           ResourceFormat format,
           uint32_t array_size,
           uint32_t mip_levels,
           ResourceBindFlags bind_flags)
        {
            if (depth > 0)
                return self.createTexture3D(width, height, depth, format, mip_levels, nullptr, bind_flags);
            else if (height > 0)
                return self.createTexture2D(width, height, format, array_size, mip_levels, nullptr, bind_flags);
            else
                return self.createTexture1D(width, format, array_size, mip_levels, nullptr, bind_flags);
        },
        "width"_a,
        "height"_a = 0,
        "depth"_a = 0,
        "format"_a = ResourceFormat::Unknown,
        "array_size"_a = 1,
        "mip_levels"_a = uint32_t(Texture::kMaxPossible),
        "bind_flags"_a = ResourceBindFlags::None
    );

    device.def(
        "create_sampler",
        [](Device& self,
           TextureFilteringMode mag_filter,
           TextureFilteringMode min_filter,
           TextureFilteringMode mip_filter,
           uint32_t max_anisotropy,
           float min_lod,
           float max_lod,
           float lod_bias,
           ComparisonFunc comparison_func,
           TextureReductionMode reduction_mode,
           TextureAddressingMode address_mode_u,
           TextureAddressingMode address_mode_v,
           TextureAddressingMode address_mode_w,
           float4 border_color)
        {
            Sampler::Desc desc;
            desc.setFilterMode(mag_filter, min_filter, mip_filter);
            desc.setMaxAnisotropy(max_anisotropy);
            desc.setLodParams(min_lod, max_lod, lod_bias);
            desc.setComparisonFunc(comparison_func);
            desc.setReductionMode(reduction_mode);
            desc.setAddressingMode(address_mode_u, address_mode_v, address_mode_w);
            desc.setBorderColor(border_color);
            return self.createSampler(desc);
        },
        "mag_filter"_a = TextureFilteringMode::Linear,
        "min_filter"_a = TextureFilteringMode::Linear,
        "mip_filter"_a = TextureFilteringMode::Linear,
        "max_anisotropy"_a = 1,
        "min_lod"_a = -1000.0f,
        "max_lod"_a = 1000.0f,
        "lod_bias"_a = 0.f,
        "comparison_func"_a = ComparisonFunc::Disabled,
        "reduction_mode"_a = TextureReductionMode::Standard,
        "address_mode_u"_a = TextureAddressingMode::Wrap,
        "address_mode_v"_a = TextureAddressingMode::Wrap,
        "address_mode_w"_a = TextureAddressingMode::Wrap,
        "border_color_r"_a = float4(0.f)
    );

    device.def(
        "create_program",
        [](ref<Device> self, std::optional<ProgramDesc> desc, pybind11::dict defines, const pybind11::kwargs& kwargs)
        {
            if (desc)
            {
                FALCOR_CHECK(kwargs.empty(), "Either provide a 'desc' or kwargs, but not both.");
                return Program::create(self, *desc, defineListFromPython(defines));
            }
            else
            {
                return Program::create(self, programDescFromPython(kwargs), defineListFromPython(defines));
            }
        },
        "desc"_a = std::optional<ProgramDesc>(),
        "defines"_a = pybind11::dict(),
        pybind11::kw_only()
    );

    device.def("wait", &Device::wait);

    device.def_property_readonly("profiler", &Device::getProfiler);
    device.def_property_readonly("type", &Device::getType);
    device.def_property_readonly("info", &Device::getInfo);
    device.def_property_readonly("limits", &Device::getLimits);
    device.def_property_readonly("render_context", &Device::getRenderContext);
}
} // namespace Falcor
