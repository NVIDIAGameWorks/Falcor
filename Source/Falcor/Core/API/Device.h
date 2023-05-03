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
#pragma once
#include "Handles.h"
#include "NativeHandle.h"
#include "Formats.h"
#include "QueryHeap.h"
#include "LowLevelContextData.h"
#include "RenderContext.h"
#include "GpuMemoryHeap.h"
#include "Core/Macros.h"

#if FALCOR_HAS_D3D12
#include <guiddef.h>
#endif

#include <array>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace Falcor
{
#ifdef _DEBUG
#define FALCOR_DEFAULT_ENABLE_DEBUG_LAYER true
#else
#define FALCOR_DEFAULT_ENABLE_DEBUG_LAYER false
#endif

#if FALCOR_HAS_D3D12
class D3D12DescriptorPool;
#endif

class PipelineCreationAPIDispatcher;
class ProgramManager;
class Profiler;

class FALCOR_API Device : public std::enable_shared_from_this<Device>
{
public:
    /**
     * Maximum number of in-flight frames.
     * Typically there are at least two frames, one being rendered to, the other being presented.
     * We add one more to be on the save side.
     */
    static constexpr uint32_t kInFlightFrameCount = 3;

    /// Device type.
    enum Type
    {
        Default, ///< Default device type, favors D3D12 over Vulkan.
        D3D12,
        Vulkan,
    };

    /// Device descriptor.
    struct Desc
    {
        /// The device type (D3D12/Vulkan).
        Type type = Type::Default;

        /// GPU index (indexing into GPU list returned by getGPUList()).
        uint32_t gpu = 0;

        /// Enable the debug layer. The default for release build is false, for debug build it's true.
        bool enableDebugLayer = FALCOR_DEFAULT_ENABLE_DEBUG_LAYER;

        /// The maximum number of entries allowable in the shader cache. A value of 0 indicates no limit.
        uint32_t maxShaderCacheEntryCount = 1000;

        /// The full path to the root directory for the shader cache. An empty string will disable the cache.
        std::string shaderCachePath = (getRuntimeDirectory() / ".shadercache").string();

#if FALCOR_HAS_D3D12
        /// GUID list for experimental features
        std::vector<GUID> experimentalFeatures;
#endif
    };

    struct Limits
    {
        uint3 maxComputeDispatchThreadGroups;
        uint32_t maxShaderVisibleSamplers;
    };

    enum class SupportedFeatures
    {
        // clang-format off
        None = 0x0,
        ProgrammableSamplePositionsPartialOnly = 0x1,   ///< On D3D12, this means tier 1 support. Allows one sample position to be set.
        ProgrammableSamplePositionsFull = 0x2,          ///< On D3D12, this means tier 2 support. Allows up to 4 sample positions to be set.
        Barycentrics = 0x4,                             ///< On D3D12, pixel shader barycentrics are supported.
        Raytracing = 0x8,                               ///< On D3D12, DirectX Raytracing is supported. It is up to the user to not use raytracing functions when not supported.
        RaytracingTier1_1 = 0x10,                       ///< On D3D12, DirectX Raytracing Tier 1.1 is supported.
        ConservativeRasterizationTier1 = 0x20,          ///< On D3D12, conservative rasterization tier 1 is supported.
        ConservativeRasterizationTier2 = 0x40,          ///< On D3D12, conservative rasterization tier 2 is supported.
        ConservativeRasterizationTier3 = 0x80,          ///< On D3D12, conservative rasterization tier 3 is supported.
        RasterizerOrderedViews = 0x100,                 ///< On D3D12, rasterizer ordered views (ROVs) are supported.
        WaveOperations = 0x200,
        ShaderExecutionReorderingAPI = 0x400,           ///< On D3D12, this means SER API is available (in the future this will be part of the shader model).
        RaytracingReordering = 0x800,                   ///< On D3D12, this means SER is supported on the hardware.

        // clang-format on
    };

    enum class ShaderModel : uint32_t
    {
        Unknown,
        SM6_0,
        SM6_1,
        SM6_2,
        SM6_3,
        SM6_4,
        SM6_5,
        SM6_6,
        SM6_7,
    };

    /**
     * Create a new device.
     * @param[in] desc Device configuration descriptor.
     * @return A pointer to a new device object, or throws an exception if creation failed.
     */
    static std::shared_ptr<Device> create(const Desc& desc);

    Device(const Desc& desc);
    ~Device();

    /**
     * Acts as the destructor for Device. Some resources use the global device pointer their cleanup.
     */
    void cleanup();

    ProgramManager* getProgramManager() const { return mpProgramManager.get(); }

    Profiler* getProfiler() const { return mpProfiler.get(); }

    /**
     * Get the default render-context.
     * The default render-context is managed completely by the device. The user should just queue commands into it, the device will take
     * care of allocation, submission and synchronization
     */
    RenderContext* getRenderContext() const { return mpRenderContext.get(); }

    /// Returns the global slang session.
    slang::IGlobalSession* getSlangGlobalSession() const { return mSlangGlobalSession; }

    /// Return the GFX define.
    gfx::IDevice* getGfxDevice() const { return mGfxDevice; }

    /// Return the GFX command queue.
    gfx::ICommandQueue* getGfxCommandQueue() const { return mGfxCommandQueue; }

    /**
     * Returns the native API handle:
     * - D3D12: ID3D12Device* (0)
     * - Vulkan: VkInstance (0), VkPhysicalDevice (1), VkDevice (2)
     */
    NativeHandle getNativeHandle(uint32_t index = 0) const;

    /**
     * End a frame.
     * This closes the current command buffer, switches to a new heap for transient resources and opens a new command buffer.
     * This also executes deferred releases of resources from past frames.
     */
    void endFrame();

    /**
     * Flushes pipeline, releases resources, and blocks until completion
     */
    void flushAndSync();

    /**
     * Get the desc
     */
    const Desc& getDesc() const { return mDesc; }

    /**
     * Get the device type.
     */
    Type getType() const { return mDesc.type; }

    /**
     * Throws an exception if the device is not a D3D12 device.
     */
    void requireD3D12() const;

    /**
     * Throws an exception if the device is not a Vulkan device.
     */
    void requireVulkan() const;

    /**
     * Get an object that represents a default sampler.
     */
    std::shared_ptr<Sampler> getDefaultSampler() const { return mpDefaultSampler; }

    /**
     * Create a new query heap.
     * @param[in] type Type of queries.
     * @param[in] count Number of queries.
     * @return New query heap.
     */
    std::weak_ptr<QueryHeap> createQueryHeap(QueryHeap::Type type, uint32_t count);

#if FALCOR_HAS_D3D12
    const std::shared_ptr<D3D12DescriptorPool>& getD3D12CpuDescriptorPool() const
    {
        requireD3D12();
        return mpD3D12CpuDescPool;
    }
    const std::shared_ptr<D3D12DescriptorPool>& getD3D12GpuDescriptorPool() const
    {
        requireD3D12();
        return mpD3D12GpuDescPool;
    }
#endif // FALCOR_HAS_D3D12

    const GpuMemoryHeap::SharedPtr& getUploadHeap() const { return mpUploadHeap; }
    void releaseResource(ISlangUnknown* pResource);

    double getGpuTimestampFrequency() const { return mGpuTimestampFrequency; } // ms/tick

    /**
     * Get the device limits.
     */
    const Limits& getLimits() const { return mLimits; }

    /**
     * Check if features are supported by the device
     */
    bool isFeatureSupported(SupportedFeatures flags) const;

    /**
     * Check if a shader model is supported by the device
     */
    bool isShaderModelSupported(ShaderModel shaderModel) const;

    /**
     * Return the highest supported shader model by the device
     */
    ShaderModel getSupportedShaderModel() const { return mSupportedShaderModel; }

    gfx::ITransientResourceHeap* getCurrentTransientResourceHeap();

    /**
     * Get the supported bind-flags for a specific format.
     */
    ResourceBindFlags getFormatBindFlags(ResourceFormat format);

    /// Report live objects in GFX.
    /// This is useful for checking clean shutdown where all resources are properly released.
    static void reportLiveObjects();

    /**
     * Try to enable D3D12 Agility SDK at runtime.
     * Note: This must be called before creating a device to have any effect.
     *
     * Prefer adding FALCOR_EXPORT_D3D12_AGILITY_SDK to the main translation unit of executables
     * to tag the application binary to load the D3D12 Agility SDK.
     *
     * If loading Falcor as a library only (the case when loading Falcor as a Python module)
     * tagging the main application (Python interpreter) is not possible. The alternative is
     * to use the D3D12SDKConfiguration API introduced in Windows SDK 20348. This however
     * requires "Developer Mode" to be enabled and the executed Python interpreter to be
     * stored on the same drive as Falcor.
     *
     * @return Return true if D3D12 Agility SDK was successfully enabled.
     */
    static bool enableAgilitySDK();

    /**
     * Get a list of all available GPUs.
     */
    static std::vector<gfx::AdapterInfo> getGPUs(Type deviceType);

    /**
     * Get the global device mutex.
     * WARNING: Falcor is generally not thread-safe. This mutex is used in very specific
     * places only, currently only for doing parallel texture loading.
     */
    std::mutex& getGlobalGfxMutex() { return mGlobalGfxMutex; }

private:
    struct ResourceRelease
    {
        uint64_t fenceValue;
        Slang::ComPtr<ISlangUnknown> mObject;
    };
    std::queue<ResourceRelease> mDeferredReleases;

    bool init();
    void executeDeferredReleases();

    Desc mDesc;
    Slang::ComPtr<slang::IGlobalSession> mSlangGlobalSession;
    Slang::ComPtr<gfx::IDevice> mGfxDevice;
    Slang::ComPtr<gfx::ICommandQueue> mGfxCommandQueue;
    Slang::ComPtr<gfx::ITransientResourceHeap> mpTransientResourceHeaps[kInFlightFrameCount];
    uint32_t mCurrentTransientResourceHeapIndex = 0;

    std::shared_ptr<Sampler> mpDefaultSampler;

    GpuMemoryHeap::SharedPtr mpUploadHeap;
#if FALCOR_HAS_D3D12
    std::shared_ptr<D3D12DescriptorPool> mpD3D12CpuDescPool;
    std::shared_ptr<D3D12DescriptorPool> mpD3D12GpuDescPool;
#endif
    GpuFence::SharedPtr mpFrameFence;

    std::unique_ptr<RenderContext> mpRenderContext;
    std::list<QueryHeap::SharedPtr> mTimestampQueryHeaps;
    double mGpuTimestampFrequency;

    Limits mLimits;
    SupportedFeatures mSupportedFeatures = SupportedFeatures::None;
    ShaderModel mSupportedShaderModel = ShaderModel::Unknown;

#if FALCOR_NVAPI_AVAILABLE
    std::unique_ptr<PipelineCreationAPIDispatcher> mpAPIDispatcher;
#endif

    std::unique_ptr<ProgramManager> mpProgramManager;
    std::unique_ptr<Profiler> mpProfiler;

    std::mutex mGlobalGfxMutex;
};

inline constexpr uint32_t getMaxViewportCount()
{
    return 8;
}

/// !!! DO NOT USE THIS !!!
/// This is only available during the migration away from having only a single global GPU device in Falcor.
FALCOR_API std::shared_ptr<Device>& getGlobalDevice();

FALCOR_ENUM_CLASS_OPERATORS(Device::SupportedFeatures);
} // namespace Falcor
