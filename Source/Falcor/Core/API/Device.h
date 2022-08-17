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
#pragma once
#include "Handles.h"
#include "Formats.h"
#include "QueryHeap.h"
#include "LowLevelContextData.h"
#include "FBO.h"
#include "RenderContext.h"
#include "GpuMemoryHeap.h"
#include "Core/Macros.h"
#include "Core/Window.h"
#if FALCOR_HAS_D3D12
#include "Core/API/Shared/D3D12DescriptorPool.h"
#endif

#include <array>
#include <list>
#include <memory>
#include <queue>
#include <vector>

namespace Falcor
{
#ifdef _DEBUG
#define FALCOR_DEFAULT_ENABLE_DEBUG_LAYER true
#else
#define FALCOR_DEFAULT_ENABLE_DEBUG_LAYER false
#endif

    struct DeviceApiData;

    class FALCOR_API Device
    {
    public:
        using SharedPtr = std::shared_ptr<Device>;
        using SharedConstPtr = std::shared_ptr<const Device>;
        using ApiHandle = DeviceHandle;
        static const uint32_t kQueueTypeCount = (uint32_t)LowLevelContextData::CommandQueueType::Count;
        static constexpr uint32_t kSwapChainBuffersCount = 3;

        /** Device configuration
        */
        struct Desc
        {
            ResourceFormat colorFormat = ResourceFormat::BGRA8UnormSrgb;    ///< The color buffer format
            ResourceFormat depthFormat = ResourceFormat::D32Float;          ///< The depth buffer format
            uint32_t apiMajorVersion = 0;                                   ///< Requested API major version. If specified, device creation will fail if not supported. Otherwise, the highest supported version will be automatically selected.
            uint32_t apiMinorVersion = 0;                                   ///< Requested API minor version. If specified, device creation will fail if not supported. Otherwise, the highest supported version will be automatically selected.
            bool enableVsync = false;                                       ///< Controls vertical-sync
            bool enableDebugLayer = FALCOR_DEFAULT_ENABLE_DEBUG_LAYER;      ///< Enable the debug layer. The default for release build is false, for debug build it's true.

            static_assert((uint32_t)LowLevelContextData::CommandQueueType::Direct == 2, "Default initialization of cmdQueues assumes that Direct queue index is 2");
            std::array<uint32_t, kQueueTypeCount> cmdQueues = { 0, 0, 1 };  ///< Command queues to create. If no direct-queues are created, mpRenderContext will not be initialized

#ifdef FALCOR_D3D12
            // GUID list for experimental features
            std::vector<UUID> experimentalFeatures;
#endif
        };

        enum class SupportedFeatures
        {
            None = 0x0,
            ProgrammableSamplePositionsPartialOnly = 0x1, // On D3D12, this means tier 1 support. Allows one sample position to be set.
            ProgrammableSamplePositionsFull = 0x2,        // On D3D12, this means tier 2 support. Allows up to 4 sample positions to be set.
            Barycentrics = 0x4,                           // On D3D12, pixel shader barycentrics are supported.
            Raytracing = 0x8,                             // On D3D12, DirectX Raytracing is supported. It is up to the user to not use raytracing functions when not supported.
            RaytracingTier1_1 = 0x10,                     // On D3D12, DirectX Raytracing Tier 1.1 is supported.
            ConservativeRasterizationTier1 = 0x20,        // On D3D12, conservative rasterization tier 1 is supported.
            ConservativeRasterizationTier2 = 0x40,        // On D3D12, conservative rasterization tier 2 is supported.
            ConservativeRasterizationTier3 = 0x80,        // On D3D12, conservative rasterization tier 3 is supported.
            RasterizerOrderedViews = 0x100,               // On D3D12, rasterizer ordered views (ROVs) are supported.
            WaveOperations = 0x200,
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

        ~Device();

        /** Create a new device.
            \param[in] pWindow a previously-created window object
            \param[in] desc Device configuration descriptor.
            \return A pointer to a new device object, or throws an exception if creation failed.
        */
        static SharedPtr create(Window::SharedPtr& pWindow, const Desc& desc);

        /** Acts as the destructor for Device. Some resources use gpDevice in their cleanup.
            Cleaning up the SharedPtr directly would clear gpDevice before calling destructors.
        */
        void cleanup();

        /** Enable/disable vertical sync
        */
        void toggleVSync(bool enable);

        /** Check if the window is occluded
        */
        bool isWindowOccluded() const;

        /** Get the FBO object associated with the swap-chain.
            This can change each frame, depending on the API used
        */
        Fbo::SharedPtr getSwapChainFbo() const;

        /** Get the default render-context.
            The default render-context is managed completely by the device. The user should just queue commands into it, the device will take care of allocation, submission and synchronization
        */
        RenderContext* getRenderContext() const { return mpRenderContext.get(); }

        /** Get the command queue handle
        */
        CommandQueueHandle getCommandQueueHandle(LowLevelContextData::CommandQueueType type, uint32_t index) const;

        /** Get the API queue type.
            \return API queue type, or throws an exception if type is unknown.
        */
        ApiCommandQueueType getApiCommandQueueType(LowLevelContextData::CommandQueueType type) const;

        /** Get the native API handle.
            For D3D12 backend, this is a ID3D12Device*. For GFX backend, this is ComPtr<gfx::IDevice>.
        */
        const DeviceHandle& getApiHandle() { return mApiHandle; }

        /** Get a D3D12 handle for user code that wants to call D3D12 directly.
            \return A valid ID3D12Device* value for all backend that are using D3D12, otherwise nullptr.
        */
        const D3D12DeviceHandle getD3D12Handle();

        /** Present the back-buffer to the window
        */
        void present();

        /** Flushes pipeline, releases resources, and blocks until completion
        */
        void flushAndSync();

        /** Check if vertical sync is enabled
        */
        bool isVsyncEnabled() const { return mDesc.enableVsync; }

        /** Resize the swap-chain
            \return A new FBO object
        */
        Fbo::SharedPtr resizeSwapChain(uint32_t width, uint32_t height);

        /** Get the desc
        */
        const Desc& getDesc() const { return mDesc; }

        /** Create a new query heap.
            \param[in] type Type of queries.
            \param[in] count Number of queries.
            \return New query heap.
        */
        std::weak_ptr<QueryHeap> createQueryHeap(QueryHeap::Type type, uint32_t count);

#if FALCOR_HAS_D3D12
        const D3D12DescriptorPool::SharedPtr& getD3D12CpuDescriptorPool() const { return mpD3D12CpuDescPool; }
        const D3D12DescriptorPool::SharedPtr& getD3D12GpuDescriptorPool() const { return mpD3D12GpuDescPool; }
#endif // FALCOR_HAS_D3D12

        DeviceApiData* getApiData() const { return mpApiData.get(); }
        const GpuMemoryHeap::SharedPtr& getUploadHeap() const { return mpUploadHeap; }
        void releaseResource(ApiObjectHandle pResource);
#ifdef FALCOR_GFX
        void releaseResource(ISlangUnknown* pResource) { releaseResource(ApiObjectHandle(pResource)); }
#endif
        double getGpuTimestampFrequency() const { return mGpuTimestampFrequency; } // ms/tick

        /** Check if features are supported by the device
        */
        bool isFeatureSupported(SupportedFeatures flags) const;

        /** Check if a shader model is supported by the device
        */
        bool isShaderModelSupported(ShaderModel shaderModel) const;

        /** Return the highest supported shader model by the device
        */
        ShaderModel getSupportedShaderModel() const { return mSupportedShaderModel; }

        /** Return the current index of the back buffer being rendered to.
        */
        uint32_t getCurrentBackBufferIndex() const { return mCurrentBackBufferIndex; }

#ifdef FALCOR_GFX
        gfx::ITransientResourceHeap* getCurrentTransientResourceHeap();
#endif

    private:
        struct ResourceRelease
        {
            size_t frameID;
            ApiObjectHandle pApiObject;
        };
        std::queue<ResourceRelease> mDeferredReleases;

        uint32_t mCurrentBackBufferIndex;
        Fbo::SharedPtr mpSwapChainFbos[kSwapChainBuffersCount];

        Device(Window::SharedPtr pWindow, const Desc& desc);
        bool init();
        void executeDeferredReleases();
        void releaseFboData();
        bool updateDefaultFBO(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat);

        Desc mDesc;
        ApiHandle mApiHandle;
        GpuMemoryHeap::SharedPtr mpUploadHeap;
#if FALCOR_HAS_D3D12
        D3D12DescriptorPool::SharedPtr mpD3D12CpuDescPool;
        D3D12DescriptorPool::SharedPtr mpD3D12GpuDescPool;
#endif
        bool mIsWindowOccluded = false;
        GpuFence::SharedPtr mpFrameFence;

        Window::SharedPtr mpWindow;
        std::unique_ptr<DeviceApiData> mpApiData;
        RenderContext::SharedPtr mpRenderContext;
        size_t mFrameID = 0;
        std::list<QueryHeap::SharedPtr> mTimestampQueryHeaps;
        double mGpuTimestampFrequency;
        std::vector<CommandQueueHandle> mCmdQueues[kQueueTypeCount];

        SupportedFeatures mSupportedFeatures = SupportedFeatures::None;
        ShaderModel mSupportedShaderModel = ShaderModel::Unknown;

        // API specific functions
        bool getApiFboData(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat, ResourceHandle apiHandles[kSwapChainBuffersCount], uint32_t& currentBackBufferIndex);
        void destroyApiObjects();
        void apiPresent();
        bool apiInit();
        bool createSwapChain(ResourceFormat colorFormat);
        void apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat);
        void toggleFullScreen(bool fullscreen);
    };

    inline constexpr uint32_t getMaxViewportCount() { return 8; }

    FALCOR_API extern Device::SharedPtr gpDevice;

    FALCOR_ENUM_CLASS_OPERATORS(Device::SupportedFeatures);
}
