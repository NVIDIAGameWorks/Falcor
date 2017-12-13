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
#pragma once
#include "API/Window.h"
#include "API/Texture.h"
#include "API/FBO.h"
#include "API/RenderContext.h"
#include "API/LowLevel/DescriptorPool.h"
#include "API/LowLevel/ResourceAllocator.h"
#include "API/QueryHeap.h"

namespace Falcor
{
#ifdef _DEBUG
#define DEFAULT_ENABLE_DEBUG_LAYER true
#else
#define DEFAULT_ENABLE_DEBUG_LAYER false
#endif

    struct DeviceApiData;

    class Device
    {
    public:
        using SharedPtr = std::shared_ptr<Device>;
        using SharedConstPtr = std::shared_ptr<const Device>;
        using ApiHandle = DeviceHandle;
        static const uint32_t kQueueTypeCount = (uint32_t)LowLevelContextData::CommandQueueType::Count;

        /** Device configuration
        */
        struct Desc
        {
            ResourceFormat colorFormat = ResourceFormat::BGRA8UnormSrgb;    ///< The color buffer format
            ResourceFormat depthFormat = ResourceFormat::D32Float;          ///< The depth buffer format
            int apiMajorVersion = DEFAULT_API_MAJOR_VERSION;                ///< Requested API major version. Context creation fails if this version is not supported.
            int apiMinorVersion = DEFAULT_API_MINOR_VERSION;                ///< Requested API minor version. Context creation fails if this version is not supported.
            std::vector<std::string> requiredExtensions;                    ///< Extensions required by the sample
            bool enableVsync = false;                                       ///< Controls vertical-sync
            bool enableDebugLayer = DEFAULT_ENABLE_DEBUG_LAYER;             ///< Enable the debug layer. The default for release build is false, for debug build it's true.
            bool enableVR = false;                                          ///< Create a device matching OpenVR requirements

            static_assert((uint32_t)LowLevelContextData::CommandQueueType::Direct == 2, "Default initialization of cmdQueues assumes that Direct queue index is 0");
            uint32_t cmdQueues[kQueueTypeCount] = { 0, 0, 1 };  ///< Command queues to create. If not direct-queues are created, mpRenderContext will not be initialized

#ifdef FALCOR_D3D
            /** The following callback allows the user to create its own device (for example, create a WARP device or choose a specific GPU in a multi-GPU machine)
            */
            using CreateDeviceFunc = std::function<DeviceHandle(IDXGIAdapter* pAdapter, D3D_FEATURE_LEVEL featureLevel)>;
            CreateDeviceFunc createDeviceFunc;
#endif
        };

        /** Create a new device.
            \param[in] pWindow a previously-created window object
            \param[in] desc Device configuration descriptor.
            \return nullptr if the function failed, otherwise a new device object
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

        /** Check if the device support an extension
        */
        bool isExtensionSupported(const std::string & name) const;

        /** Get the FBO object associated with the swap-chain.
            This can change each frame, depending on the API used
        */
        Fbo::SharedPtr getSwapChainFbo() const;

        /** Get the default render-context.
            The default render-context is managed completely by the device. The user should just queue commands into it, the device will take care of allocation, submission and synchronization
        */
        const RenderContext::SharedPtr& getRenderContext() const { return mpRenderContext; }

        /** Get the command queue handle
        */
        CommandQueueHandle getCommandQueueHandle(LowLevelContextData::CommandQueueType type, uint32_t index) const;

        /** Get the API queue type
        */
        ApiCommandQueueType getApiCommandQueueType(LowLevelContextData::CommandQueueType type) const;

        /** Get the native API handle
        */
        DeviceHandle getApiHandle() { return mApiHandle; }

        /** Present the back-buffer to the window
        */
        void present();

        /** Flushes pipeline, releases resources, and blocks until completion
        */
        void flushAndSync();

        /** Check if vertical sync is enabled
        */
        bool isVsyncEnabled() const { return mVsyncOn; }

        /** Resize the swap-chain
            \return A new FBO object
        */
        Fbo::SharedPtr resizeSwapChain(uint32_t width, uint32_t height);

        const DescriptorPool::SharedPtr& getCpuDescriptorPool() const { return mpCpuDescPool; }
        const DescriptorPool::SharedPtr& getGpuDescriptorPool() const { return mpGpuDescPool; }
        const ResourceAllocator::SharedPtr& getResourceAllocator() const { return mpResourceAllocator; }
        const QueryHeap::SharedPtr& getTimestampQueryHeap() const { return mTimestampQueryHeap; }
        void releaseResource(ApiObjectHandle pResource);
        double getGpuTimestampFrequency() const { return mGpuTimestampFrequency; } // ms/tick
        bool isRgb32FloatSupported() const { return mRgb32FloatSupported; }

#ifdef FALCOR_VK
        enum class MemoryType
        {
            Default,
            Upload,
            Readback,
            Count
        };
        uint32_t getVkMemoryType(MemoryType falcorType, uint32_t memoryTypeBits) const;
        const VkPhysicalDeviceLimits& getPhysicalDeviceLimits() const;
        uint32_t  getDeviceVendorID() const;
#endif
    private:
        struct ResourceRelease
        {
            size_t frameID;
            ApiObjectHandle pApiObject;
        };
        std::queue<ResourceRelease> mDeferredReleases;

        uint32_t mCurrentBackBufferIndex;
        std::vector<Fbo::SharedPtr> mpSwapChainFbos;
        uint32_t mSwapChainBufferCount = kDefaultSwapChainBuffers;

        Device(Window::SharedPtr pWindow) : mpWindow(pWindow) {}
        bool init(const Desc& desc);
        void executeDeferredReleases();
        void releaseFboData();
        bool updateDefaultFBO(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat);

        ApiHandle mApiHandle;
        ResourceAllocator::SharedPtr mpResourceAllocator;
        DescriptorPool::SharedPtr mpCpuDescPool;
        DescriptorPool::SharedPtr mpGpuDescPool;
        bool mIsWindowOccluded = false;
        GpuFence::SharedPtr mpFrameFence;

        Window::SharedPtr mpWindow;
        DeviceApiData* mpApiData;
        RenderContext::SharedPtr mpRenderContext;
        bool mVsyncOn;
        size_t mFrameID = 0;
        QueryHeap::SharedPtr mTimestampQueryHeap;
        double mGpuTimestampFrequency;
        bool mRgb32FloatSupported = true;
        std::vector<CommandQueueHandle> mCmdQueues[kQueueTypeCount];

        // API specific functions
        bool getApiFboData(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat, std::vector<ResourceHandle>& apiHandles, uint32_t& currentBackBufferIndex);
        void destroyApiObjects();
        void apiPresent();
        bool apiInit(const Desc& desc);
        bool createSwapChain(ResourceFormat colorFormat);
        void apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat);
    };

    extern Device::SharedPtr gpDevice;
}
