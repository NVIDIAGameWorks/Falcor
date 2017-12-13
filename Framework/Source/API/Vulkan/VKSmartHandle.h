/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

namespace Falcor
{
    class VkBaseApiHandle : public std::enable_shared_from_this<VkBaseApiHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<VkBaseApiHandle>;
        virtual ~VkBaseApiHandle() = default;
    };

    template<typename ApiHandle>
    class VkHandle : public VkBaseApiHandle, public inherit_shared_from_this<VkBaseApiHandle, VkHandle<ApiHandle>>
    {
    public:
        class SharedPtr : public std::shared_ptr<VkHandle<ApiHandle>>
        {
        public:
            SharedPtr() = default;
            SharedPtr(VkHandle<ApiHandle>* pHandle) : std::shared_ptr<VkHandle<ApiHandle>>(pHandle) {}
            static SharedPtr create(ApiHandle handle) { return SharedPtr(new VkHandle(handle)); }
            operator ApiHandle() const { return get()->mApiHandle; }
        private:
            VkHandle<ApiHandle>* get() const { return std::shared_ptr< VkHandle<ApiHandle>>::get(); }
        };

        ~VkHandle()
        {
#ifdef _WIN32
            static_assert(false, "VkHandle missing destructor specialization"); 
#endif
        }
    private:
        friend class SharedPtr;
        VkHandle(const ApiHandle& apiHandle) : mApiHandle(apiHandle) {}
        ApiHandle mApiHandle;
    };

    class VkRootSignature : public VkBaseApiHandle, public inherit_shared_from_this<VkBaseApiHandle, VkRootSignature>
    {
    public:
        class SharedPtr : public std::shared_ptr<VkRootSignature>
        {
        public:
            SharedPtr() = default;
            SharedPtr(VkRootSignature* pHandle) : std::shared_ptr<VkRootSignature>(pHandle) {}
            static SharedPtr create(VkPipelineLayout layout, const std::vector<VkDescriptorSetLayout>& sets) { return SharedPtr(new VkRootSignature(layout, sets)); }
            operator VkPipelineLayout() const { return get()->mApiHandle; }
        private:
            VkRootSignature* get() const { return std::shared_ptr<VkRootSignature>::get(); }
        };

        ~VkRootSignature();
    private:
        friend class SharedPtr;
        VkRootSignature(VkPipelineLayout layout, const std::vector<VkDescriptorSetLayout>& sets) : mApiHandle(layout), mSets(sets) {}
        VkPipelineLayout mApiHandle;
        std::vector<VkDescriptorSetLayout> mSets;
    };
    
    class VkDeviceData : public VkBaseApiHandle, public inherit_shared_from_this<VkBaseApiHandle, VkDeviceData>
    {
    public:
        class SharedPtr : public std::shared_ptr<VkDeviceData>
        {
        public:
            SharedPtr() = default;
            SharedPtr(VkDeviceData* pData) : std::shared_ptr<VkDeviceData>(pData) {}
            static SharedPtr create(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface)
            {
                return SharedPtr(new VkDeviceData(instance, physicalDevice, device, surface));
            }

            operator VkInstance() const { return get()->mInstance; }
            operator VkPhysicalDevice() const { return get()->mPhysicalDevice; }
            operator VkDevice() const { return get()->mLogicalDevice; }
            operator VkSurfaceKHR() const { return get()->mSurface; }
        private:
            VkDeviceData* get() const { return std::shared_ptr<VkDeviceData>::get(); }
        };

        ~VkDeviceData();
    private:
        friend SharedPtr;
        VkDeviceData(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface) :
            mInstance(instance), mPhysicalDevice(physicalDevice), mLogicalDevice(device), mSurface(surface) {}
        VkInstance          mInstance;
        VkPhysicalDevice    mPhysicalDevice;
        VkDevice            mLogicalDevice;
        VkSurfaceKHR        mSurface;

    };

    enum class VkResourceType
    {
        None,
        Image,
        Buffer
    };

    template<typename ImageType, typename BufferType>
    class VkResource : public VkBaseApiHandle, public inherit_shared_from_this<VkBaseApiHandle, VkResource<ImageType, BufferType>>
    {
    public:
        class SharedPtr : public std::shared_ptr<VkResource<ImageType, BufferType>>
        {
        public:
            SharedPtr() = default;
            SharedPtr(VkResource<ImageType, BufferType>* pRes) : std::shared_ptr<VkResource<ImageType, BufferType>>(pRes) {}
            static SharedPtr create(ImageType image, VkDeviceMemory mem) { return SharedPtr(new VkResource(image, mem)); }
            static SharedPtr create(BufferType buffer, VkDeviceMemory mem) { return SharedPtr(new VkResource(buffer, mem)); }

            VkResourceType getType() const { return get()->mType; }
            operator ImageType() const { assert(get()->mType == VkResourceType::Image); return get()->mImage; }
            operator BufferType() const { assert(get()->mType == VkResourceType::Buffer); return get()->mBuffer; }
            operator VkDeviceMemory() const { return get()->mDeviceMem; }
        private:
            VkResource<ImageType, BufferType>* get() const { return std::shared_ptr<VkResource<ImageType, BufferType>>::get(); }
        };

        ~VkResource()
        {
#ifdef _WIN32
            static_assert(false, "VkResource missing destructor specialization"); 
#endif
        }
    private:
        friend SharedPtr;
        VkResource(ImageType image, VkDeviceMemory mem) : mType(VkResourceType::Image), mImage(image), mDeviceMem(mem) {}
        VkResource(BufferType buffer, VkDeviceMemory mem) : mType(VkResourceType::Buffer), mBuffer(buffer), mDeviceMem(mem) {}

        VkResourceType mType = VkResourceType::None;
        ImageType mImage = VK_NULL_HANDLE;
        BufferType mBuffer = VK_NULL_HANDLE;
        VkDeviceMemory mDeviceMem = VK_NULL_HANDLE;
    };

    class VkFbo : public VkBaseApiHandle
    {
    public:
        class SharedPtr : public std::shared_ptr<VkFbo>
        {
        public:
            SharedPtr() = default;
            SharedPtr(VkFbo* pFbo) : std::shared_ptr<VkFbo>(pFbo) {}
            static SharedPtr create(VkRenderPass renderPass, VkFramebuffer fbo) { return SharedPtr(new VkFbo(renderPass, fbo)); }

            operator VkFramebuffer() const { return get()->mVkFbo; }
            operator VkRenderPass() const { return get()->mVkRenderPass; }
        private:
            VkFbo* get() const { return std::shared_ptr<VkFbo>::get(); }
        };

        ~VkFbo();
        operator VkRenderPass() const { return mVkRenderPass; }
        operator VkFramebuffer() const { return mVkFbo; }
    private:
        friend SharedPtr;
        VkFbo(VkRenderPass renderPass, VkFramebuffer fbo) : mVkRenderPass(renderPass), mVkFbo(fbo) {}
        VkRenderPass mVkRenderPass = VK_NULL_HANDLE;
        VkFramebuffer mVkFbo = VK_NULL_HANDLE;
    };

    // Destructors
    template<> VkHandle<VkSwapchainKHR>::~VkHandle();
    template<> VkHandle<VkCommandPool>::~VkHandle();
    template<> VkHandle<VkSemaphore>::~VkHandle();
    template<> VkHandle<VkSampler>::~VkHandle();
    template<> VkHandle<VkDescriptorSetLayout>::~VkHandle();
    template<> VkHandle<VkPipeline>::~VkHandle();
    template<> VkHandle<VkShaderModule>::~VkHandle();
    template<> VkHandle<VkPipelineLayout>::~VkHandle();
    template<> VkHandle<VkDescriptorPool>::~VkHandle();
    template<> VkHandle<VkQueryPool>::~VkHandle();

    template<> VkResource<VkImage, VkBuffer>::~VkResource();
    template<> VkResource<VkImageView, VkBufferView>::~VkResource();
}
