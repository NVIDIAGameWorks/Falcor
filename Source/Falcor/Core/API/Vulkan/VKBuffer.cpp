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
#include "API/Buffer.h"
#include "API/Device.h"
#include "API/LowLevel/ResourceAllocator.h"
#include "API/Vulkan/FalcorVK.h"
#include "API/Device.h"

namespace Falcor
{
    VkDeviceMemory allocateDeviceMemory(GpuMemoryHeap::Type memType, uint32_t memoryTypeBits, size_t size)
    {
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = size;
        allocInfo.memoryTypeIndex = gpDevice->getVkMemoryType(memType, memoryTypeBits);

        VkDeviceMemory deviceMem;
        vk_call(vkAllocateMemory(gpDevice->getApiHandle(), &allocInfo, nullptr, &deviceMem));
        return deviceMem;
    }

    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size)
    {
        void* pData;
        vk_call(vkMapMemory(gpDevice->getApiHandle(), apiHandle, 0, size, 0, &pData));
        return pData;
    }

    VkBufferUsageFlags getBufferUsageFlag(Buffer::BindFlags bindFlags)
    {       
        // Assume every buffer can be read from and written into
        VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto setBit = [&flags, &bindFlags](Buffer::BindFlags f, VkBufferUsageFlags vkBit) {if (is_set(bindFlags, f)) flags |= vkBit; };
        setBit(Buffer::BindFlags::Vertex,           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        setBit(Buffer::BindFlags::Index,            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        setBit(Buffer::BindFlags::UnorderedAccess,  VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        setBit(Buffer::BindFlags::ShaderResource,   VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
        setBit(Buffer::BindFlags::IndirectArg,      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        setBit(Buffer::BindFlags::Constant,         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        return flags;
    }

    size_t getBufferDataAlignment(const Buffer* pBuffer)
    {
        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(gpDevice->getApiHandle(), pBuffer->getApiHandle(), &reqs);
        return reqs.alignment;
    }

    Buffer::ApiHandle createBuffer(size_t size, Buffer::BindFlags bindFlags, GpuMemoryHeap::Type memType)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.flags = 0;
        bufferInfo.size = size;
        bufferInfo.usage = getBufferUsageFlag(bindFlags);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.queueFamilyIndexCount = 0;
        bufferInfo.pQueueFamilyIndices = nullptr;
        
        VkBuffer buffer;
        vk_call(vkCreateBuffer(gpDevice->getApiHandle(), &bufferInfo, nullptr, &buffer));

        // Get the required buffer size
        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(gpDevice->getApiHandle(), buffer, &reqs);

        VkDeviceMemory mem = allocateDeviceMemory(memType, reqs.memoryTypeBits, reqs.size);
        vk_call(vkBindBufferMemory(gpDevice->getApiHandle(), buffer, mem, 0));
        Buffer::ApiHandle apiHandle = Buffer::ApiHandle::create(buffer, mem);

        return apiHandle;
    }
    
    void Buffer::apiInit(bool hasInitData)
    {
        if (mCpuAccess == CpuAccess::Write)
        {
            mDynamicData = gpDevice->getUploadHeap()->allocate(mSize);
            mApiHandle = mDynamicData.pResourceHandle;
        }
        else
        {
            if (mCpuAccess == CpuAccess::Read && mBindFlags == BindFlags::None)
            {
                mApiHandle = createBuffer(mSize, mBindFlags, Device::MemoryType::Readback);
            }
            else
            {
                mApiHandle = createBuffer(mSize, mBindFlags, Device::MemoryType::Default);
            }
        }
    }

    uint64_t Buffer::getGpuAddress() const
    {
        UNSUPPORTED_IN_VULKAN(__FUNCTION__);
        return 0;
    }

    void Buffer::unmap()
    {
        if (mpStagingResource)
        {
            mpStagingResource->unmap();
            mpStagingResource = nullptr;
        }
        // We only unmap staging buffers
        else if (mDynamicData.pData == nullptr && mBindFlags == BindFlags::None)
        {
            assert(mCpuAccess == CpuAccess::Read);
            vkUnmapMemory(gpDevice->getApiHandle(), mApiHandle);
        }
    }

    uint64_t Buffer::makeResident(Buffer::GpuAccessFlags flags) const
    {
        UNSUPPORTED_IN_VULKAN(__FUNCTION__);
        return 0;
    }

    void Buffer::evict() const
    {
        UNSUPPORTED_IN_VULKAN(__FUNCTION__);
    }
}
