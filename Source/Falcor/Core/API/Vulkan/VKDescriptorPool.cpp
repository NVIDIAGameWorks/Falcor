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
#include "API/LowLevel/DescriptorPool.h"
#include "API/Device.h"
#include "API/Vulkan/LowLevel/VKDescriptorData.h"

namespace Falcor
{
    VkDescriptorType falcorToVkDescType(DescriptorPool::Type type)
    {
        switch (type)
        {
        case DescriptorPool::Type::TextureSrv:
            return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case DescriptorPool::Type::TextureUav:
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case DescriptorPool::Type::RawBufferSrv:
        case DescriptorPool::Type::TypedBufferSrv:
            return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        case DescriptorPool::Type::RawBufferUav:
        case DescriptorPool::Type::TypedBufferUav:
            return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        case DescriptorPool::Type::Cbv:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorPool::Type::StructuredBufferSrv:
        case DescriptorPool::Type::StructuredBufferUav:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case DescriptorPool::Type::Dsv:
        case DescriptorPool::Type::Rtv:
            return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case DescriptorPool::Type::Sampler:
            return VK_DESCRIPTOR_TYPE_SAMPLER;
        default:
            should_not_get_here();
            return VK_DESCRIPTOR_TYPE_MAX_ENUM;
        }
    }

    void DescriptorPool::apiInit()
    {
        mpApiData = std::make_shared<DescriptorPool::ApiData>();
        uint32_t totalDescCount = 0;
        VkDescriptorPoolSize poolSizeForType[kTypeCount];

        uint32_t usedSlots = 0;
        for (uint32_t i = 0; i < kTypeCount; i++)
        {
            if(mDesc.mDescCount[i])
            {
                poolSizeForType[usedSlots].type = falcorToVkDescType((DescriptorPool::Type)i);
                poolSizeForType[usedSlots].descriptorCount = mDesc.mDescCount[i];
                totalDescCount += mDesc.mDescCount[usedSlots];
                usedSlots++;
            }
        }

        VkDescriptorPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = totalDescCount;
        info.poolSizeCount = usedSlots;
        info.pPoolSizes = poolSizeForType;
        info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        VkDescriptorPool pool;
        if (VK_FAILED(vkCreateDescriptorPool(gpDevice->getApiHandle(), &info, nullptr, &pool)))
        {
            throw std::exception("Error creating descriptor pool!");
        }
        mpApiData->descriptorPool = ApiHandle::create(pool);
    }

    const DescriptorPool::ApiHandle& DescriptorPool::getApiHandle(uint32_t heapIndex) const
    {
        return mpApiData->descriptorPool;
    }
}
