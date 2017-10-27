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
#include "Framework.h"
#include "API/LowLevel/RootSignature.h"
#include "API/Device.h"
#include <set>

namespace Falcor
{
    VkDescriptorType falcorToVkDescType(DescriptorPool::Type type);

    VkShaderStageFlags getShaderVisibility(ShaderVisibility visibility)
    {
        VkShaderStageFlags flags = 0;

        if ((visibility & ShaderVisibility::Vertex) != ShaderVisibility::None)
        {
            flags |= VK_SHADER_STAGE_VERTEX_BIT;
        }
        if ((visibility & ShaderVisibility::Pixel) != ShaderVisibility::None)
        {
            flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        if ((visibility & ShaderVisibility::Geometry) != ShaderVisibility::None)
        {
            flags |= VK_SHADER_STAGE_GEOMETRY_BIT;
        }
        if ((visibility & ShaderVisibility::Domain) != ShaderVisibility::None)
        {
            flags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;;
        }
        if ((visibility & ShaderVisibility::Hull) != ShaderVisibility::None)
        {
            flags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        }
        if ((visibility & ShaderVisibility::Compute) != ShaderVisibility::None)
        {
            flags |= VK_SHADER_STAGE_COMPUTE_BIT;
        }
        return flags;
    }
    
    VkDescriptorSetLayout createDescriptorSetLayout(const DescriptorSet::Layout& layout)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(layout.getRangeCount());

        uint32_t space;
        for (uint32_t r = 0; r < layout.getRangeCount(); r++)
        {
            VkDescriptorSetLayoutBinding& b = bindings[r];
            const auto& range = layout.getRange(r);
            assert(r == 0 || space == range.regSpace);
            space = range.regSpace;
            b.binding = range.baseRegIndex;
            b.descriptorCount = range.descCount;
            b.descriptorType = falcorToVkDescType(range.type);
            b.pImmutableSamplers = nullptr;
            b.stageFlags = getShaderVisibility(layout.getVisibility());
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = (uint32_t)bindings.size();
        layoutInfo.pBindings = bindings.data();
        VkDescriptorSetLayout vkHandle;
        vk_call(vkCreateDescriptorSetLayout(gpDevice->getApiHandle(), &layoutInfo, nullptr, &vkHandle));
        return vkHandle;
    }

    bool RootSignature::apiInit()
    {
        // Find the max set index
        uint32_t maxIndex = 0;
        for (const auto& set : mDesc.mSets)
        {
            maxIndex = max(set.getRange(0).regSpace, maxIndex);
        }

        static VkDescriptorSetLayout emptyLayout = createDescriptorSetLayout({});   // #VKTODO This gets deleted multiple times on exit
        std::vector<VkDescriptorSetLayout> vkSetLayouts(maxIndex + 1, emptyLayout);
        for (const auto& set : mDesc.mSets)
        {
            vkSetLayouts[set.getRange(0).regSpace] = createDescriptorSetLayout(set); //createDescriptorSetLayout() verifies that all ranges use the same register space
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.pSetLayouts = vkSetLayouts.data();
        pipelineLayoutInfo.setLayoutCount = (uint32_t)vkSetLayouts.size();
        VkPipelineLayout layout;
        vk_call(vkCreatePipelineLayout(gpDevice->getApiHandle(), &pipelineLayoutInfo, nullptr, &layout));
        mApiHandle = ApiHandle::create(layout, vkSetLayouts);
        return true;
    }

    struct Range
    {
        uint32_t baseIndex;
        uint32_t count;

        bool operator<(const Range& other) const { return baseIndex < other.baseIndex; }
    };
    using SetRangeMap = std::map<RootSignature::DescType, std::vector<Range>>;
    using SetMap = std::map<uint32_t, SetRangeMap>;

    struct ResData
    {
        RootSignature::DescType type;
        uint32_t regSpace;
        uint32_t regIndex;
        uint32_t descOffst;
        uint32_t count;
    };

    static ResData getResData(const ProgramReflection::Resource& resource)
    {
        ResData data;
        switch (resource.type)
        {
        case ProgramReflection::Resource::ResourceType::Sampler:
                data.type = RootSignature::DescType::Sampler;
                break;
            case ProgramReflection::Resource::ResourceType::StructuredBuffer:
                data.type = (resource.shaderAccess == ProgramReflection::ShaderAccess::Read) ? RootSignature::DescType::StructuredBufferSrv : RootSignature::DescType::StructuredBufferUav;
                break;
            case ProgramReflection::Resource::ResourceType::TypedBuffer:
                data.type = (resource.shaderAccess == ProgramReflection::ShaderAccess::Read) ? RootSignature::DescType::TypedBufferSrv : RootSignature::DescType::TypedBufferUav;
                break;
            case ProgramReflection::Resource::ResourceType::Texture:
                data.type = (resource.shaderAccess == ProgramReflection::ShaderAccess::Read) ? RootSignature::DescType::TextureSrv : RootSignature::DescType::TextureUav;
                break;
            default:
                should_not_get_here();
        }


        data.count = resource.arraySize ? resource.arraySize : 1;
        data.regIndex = resource.regIndex;
        data.regSpace = resource.regSpace;
        data.descOffst = resource.descOffset;
        return data;
    }

    void insertResData(SetMap& map, const ResData& data)
    {
        if (map.find(data.regSpace) == map.end())
        {
            map[data.regSpace] = {};
        }

        SetRangeMap& rangeMap = map[data.regSpace];
        if (rangeMap.find(data.type) == rangeMap.end())
        {
            rangeMap[data.type] = {};
        }

        // Check if we already have a range with the same base index
        for (auto& r : rangeMap[data.type])
        {
            if (r.baseIndex == data.regIndex)
            {
                r.count = max(r.count, data.count + data.descOffst);
                return;
            }
        }

        // New Range
        rangeMap[data.type].push_back({ data.regIndex, data.count + data.descOffst });
    }

    ProgramReflection::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type);

    static void insertBuffers(const ProgramReflection* pReflector, SetMap& setMap, ProgramReflection::BufferReflection::Type bufferType, RootSignature::DescType descType)
    {
        const auto& bufMap = pReflector->getBufferMap(bufferType);
        for (const auto& buf : bufMap)
        {
            const ProgramReflection::BufferReflection* pBuffer = buf.second.get();
            if (pBuffer->getShaderAccess() == getRequiredShaderAccess(descType))
            {
                ResData resData;
                resData.count = 1;
                resData.descOffst = 0;
                resData.regIndex = pBuffer->getRegisterIndex();
                resData.regSpace = pBuffer->getRegisterSpace();
                resData.type = descType;
                insertResData(setMap, resData);
            }
        }
    }

    RootSignature::SharedPtr RootSignature::create(const ProgramReflection* pReflector)
    {
        // We'd like to create an optimized signature (minimize the number of ranges per set). First, go over all of the resources and find packed ranges
        SetMap setMap;

        const auto& resMap = pReflector->getResourceMap();
        for (const auto& res : resMap)
        {
            const ProgramReflection::Resource& resource = res.second;
            ResData resData = getResData(resource);
            insertResData(setMap, resData);
        }

        insertBuffers(pReflector, setMap, ProgramReflection::BufferReflection::Type::Constant, RootSignature::DescType::Cbv);
        insertBuffers(pReflector, setMap, ProgramReflection::BufferReflection::Type::Structured, RootSignature::DescType::StructuredBufferSrv);
        insertBuffers(pReflector, setMap, ProgramReflection::BufferReflection::Type::Structured, RootSignature::DescType::StructuredBufferUav);

        std::map<uint32_t, DescriptorSetLayout> setLayouts;
        for (auto& s : setMap)
        {
            for (auto& r : s.second)
            {
                for (const auto& range : r.second)
                {
                    // #VKTODO set the correct shader flags in the layout
                    if (setLayouts.find(s.first) == setLayouts.end()) setLayouts[s.first] = {};
                    setLayouts[s.first].addRange(r.first, range.baseIndex, range.count, s.first);
                }
            }
        }

        Desc d;
        for (const auto& s : setLayouts)
        {
            d.addDescriptorSet(s.second);
        }
        return create(d);
    }

    void RootSignature::bindForGraphics(CopyContext* pCtx) {}
    void RootSignature::bindForCompute(CopyContext* pCtx) {}
}
