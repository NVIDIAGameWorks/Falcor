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

    void RootSignature::bindForGraphics(CopyContext* pCtx) {}
    void RootSignature::bindForCompute(CopyContext* pCtx) {}
}
