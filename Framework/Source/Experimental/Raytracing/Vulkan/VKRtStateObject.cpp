/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Experimental/Raytracing/RtStateObject.h"
#include "Experimental/Raytracing/RtProgramVars.h"
#include "API/Device.h"
#include "API/LowLevel/RootSignature.h"

namespace Falcor
{
    RtStateObject::SharedPtr RtStateObject::create(const Desc& desc)
    {
        SharedPtr pState = SharedPtr(new RtStateObject(desc));

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        std::vector<VkRayTracingShaderGroupCreateInfoNV> shaderGroups;

        std::map<std::wstring, uint32_t> exportNameToGroupID;

        // Loop over the programs
        for (const auto& pProg : pState->getProgramList())
        {
            if (pProg->getType() == RtProgramVersion::Type::Hit)
            {
                const RtShader* pIntersection = pProg->getShader(ShaderType::Intersection).get();
                const RtShader* pAhs = pProg->getShader(ShaderType::AnyHit).get();
                const RtShader* pChs = pProg->getShader(ShaderType::ClosestHit).get();

                VkRayTracingShaderGroupCreateInfoNV group;
                group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
                group.pNext = nullptr;
                group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
                group.generalShader = VK_SHADER_UNUSED_NV;
                group.closestHitShader = VK_SHADER_UNUSED_NV;
                group.anyHitShader = VK_SHADER_UNUSED_NV;
                group.intersectionShader = VK_SHADER_UNUSED_NV;

                if (pIntersection)
                {
                    shaderStages.push_back(pIntersection->getShaderStage(VK_SHADER_STAGE_INTERSECTION_BIT_NV));
                    group.intersectionShader = (uint32_t)shaderStages.size() - 1;
                }

                if (pAhs)
                {
                    shaderStages.push_back(pAhs->getShaderStage(VK_SHADER_STAGE_ANY_HIT_BIT_NV));
                    group.anyHitShader = (uint32_t)shaderStages.size() - 1;
                }

                if (pChs)
                {
                    shaderStages.push_back(pChs->getShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV));
                    group.closestHitShader = (uint32_t)shaderStages.size() - 1;
                }

                shaderGroups.push_back(group);
                exportNameToGroupID[pProg->getExportName()] = (uint32_t)shaderGroups.size() - 1;
            }
            else
            {
                const RtShader* pShader = pProg->getShader(pProg->getType() == RtProgramVersion::Type::Miss ? ShaderType::Miss : ShaderType::RayGeneration).get();
                shaderStages.push_back(pShader->getShaderStage(pProg->getType() == RtProgramVersion::Type::Miss ? VK_SHADER_STAGE_MISS_BIT_NV : VK_SHADER_STAGE_RAYGEN_BIT_NV));

                VkRayTracingShaderGroupCreateInfoNV group;
                group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
                group.pNext = nullptr;
                group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
                group.generalShader = (uint32_t)shaderStages.size() - 1;
                group.closestHitShader = VK_SHADER_UNUSED_NV;
                group.anyHitShader = VK_SHADER_UNUSED_NV;
                group.intersectionShader = VK_SHADER_UNUSED_NV;

                shaderGroups.push_back(group);
                exportNameToGroupID[pProg->getExportName()] = (uint32_t)shaderGroups.size() - 1;
            }
        }

        RootSignature* pRootSig = desc.mpGlobalRootSignature ? desc.mpGlobalRootSignature.get() : RootSignature::getEmpty().get();

        VkRayTracingPipelineCreateInfoNV rayPipelineInfo;
        rayPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
        rayPipelineInfo.pNext = nullptr;
        rayPipelineInfo.flags = 0;
        rayPipelineInfo.stageCount = (uint32_t)shaderStages.size();
        rayPipelineInfo.pStages = shaderStages.data();
        rayPipelineInfo.groupCount = (uint32_t)shaderGroups.size();
        rayPipelineInfo.pGroups = shaderGroups.data();
        rayPipelineInfo.maxRecursionDepth = desc.mMaxTraceRecursionDepth;
        rayPipelineInfo.layout = pRootSig->getApiHandle();
        rayPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        rayPipelineInfo.basePipelineIndex = 0;

        VkPipeline pipeline;
        vk_call(vkCreateRayTracingPipelinesNV(gpDevice->getApiHandle(), nullptr, 1, &rayPipelineInfo, nullptr, &pipeline));
        pState->mApiHandle = ApiHandle::create(pipeline);

        // Cache shader group handles
        for (auto& pair : exportNameToGroupID)
        {
            const uint32_t handleSize = RtProgramVars::getProgramIdentifierSize();
            std::unique_ptr<uint8_t[]> shaderGroupHandle = std::make_unique<uint8_t[]>(handleSize);
            vk_call(vkGetRayTracingShaderGroupHandlesNV(gpDevice->getApiHandle(), pState->getApiHandle(), pair.second, 1, handleSize, reinterpret_cast<void*>(shaderGroupHandle.get())));
            pState->mShaderGroupHandleCache[pair.first] = std::move(shaderGroupHandle);
        }

        return pState;
    }

    const void* RtStateObject::getShaderGroupHandle(const RtProgramVersion* pProgVersion) const
    {
        const auto it = mShaderGroupHandleCache.find(pProgVersion->getExportName());
        if (it == mShaderGroupHandleCache.end())
        {
            logError("Could not find cached shader group handle");
            return nullptr;
        }

        return it->second.get();
    }
}
