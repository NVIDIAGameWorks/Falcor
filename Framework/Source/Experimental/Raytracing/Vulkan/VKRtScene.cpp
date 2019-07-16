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
#include "Experimental/Raytracing/RtScene.h"
#include "API/Device.h"

namespace Falcor
{
    VkDeviceMemory allocateDeviceMemory(Device::MemoryType memType, uint32_t memoryTypeBits, size_t size);
    VkMemoryRequirements getAccelerationStructureMemoryRequirements(VkAccelerationStructureNV handle, VkAccelerationStructureMemoryRequirementsTypeNV type);

    // TODO: Cache TLAS per hitProgCount, as some render pipelines need multiple TLAS:es with different #hit progs in same frame, currently that trigger rebuild every frame. See issue #365.
    void RtScene::createTlas(uint32_t hitProgCount)
    {
        if (mTlasHitProgCount == hitProgCount) return;
        mTlasHitProgCount = hitProgCount;

        // Early out if hit program count is zero or if scene is empty.
        if (hitProgCount == 0 || getModelCount() == 0)
        {
            mModelInstanceData.clear();
            mpTopLevelAS = nullptr;
            mGeometryCount = 0;
            mInstanceCount = 0;
            mRefit = false;
            return;
        }

        // todo: move this somewhere fair.
        mRtFlags |= RtBuildFlags::AllowUpdate;

        VkBuildAccelerationStructureFlagsNV vkRayFlags = getVKRayBuildFlags(mRtFlags);

        RenderContext* pContext = gpDevice->getRenderContext();
        std::vector<VkGeometryInstance> instanceDesc = createInstanceDesc(this, hitProgCount);

        // todo: improve this check - make sure things have not changed much and update was enabled last time
        bool isRefitPossible = mRefit && mpTopLevelAS && (mInstanceCount == (uint32_t)instanceDesc.size());

        mInstanceCount = (uint32_t)instanceDesc.size();
        
        if (!isRefitPossible)
        {
            // Create the top-level acceleration buffers
            VkAccelerationStructureCreateInfoNV accelerationStructureInfo;
            accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
            accelerationStructureInfo.pNext = nullptr;
            accelerationStructureInfo.compactedSize = 0;
            accelerationStructureInfo.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
            accelerationStructureInfo.info.pNext = NULL;
            accelerationStructureInfo.info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
            accelerationStructureInfo.info.flags = 0;
            accelerationStructureInfo.info.instanceCount = mInstanceCount;
            accelerationStructureInfo.info.geometryCount = 0;
            accelerationStructureInfo.info.pGeometries = nullptr;

            VkAccelerationStructureNV as;
            vk_call(vkCreateAccelerationStructureNV(gpDevice->getApiHandle(), &accelerationStructureInfo, nullptr, &as));
            mpTopLevelAS = AccelerationStructureHandle::create(as);

            // Bind acceleration structure memory
            VkMemoryRequirements reqs = getAccelerationStructureMemoryRequirements(mpTopLevelAS, VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV);
            VkDeviceMemory memory = allocateDeviceMemory(Device::MemoryType::Default, reqs.memoryTypeBits, reqs.size);

            VkBindAccelerationStructureMemoryInfoNV bindInfo;
            bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
            bindInfo.pNext = nullptr;
            bindInfo.accelerationStructure = mpTopLevelAS;
            bindInfo.memory = memory;
            bindInfo.memoryOffset = 0;
            bindInfo.deviceIndexCount = 0;
            bindInfo.pDeviceIndices = nullptr;

            vk_call(vkBindAccelerationStructureMemoryNV(gpDevice->getApiHandle(), 1, &bindInfo));
        }
        else
        {
            pContext->accelerationStructureBarrier();
        }

        // Create instance buffer
        uint32_t instanceBufferSize = mInstanceCount * (uint32_t)sizeof(VkGeometryInstance);
        Buffer::SharedPtr pInstanceData = Buffer::create(instanceBufferSize, Resource::BindFlags::RayTracing, Buffer::CpuAccess::None, instanceDesc.data());

        // Create scratch buffer
        VkAccelerationStructureMemoryRequirementsTypeNV type = isRefitPossible ? VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV : VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
        VkDeviceSize scratchBufferSize = getAccelerationStructureMemoryRequirements(mpTopLevelAS, type).size;
        Buffer::SharedPtr pScratchBuffer = Buffer::create(scratchBufferSize, Resource::BindFlags::RayTracing, Buffer::CpuAccess::None);

        assert((mInstanceCount != 0) && pInstanceData->getApiHandle() && mpTopLevelAS && pScratchBuffer->getApiHandle());

        // Build acceleration structure
        VkAccelerationStructureInfoNV asInfo;
        asInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
        asInfo.pNext = nullptr;
        asInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
        asInfo.flags = vkRayFlags;
        asInfo.instanceCount = mInstanceCount;
        asInfo.geometryCount = 0;
        asInfo.pGeometries = nullptr;

        if (!isRefitPossible)
        {
            vkCmdBuildAccelerationStructureNV(pContext->getLowLevelData()->getCommandList(), &asInfo, pInstanceData->getApiHandle(), 0, VK_FALSE, mpTopLevelAS, VK_NULL_HANDLE, pScratchBuffer->getApiHandle(), 0);
        }
        else
        {
            vkCmdBuildAccelerationStructureNV(pContext->getLowLevelData()->getCommandList(), &asInfo, pInstanceData->getApiHandle(), 0, VK_TRUE, mpTopLevelAS, mpTopLevelAS, pScratchBuffer->getApiHandle(), 0);
        }

        pContext->accelerationStructureBarrier();

        mRefit = false;
    }
}
