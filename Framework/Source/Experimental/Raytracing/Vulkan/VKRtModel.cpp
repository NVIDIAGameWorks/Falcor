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
#include "Experimental/Raytracing/RtModel.h"
#include "API/Device.h"
#include "API/RenderContext.h"
#include "API/LowLevel/LowLevelContextData.h"
#include "API/Vulkan/FalcorVK.h"
#include "API/VAO.h"

namespace Falcor
{
    VkIndexType getVkIndexType(ResourceFormat format);
    VkDeviceMemory allocateDeviceMemory(Device::MemoryType memType, uint32_t memoryTypeBits, size_t size);
    
    VkMemoryRequirements getAccelerationStructureMemoryRequirements(VkAccelerationStructureNV handle, VkAccelerationStructureMemoryRequirementsTypeNV type)
    {
        VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
        memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
        memoryRequirementsInfo.pNext = nullptr;
        memoryRequirementsInfo.accelerationStructure = handle;
        memoryRequirementsInfo.type = type;

        VkMemoryRequirements2 memoryRequirements;
        vkGetAccelerationStructureMemoryRequirementsNV(gpDevice->getApiHandle(), &memoryRequirementsInfo, &memoryRequirements);

        return memoryRequirements.memoryRequirements;
    };

    void RtModel::buildAccelerationStructure()
    {
        RenderContext* pContext = gpDevice->getRenderContext();

        VkBuildAccelerationStructureFlagsNV vkRayFlags = getVKRayBuildFlags(mBuildFlags);

        // Create an AS for each mesh-group
        for (auto& blasData : mBottomLevelData)
        {
            std::vector<VkGeometryNV> geometries(blasData.meshCount);
            for (size_t meshIndex = blasData.meshBaseIndex; meshIndex < blasData.meshBaseIndex + blasData.meshCount; meshIndex++)
            {
                assert(meshIndex < mMeshes.size());
                const Mesh* pMesh = getMesh((uint32_t)meshIndex).get();

                VkGeometryNV& geometry = geometries[meshIndex - blasData.meshBaseIndex];
                geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
                geometry.pNext = nullptr;
                geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
                geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
                geometry.geometry.triangles.pNext = nullptr;
                geometry.geometry.triangles.transformData = VK_NULL_HANDLE;
                geometry.geometry.triangles.transformOffset = 0;
                geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV;
                geometry.geometry.aabbs.aabbData = NULL;
                geometry.flags = 0;

                // Get the position VB
                const Vao* pVao = getMeshVao(pMesh).get();
                const auto& elemDesc = pVao->getElementIndexByLocation(VERTEX_POSITION_LOC);
                const auto& pVbLayout = pVao->getVertexLayout()->getBufferLayout(elemDesc.vbIndex);

                const Buffer* pVB = pVao->getVertexBuffer(elemDesc.vbIndex).get();
                geometry.geometry.triangles.vertexData = pVB->getApiHandle();
                geometry.geometry.triangles.vertexOffset = pVB->getGpuAddressOffset() + pVbLayout->getElementOffset(elemDesc.elementIndex);
                geometry.geometry.triangles.vertexCount = pMesh->getVertexCount();
                geometry.geometry.triangles.vertexStride = pVbLayout->getStride();
                geometry.geometry.triangles.vertexFormat = getVkFormat(pVbLayout->getElementFormat(elemDesc.elementIndex));

                // Get the IB
                const Buffer* pIB = pVao->getIndexBuffer().get();
                geometry.geometry.triangles.indexData = pIB->getApiHandle();
                geometry.geometry.triangles.indexOffset = pIB->getGpuAddressOffset();
                geometry.geometry.triangles.indexCount = pMesh->getIndexCount();
                geometry.geometry.triangles.indexType = getVkIndexType(pVao->getIndexBufferFormat());
                
                // If this is an opaque mesh, set the opaque flag
                if (pMesh->getMaterial()->getAlphaMode() == AlphaModeOpaque)
                {
                    geometry.flags |= VK_GEOMETRY_OPAQUE_BIT_NV;
                }
            }

            // Create the acceleration structure
            VkAccelerationStructureCreateInfoNV accelerationStructureInfo;
            accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
            accelerationStructureInfo.pNext = nullptr;
            accelerationStructureInfo.compactedSize = 0;
            accelerationStructureInfo.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
            accelerationStructureInfo.info.pNext = NULL;
            accelerationStructureInfo.info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
            accelerationStructureInfo.info.flags = 0; // vkRayFlags
            accelerationStructureInfo.info.instanceCount = 0;
            accelerationStructureInfo.info.geometryCount = (uint32_t)geometries.size();
            accelerationStructureInfo.info.pGeometries = geometries.data();

            VkAccelerationStructureNV as;
            vk_call(vkCreateAccelerationStructureNV(gpDevice->getApiHandle(), &accelerationStructureInfo, nullptr, &as));
            blasData.pBlas = AccelerationStructureHandle::create(as);

            // Bind acceleration structure memory
            VkMemoryRequirements reqs = getAccelerationStructureMemoryRequirements(blasData.pBlas, VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV);
            VkDeviceMemory memory = allocateDeviceMemory(Device::MemoryType::Default, reqs.memoryTypeBits, reqs.size);

            VkBindAccelerationStructureMemoryInfoNV bindInfo;
            bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
            bindInfo.pNext = nullptr;
            bindInfo.accelerationStructure = blasData.pBlas;
            bindInfo.memory = memory;
            bindInfo.memoryOffset = 0;
            bindInfo.deviceIndexCount = 0;
            bindInfo.pDeviceIndices = nullptr;

            vk_call(vkBindAccelerationStructureMemoryNV(gpDevice->getApiHandle(), 1, &bindInfo));

            // Create scratch buffer
            VkDeviceSize scratchBufferSize = getAccelerationStructureMemoryRequirements(blasData.pBlas, VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV).size;
            Buffer::SharedPtr pScratchBuffer = Buffer::create(scratchBufferSize, Resource::BindFlags::RayTracing, Buffer::CpuAccess::None);

            // Build acceleration structure
            VkAccelerationStructureInfoNV asInfo;
            asInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
            asInfo.pNext = NULL;
            asInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
            asInfo.flags = vkRayFlags;
            asInfo.instanceCount = 0;
            asInfo.geometryCount = (uint32_t)geometries.size();
            asInfo.pGeometries = &geometries[0];

            vkCmdBuildAccelerationStructureNV(pContext->getLowLevelData()->getCommandList(), &asInfo, VK_NULL_HANDLE, 0, VK_FALSE, blasData.pBlas, VK_NULL_HANDLE, pScratchBuffer->getApiHandle(), 0);

            pContext->accelerationStructureBarrier();
        }
    }
};
