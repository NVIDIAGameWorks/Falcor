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
#include "API/Device.h"

#define VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, funcName) \
    { \
        funcName = reinterpret_cast<PFN_##funcName>(vkGetDeviceProcAddr(device, ""#funcName)); \
        if (funcName == nullptr) \
        { \
            const std::string name = #funcName; \
            logErrorAndExit(std::string("Can't get device function address: ") + name); \
        } \
    }

namespace Falcor
{
    PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructureNV = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructureNV = VK_NULL_HANDLE;
    PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV = VK_NULL_HANDLE;
    PFN_vkCmdCopyAccelerationStructureNV vkCmdCopyAccelerationStructureNV = VK_NULL_HANDLE;
    PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemoryNV = VK_NULL_HANDLE;
    PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructureNV = VK_NULL_HANDLE;
    PFN_vkCmdTraceRaysNV vkCmdTraceRaysNV = VK_NULL_HANDLE;
    PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandlesNV = VK_NULL_HANDLE;
    PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelinesNV = VK_NULL_HANDLE;
    PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandleNV = VK_NULL_HANDLE;

    void loadRaytracingEntrypoints()
    {
        VkDevice device = gpDevice->getApiHandle();
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkCreateAccelerationStructureNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkDestroyAccelerationStructureNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkGetAccelerationStructureMemoryRequirementsNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkCmdCopyAccelerationStructureNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkBindAccelerationStructureMemoryNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkCmdBuildAccelerationStructureNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkCmdTraceRaysNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkGetRayTracingShaderGroupHandlesNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkCreateRayTracingPipelinesNV);
        VK_RESOLVE_DEVICE_FUNCTION_ADDRESS(device, vkGetAccelerationStructureHandleNV);
    }
}
