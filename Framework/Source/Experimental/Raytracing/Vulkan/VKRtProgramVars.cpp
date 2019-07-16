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
#include "Experimental/Raytracing/RtProgramVars.h"
#include "Experimental/Raytracing/RtStateObject.h"
#include "VKRtProgramVarsHelper.h"

namespace Falcor
{
    uint32_t RtProgramVars::getProgramIdentifierSize()
    {
        static uint32_t sShaderGroupHandleSize = ~0;
        if (sShaderGroupHandleSize == ~0)
        {
            VkPhysicalDeviceRayTracingPropertiesNV rayTracingProperties = {};
            rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;
            rayTracingProperties.pNext = nullptr;
            rayTracingProperties.maxRecursionDepth = 0;
            rayTracingProperties.shaderGroupHandleSize = 0;

            VkPhysicalDeviceProperties2 props;
            props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props.pNext = &rayTracingProperties;
            props.properties = { };
            vkGetPhysicalDeviceProperties2(gpDevice->getApiHandle(), &props);

            sShaderGroupHandleSize = rayTracingProperties.shaderGroupHandleSize;
        }

        return sShaderGroupHandleSize;
    }

    bool RtProgramVars::applyRtProgramVars(uint8_t* pRecord, const RtProgramVersion* pProgVersion, const RtStateObject* pRtso, ProgramVars* pVars, RtVarsContext* pContext)
    {
        memcpy(reinterpret_cast<void*>(pRecord), pRtso->getShaderGroupHandle(pProgVersion), mProgramIdentifierSize);
        pRecord += mProgramIdentifierSize;

        // Sets the write head for the proxy command list to copy constants to
        pContext->getRtVarsCmdList()->setRootParams(pProgVersion->getLocalRootSignature(), pRecord);

        // Should only have one parameter block and one root set
        assert(pVars->getParameterBlockCount() == 1);
        ParameterBlock* pBlock = const_cast<ParameterBlock*>(pVars->getParameterBlock(0).get());

        // Grab the shader record constant buffer, We don't care about the descriptor set
        auto pCB = pBlock->getConstantBuffer(ParameterBlock::BindLocation(0, 0), 0);
        if (pCB)
        {
            // Since Vulkan ShaderRecord only supports embedded constants, copy the constants into the memory of the SBT
            // instead of writing the descriptor table handle like D3D12 does.
            pContext->getRtVarsCmdList()->setRootConstants(pCB->getData().data(), static_cast<uint32_t>(pCB->getSize()));
        }

        return true;
    }

    void RtVarsContext::apiInit()
    {
        mpList = RtVarsCmdList::create();
    }
}
