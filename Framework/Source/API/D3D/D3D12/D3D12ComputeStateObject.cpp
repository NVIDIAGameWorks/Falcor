/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "D3D12NvApiExDesc.h"
#include "API/ComputeStateObject.h"
#include "API/Device.h"

namespace Falcor
{
#if _ENABLE_NVAPI
    void getNvApiComputePsoDesc(const ComputeStateObject::Desc& desc, std::vector<NvApiPsoExDesc>& nvApiPsoExDescs)
    {
        auto ret = NvAPI_Initialize();

        if (ret != NVAPI_OK)
        {
            logError("Failed to initialize NvApi", true);
        }

        auto uav = desc.getProgramVersion()->getReflector()->getBufferBinding("g_NvidiaExt");
        if (uav.baseRegIndex != ProgramReflection::kInvalidLocation)
        {
            nvApiPsoExDescs.push_back(NvApiPsoExDesc());
            createNvApiUavSlotExDesc(nvApiPsoExDescs.back(), uav.baseRegIndex);
        }
    }

    ComputeStateObject::ApiHandle getNvApiComputePsoHandle(const std::vector<NvApiPsoExDesc>& nvDescVec, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc)
    {
        const NVAPI_D3D12_PSO_EXTENSION_DESC* ppPSOExtensionsDesc[1];

        for (uint32_t ex = 0; ex < nvDescVec.size(); ex++)
        {
            switch (nvDescVec[ex].psoExtension)
            {
            case NV_PSO_SET_SHADER_EXTNENSION_SLOT_AND_SPACE:   ppPSOExtensionsDesc[ex] = &nvDescVec[ex].mExtSlotDesc; break;
            default: should_not_get_here();
            }
        }
        ComputeStateObject::ApiHandle apiHandle;
        auto ret = NvAPI_D3D12_CreateComputePipelineState(gpDevice->getApiHandle(), &desc, (NvU32)nvDescVec.size(), ppPSOExtensionsDesc, &apiHandle);

        if (ret != NVAPI_OK || apiHandle == nullptr)
        {
            logError("Failed to create a compute pipeline state object with NVAPI extensions", true);
            return nullptr;
        }

        return apiHandle;
    }

    bool getIsNvApiComputePsoRequired(const ComputeStateObject::Desc& desc)
    {
        auto uav = desc.getProgramVersion()->getReflector()->getBufferBinding("g_NvidiaExt");
        return uav.baseRegIndex != ProgramReflection::kInvalidLocation;
    }

#else
    void getNvApiComputePsoDesc(const ComputeStateObject::Desc& desc, std::vector<NvApiPsoExDesc>& nvApiPsoExDescs) { should_not_get_here(); }
    ComputeStateObject::ApiHandle getNvApiComputePsoHandle(const std::vector<NvApiPsoExDesc>& psoDesc, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc) { should_not_get_here(); return nullptr; }
    bool getIsNvApiComputePsoRequired(const ComputeStateObject::Desc& desc) { return false; }
#endif

    bool ComputeStateObject::apiInit()
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        assert(mDesc.mpProgram);
        desc.CS = mDesc.mpProgram->getShader(ShaderType::Compute)->getApiHandle();
        
        desc.pRootSignature = mDesc.mpRootSignature ? mDesc.mpRootSignature->getApiHandle() : nullptr;

        if (getIsNvApiComputePsoRequired(mDesc))
        {
            std::vector<NvApiPsoExDesc> nvApiDesc;
            getNvApiComputePsoDesc(mDesc, nvApiDesc);
            mApiHandle = getNvApiComputePsoHandle(nvApiDesc, desc);
        }
        else
        {
            d3d_call(gpDevice->getApiHandle()->CreateComputePipelineState(&desc, IID_PPV_ARGS(&mApiHandle)));
        }
        return true;
    }
}