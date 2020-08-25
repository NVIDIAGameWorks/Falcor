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
#include "Core/API/GraphicsStateObject.h"
#include "D3D12NvApiExDesc.h"
#include "Core/API/Device.h"
#include "D3D12State.h"

namespace Falcor
{
#if _ENABLE_NVAPI
    bool getNvApiGraphicsPsoDesc(const GraphicsStateObject::Desc& desc, std::vector<NvApiPsoExDesc>& nvApiPsoExDescs)
    {
        auto ret = NvAPI_Initialize();

        if (ret != NVAPI_OK)
        {
            logError("Failed to initialize NvApi");
            return false;
        }

        if (auto optRegisterIndex = findNvApiShaderRegister(desc.getProgramKernels()))
        {
            auto registerIndex = *optRegisterIndex;
            nvApiPsoExDescs.push_back(NvApiPsoExDesc());
            createNvApiUavSlotExDesc(nvApiPsoExDescs.back(), registerIndex);
        }
        return true;
    }

    GraphicsStateObject::ApiHandle getNvApiGraphicsPsoHandle(const std::vector<NvApiPsoExDesc>& nvDescVec, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc)
    {
        const NVAPI_D3D12_PSO_EXTENSION_DESC* ppPSOExtensionsDesc[5];

        for (uint32_t ex = 0; ex < nvDescVec.size(); ex++)
        {
            switch (nvDescVec[ex].psoExtension)
            {
            case NV_PSO_VERTEX_SHADER_EXTENSION:                ppPSOExtensionsDesc[ex] = &nvDescVec[ex].mVsExDesc; break;
            case NV_PSO_HULL_SHADER_EXTENSION:                  ppPSOExtensionsDesc[ex] = &nvDescVec[ex].mHsExDesc; break;
            case NV_PSO_DOMAIN_SHADER_EXTENSION:                ppPSOExtensionsDesc[ex] = &nvDescVec[ex].mDsExDesc; break;
            case NV_PSO_GEOMETRY_SHADER_EXTENSION:              ppPSOExtensionsDesc[ex] = &nvDescVec[ex].mGsExDesc; break;
            case NV_PSO_SET_SHADER_EXTNENSION_SLOT_AND_SPACE:   ppPSOExtensionsDesc[ex] = &nvDescVec[ex].mExtSlotDesc; break;
            default: should_not_get_here();
            }
        }
        GraphicsStateObject::ApiHandle apiHandle;
        auto ret = NvAPI_D3D12_CreateGraphicsPipelineState(gpDevice->getApiHandle(), &desc, (NvU32)nvDescVec.size(), ppPSOExtensionsDesc, &apiHandle);

        if (ret != NVAPI_OK || apiHandle == nullptr)
        {
            logError("Failed to create a graphics pipeline state object with NVAPI extensions");
            return nullptr;
        }

        return apiHandle;
    }

    bool getIsNvApiGraphicsPsoRequired(const GraphicsStateObject::Desc& desc)
    {
        return findNvApiShaderRegister(desc.getProgramKernels()).has_value();
    }
#else
    bool getNvApiGraphicsPsoDesc(const GraphicsStateObject::Desc& desc, std::vector<NvApiPsoExDesc>& nvApiPsoExDescs) { should_not_get_here(); return false; }
    GraphicsStateObject::ApiHandle getNvApiGraphicsPsoHandle(const std::vector<NvApiPsoExDesc>& psoDesc, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc) { should_not_get_here(); return nullptr; }
    bool getIsNvApiGraphicsPsoRequired(const GraphicsStateObject::Desc& desc) { return false; }
#endif

    void GraphicsStateObject::apiInit()
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC d3dDesc;
        InputLayoutDesc inputDesc;
        initD3D12GraphicsStateDesc(mDesc, d3dDesc, inputDesc);

        if (getIsNvApiGraphicsPsoRequired(mDesc))
        {
            std::vector<NvApiPsoExDesc> nvApiDesc;
            bool ret = getNvApiGraphicsPsoDesc(mDesc, nvApiDesc);
            if (!ret) throw std::exception("Failed to create graphics PSO desc with NVAPI extensions");

            mApiHandle = getNvApiGraphicsPsoHandle(nvApiDesc, d3dDesc);
            if (mApiHandle == nullptr) throw std::exception("Failed to create graphics PSO with NVAPI extensions");
        }
        else
        {
            d3d_call(gpDevice->getApiHandle()->CreateGraphicsPipelineState(&d3dDesc, IID_PPV_ARGS(&mApiHandle)));
        }
    }
}
