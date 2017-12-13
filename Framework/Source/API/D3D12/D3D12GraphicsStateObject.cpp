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
#include "API/GraphicsStateObject.h"
#include "API/D3D12/D3D12State.h"
#include "API/FBO.h"
#include "API/Texture.h"
#include "API/Device.h"

namespace Falcor
{
#if _ENABLE_NVAPI
    void getNvApiGraphicsPsoDesc(const GraphicsStateObject::Desc& desc, std::vector<NvApiPsoExDesc>& nvApiPsoExDescs)
    {
        auto ret = NvAPI_Initialize();

        if (ret != NVAPI_OK)
        {
            logError("Failed to initialize NvApi", true);
        }

        if (desc.getSinglePassStereoEnabled())
        {
            nvApiPsoExDescs.push_back(NvApiPsoExDesc());
            createNvApiVsExDesc(nvApiPsoExDescs.back());
        }

        auto uav = desc.getProgramVersion()->getReflector()->getBufferBinding("g_NvidiaExt");
        if (uav.baseRegIndex != ProgramReflection::kInvalidLocation)
        {
            nvApiPsoExDescs.push_back(NvApiPsoExDesc());
            createNvApiUavSlotExDesc(nvApiPsoExDescs.back(), uav.baseRegIndex);
        }
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
            logError("Failed to create a graphics pipeline state object with NVAPI extensions", true);
            return nullptr;
        }

        return apiHandle;
    }

    bool getIsNvApiGraphicsPsoRequired(const GraphicsStateObject::Desc& desc)
    {
        auto uav = desc.getProgramVersion()->getReflector()->getBufferBinding("g_NvidiaExt");
        return uav.baseRegIndex != ProgramReflection::kInvalidLocation || desc.getSinglePassStereoEnabled();
    }
#else
    void getNvApiGraphicsPsoDesc(const GraphicsStateObject::Desc& desc, std::vector<NvApiPsoExDesc>& nvApiPsoExDescs) { should_not_get_here(); }
    GraphicsStateObject::ApiHandle getNvApiGraphicsPsoHandle(const std::vector<NvApiPsoExDesc>& psoDesc, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc) { should_not_get_here(); return nullptr; }
    bool getIsNvApiGraphicsPsoRequired(const GraphicsStateObject::Desc& desc) { return false; }
#endif
    
    bool GraphicsStateObject::apiInit()
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        assert(mDesc.mpProgram);
#define get_shader_handle(_type) mDesc.mpProgram->getShader(_type) ? mDesc.mpProgram->getShader(_type)->getApiHandle() : D3D12_SHADER_BYTECODE{}

        desc.VS = get_shader_handle(ShaderType::Vertex);
        desc.PS = get_shader_handle(ShaderType::Pixel);
        desc.GS = get_shader_handle(ShaderType::Geometry);
        desc.HS = get_shader_handle(ShaderType::Hull);
        desc.DS = get_shader_handle(ShaderType::Domain);
#undef get_shader_handle

        initD3D12BlendDesc(mDesc.mpBlendState.get(), desc.BlendState);
        initD3D12RasterizerDesc(mDesc.mpRasterizerState.get(), desc.RasterizerState);
        initD3DDepthStencilDesc(mDesc.mpDepthStencilState.get(), desc.DepthStencilState);

        InputLayoutDesc layoutDesc;
        if(mDesc.mpLayout)
        {
            initD3D12VertexLayout(mDesc.mpLayout.get(), layoutDesc);
            desc.InputLayout.NumElements = (uint32_t)layoutDesc.elements.size();
            desc.InputLayout.pInputElementDescs = layoutDesc.elements.data();
        }
        desc.SampleMask = mDesc.mSampleMask;
        desc.pRootSignature = mDesc.mpRootSignature ? mDesc.mpRootSignature->getApiHandle() : nullptr;

        uint32_t numRtvs = 0;
        for (uint32_t rt = 0; rt < Fbo::getMaxColorTargetCount(); rt++)
        {
            desc.RTVFormats[rt] = getDxgiFormat(mDesc.mFboDesc.getColorTargetFormat(rt));
            if (desc.RTVFormats[rt] != DXGI_FORMAT_UNKNOWN)
            {
                numRtvs = rt + 1;
            }
        }
        desc.NumRenderTargets = numRtvs;
        desc.DSVFormat = getDxgiFormat(mDesc.mFboDesc.getDepthStencilFormat());
        desc.SampleDesc.Count = mDesc.mFboDesc.getSampleCount();

        desc.PrimitiveTopologyType = getD3DPrimitiveType(mDesc.mPrimType);

        if (getIsNvApiGraphicsPsoRequired(mDesc))
        {
            std::vector<NvApiPsoExDesc> nvApiDesc;
            getNvApiGraphicsPsoDesc(mDesc, nvApiDesc);
            mApiHandle = getNvApiGraphicsPsoHandle(nvApiDesc, desc);
        }
        else
        {
            d3d_call(gpDevice->getApiHandle()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&mApiHandle)));
        }
        return true;
    }
}