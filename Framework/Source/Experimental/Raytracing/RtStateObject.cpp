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
#include "RtStateObject.h"
#include "RtStateObjectHelper.h"
#include "API/Device.h"
#include "Utils/StringUtils.h"
#include "API/LowLevel/RootSignature.h"

namespace Falcor
{
    bool RtStateObject::Desc::operator==(const RtStateObject::Desc& other) const
    {
        bool b = true;
        b = b && (mMaxTraceRecursionDepth == other.mMaxTraceRecursionDepth);
        b = b && (mProgList.size() == other.mProgList.size());

        if (b)
        {
            for (size_t i = 0; i < mProgList.size(); i++)
            {
                b = b && (mProgList[i] == other.mProgList[i]);
            }
        }
        return b;
    }
    
    RtStateObject::SharedPtr RtStateObject::create(const Desc& desc)
    {
        SharedPtr pState = SharedPtr(new RtStateObject(desc));

        RtStateObjectHelper rtsoHelper;
        // Pipeline config
        rtsoHelper.addPipelineConfig(desc.mMaxTraceRecursionDepth);

        // Loop over the programs
        for (const auto& pProg : pState->getProgramList())
        {
            if (pProg->getType() == RtProgramVersion::Type::Hit)
            {
                const RtShader* pIntersection = pProg->getShader(ShaderType::Intersection).get();
                const RtShader* pAhs = pProg->getShader(ShaderType::AnyHit).get();
                const RtShader* pChs = pProg->getShader(ShaderType::ClosestHit).get();

                ID3DBlobPtr pIntersectionBlob = pIntersection ? pIntersection->getD3DBlob() : nullptr;
                ID3DBlobPtr pAhsBlob = pAhs ? pAhs->getD3DBlob() : nullptr;
                ID3DBlobPtr pChsBlob = pChs ? pChs->getD3DBlob() : nullptr;

                const std::wstring& exportName = pProg->getExportName();
                const std::wstring& intersectionExport = pIntersection ? string_2_wstring(pIntersection->getEntryPoint()) : L"";
                const std::wstring& ahsExport = pAhs ? string_2_wstring(pAhs->getEntryPoint()) : L"";
                const std::wstring& chsExport = pChs ? string_2_wstring(pChs->getEntryPoint()) : L"";

                rtsoHelper.addHitProgramDesc(pAhsBlob, ahsExport, pChsBlob, chsExport, pIntersectionBlob, intersectionExport, exportName);

                if (intersectionExport.size())
                {
                    rtsoHelper.addLocalRootSignature(&intersectionExport, 1, pProg->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                    rtsoHelper.addShaderConfig(&intersectionExport, 1, pProg->getMaxPayloadSize(), pProg->getMaxAttributesSize());
                }

                if (ahsExport.size())
                {
                    rtsoHelper.addLocalRootSignature(&ahsExport, 1, pProg->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                    rtsoHelper.addShaderConfig(&ahsExport, 1, pProg->getMaxPayloadSize(), pProg->getMaxAttributesSize());
                }

                if (chsExport.size())
                {
                    rtsoHelper.addLocalRootSignature(&chsExport, 1, pProg->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                    rtsoHelper.addShaderConfig(&chsExport, 1, pProg->getMaxPayloadSize(), pProg->getMaxAttributesSize());
                }
            }
            else
            {
                const RtShader* pShader = pProg->getShader(pProg->getType() == RtProgramVersion::Type::Miss ? ShaderType::Miss : ShaderType::RayGeneration).get();
                rtsoHelper.addProgramDesc(pShader->getD3DBlob(), pProg->getExportName());

                // Root signature
                const std::wstring& exportName = pProg->getExportName();
                rtsoHelper.addLocalRootSignature(&exportName, 1, pProg->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                // Payload size
                rtsoHelper.addShaderConfig(&exportName, 1, pProg->getMaxPayloadSize(), pProg->getMaxAttributesSize());
            }
        }

        // Add an empty global root-signature
        RootSignature* pRootSig = desc.mpGlobalRootSignature ? desc.mpGlobalRootSignature.get() : RootSignature::getEmpty().get();
        rtsoHelper.addGlobalRootSignature(pRootSig->getApiHandle());

        // Create the state
        D3D12_STATE_OBJECT_DESC objectDesc = rtsoHelper.getDesc();
        GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        d3d_call(pDevice5->CreateStateObject(&objectDesc, IID_PPV_ARGS(&pState->mApiHandle)));

        return pState;
    }
}
