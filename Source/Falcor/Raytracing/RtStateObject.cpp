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
#include "RtStateObject.h"
#include "RtStateObjectHelper.h"
#include "Utils/StringUtils.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12NvApiExDesc.h"
#include "ShaderTable.h"

namespace Falcor
{
    bool RtStateObject::Desc::operator==(const RtStateObject::Desc& other) const
    {
        bool b = true;
        b = b && (mMaxTraceRecursionDepth == other.mMaxTraceRecursionDepth);
        b = b && (mpKernels == other.mpKernels);
        return b;
    }

    RtStateObject::SharedPtr RtStateObject::create(const Desc& desc)
    {
        SharedPtr pState = SharedPtr(new RtStateObject(desc));

        RtStateObjectHelper rtsoHelper;
        // Pipeline config
        rtsoHelper.addPipelineConfig(desc.mMaxTraceRecursionDepth);

        auto pKernels = pState->getKernels();

#if _ENABLE_NVAPI
        // Enable NVAPI extension if required
        auto nvapiRegisterIndex = findNvApiShaderRegister(pKernels);
        if (nvapiRegisterIndex)
        {
            if (NvAPI_Initialize() != NVAPI_OK) throw std::exception("Failed to initialize NvApi");
            if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(gpDevice->getApiHandle(), *nvapiRegisterIndex, 0) != NVAPI_OK) throw std::exception("Failed to set NvApi extension");
        }
#endif

        // Loop over the programs
        for (const auto& pBaseEntryPointGroup : pKernels->getUniqueEntryPointGroups() )
        {
            assert(dynamic_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get()));
            auto pEntryPointGroup = static_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get());
            switch( pBaseEntryPointGroup->getType() )
            {
            case EntryPointGroupKernels::Type::RtHitGroup:
                {
                    const Shader* pIntersection = pEntryPointGroup->getShader(ShaderType::Intersection);
                    const Shader* pAhs = pEntryPointGroup->getShader(ShaderType::AnyHit);
                    const Shader* pChs = pEntryPointGroup->getShader(ShaderType::ClosestHit);

                    ID3DBlobPtr pIntersectionBlob = pIntersection ? pIntersection->getD3DBlob() : nullptr;
                    ID3DBlobPtr pAhsBlob = pAhs ? pAhs->getD3DBlob() : nullptr;
                    ID3DBlobPtr pChsBlob = pChs ? pChs->getD3DBlob() : nullptr;

                    const std::wstring& exportName = string_2_wstring(pEntryPointGroup->getExportName());
                    const std::wstring& intersectionExport = pIntersection ? string_2_wstring(pIntersection->getEntryPoint()) : L"";
                    const std::wstring& ahsExport = pAhs ? string_2_wstring(pAhs->getEntryPoint()) : L"";
                    const std::wstring& chsExport = pChs ? string_2_wstring(pChs->getEntryPoint()) : L"";

                    rtsoHelper.addHitProgramDesc(pAhsBlob, ahsExport, pChsBlob, chsExport, pIntersectionBlob, intersectionExport, exportName);

                    if (intersectionExport.size())
                    {
                        rtsoHelper.addLocalRootSignature(&intersectionExport, 1, pEntryPointGroup->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                        rtsoHelper.addShaderConfig(&intersectionExport, 1, pEntryPointGroup->getMaxPayloadSize(), pEntryPointGroup->getMaxAttributesSize());
                    }

                    if (ahsExport.size())
                    {
                        rtsoHelper.addLocalRootSignature(&ahsExport, 1, pEntryPointGroup->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                        rtsoHelper.addShaderConfig(&ahsExport, 1, pEntryPointGroup->getMaxPayloadSize(), pEntryPointGroup->getMaxAttributesSize());
                    }

                    if (chsExport.size())
                    {
                        rtsoHelper.addLocalRootSignature(&chsExport, 1, pEntryPointGroup->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                        rtsoHelper.addShaderConfig(&chsExport, 1, pEntryPointGroup->getMaxPayloadSize(), pEntryPointGroup->getMaxAttributesSize());
                    }
                }
                break;

            default:
                {
                    const std::wstring& exportName = string_2_wstring(pEntryPointGroup->getExportName());


                    const Shader* pShader = pEntryPointGroup->getShaderByIndex(0);
                    rtsoHelper.addProgramDesc(pShader->getD3DBlob(), exportName);

                    // Root signature
                    rtsoHelper.addLocalRootSignature(&exportName, 1, pEntryPointGroup->getLocalRootSignature()->getApiHandle().GetInterfacePtr());
                    // Payload size
                    rtsoHelper.addShaderConfig(&exportName, 1, pEntryPointGroup->getMaxPayloadSize(), pEntryPointGroup->getMaxAttributesSize());
                }
                break;
            }
        }

        // Add an empty global root-signature
        RootSignature* pRootSig = desc.mpGlobalRootSignature ? desc.mpGlobalRootSignature.get() : RootSignature::getEmpty().get();
        rtsoHelper.addGlobalRootSignature(pRootSig->getApiHandle());

        // Create the state
        D3D12_STATE_OBJECT_DESC objectDesc = rtsoHelper.getDesc();
        GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        d3d_call(pDevice5->CreateStateObject(&objectDesc, IID_PPV_ARGS(&pState->mApiHandle)));

        MAKE_SMART_COM_PTR(ID3D12StateObjectProperties);
        ID3D12StateObjectPropertiesPtr pRtsoProps = pState->getApiHandle();

        for( const auto& pBaseEntryPointGroup : pKernels->getUniqueEntryPointGroups() )
        {
            assert(dynamic_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get()));
            auto pEntryPointGroup = static_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get());
            const std::wstring& exportName = string_2_wstring(pEntryPointGroup->getExportName());

            void const* pShaderIdentifier = pRtsoProps->GetShaderIdentifier(exportName.c_str());
            pState->mShaderIdentifiers.push_back(pShaderIdentifier);
        }

#if _ENABLE_NVAPI
        if (nvapiRegisterIndex)
        {
            if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(gpDevice->getApiHandle(), 0xFFFFFFFF, 0) != NVAPI_OK) throw std::exception("Failed to unset NvApi extension");
        }
#endif

        return pState;
    }
}
