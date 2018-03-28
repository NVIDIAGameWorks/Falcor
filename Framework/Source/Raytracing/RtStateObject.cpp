/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#ifdef FALCOR_DXR
#include "Framework.h"
#include "DXR.h"
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
            }
            else
            {
                const RtShader* pShader = pProg->getShader(pProg->getType() == RtProgramVersion::Type::Miss ? ShaderType::Miss : ShaderType::RayGeneration).get();
                rtsoHelper.addProgramDesc(pShader->getD3DBlob(), pProg->getExportName());
            }

            // Root signature
            const std::wstring& exportName = pProg->getExportName();
            rtsoHelper.addLocalRootSignature(&exportName, 1, pProg->getRootSignature()->getApiHandle().GetInterfacePtr());
            // Payload size
            rtsoHelper.addShaderConfig(&exportName, 1, pProg->getMaxPayloadSize(), pProg->getMaxAttributesSize());
        }

        // Add an empty global root-signature
        rtsoHelper.addGlobalRootSignature(RootSignature::getEmpty()->getApiHandle().GetInterfacePtr());

        // Create the state
        D3D12_STATE_OBJECT_DESC objectDesc = rtsoHelper.getDesc();
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();
        d3d_call(pRtDevice->CreateStateObject(&objectDesc, IID_PPV_ARGS(&pState->mApiHandle)));

        return pState;
    }
}
#endif