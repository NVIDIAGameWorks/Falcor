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
#pragma once
#include "Framework.h"
#include "RtProgramVars.h"
#include "RtState.h"

namespace Falcor
{
    void raytrace(RenderContext* pContext, RtProgramVars::SharedPtr pVars, RtState::SharedPtr pState, uint32_t width, uint32_t height)
    {
        pContext->resourceBarrier(pVars->getSBT().get(), Resource::State::ShaderResource);

        Buffer* pSBT = pVars->getSBT().get();
        uint32_t sbtRecordSize = pVars->getRecordSize();
        D3D12_GPU_VIRTUAL_ADDRESS sbtStartAddress = pSBT->getGpuAddress();

        D3D12_DISPATCH_RAYS_DESC raytraceDesc = {};
        raytraceDesc.Width = width;
        raytraceDesc.Height = height;

        // RayGen is the first entry in the SBT
        raytraceDesc.RayGenerationShaderRecord.StartAddress = sbtStartAddress + pVars->getRayGenSbtRecordIndex() * sbtRecordSize;
        raytraceDesc.RayGenerationShaderRecord.SizeInBytes = sbtRecordSize;

        // Miss is the second entry in the SBT
        raytraceDesc.MissShaderTable.StartAddress = sbtStartAddress + pVars->getFirstMissSbtRecordIndex() * sbtRecordSize;
        raytraceDesc.MissShaderTable.StrideInBytes = sbtRecordSize;
        raytraceDesc.MissShaderTable.SizeInBytes = sbtRecordSize * pVars->getMissProgramsCount();

        raytraceDesc.HitGroupTable.StartAddress = sbtStartAddress + pVars->getFirstHitSbtRecordIndex() * sbtRecordSize;
        raytraceDesc.HitGroupTable.StrideInBytes = sbtRecordSize;
        raytraceDesc.HitGroupTable.SizeInBytes = pVars->getSBT()->getSize() - (pVars->getFirstHitSbtRecordIndex() * sbtRecordSize);

        // Currently, we need to set an empty root-signature. Some wizardry is required to make sure we restore the state
        const auto& pComputeVars = pContext->getComputeVars();
        pContext->setComputeVars(nullptr);
        ID3D12GraphicsCommandListPtr pCmdList = pContext->getLowLevelData()->getCommandList();
        pCmdList->SetComputeRootSignature(RootSignature::getEmpty()->getApiHandle().GetInterfacePtr());

        // Dispatch
        ID3D12CommandListRaytracingPrototypePtr pRtCmdList = pCmdList;
        pRtCmdList->DispatchRays(pState->getRtso()->getApiHandle().GetInterfacePtr(), &raytraceDesc);

        // Restore the vars
        pContext->setComputeVars(pComputeVars);
    }
}
