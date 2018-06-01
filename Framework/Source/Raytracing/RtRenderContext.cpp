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
#pragma once
#include "Framework.h"
#include "RtProgramVars.h"
#include "RtState.h"

namespace Falcor
{
    void RenderContext::raytrace(RtProgramVars::SharedPtr pVars, RtState::SharedPtr pState, uint32_t width, uint32_t height)
    {
        resourceBarrier(pVars->getShaderTable().get(), Resource::State::NonPixelShader);

        Buffer* pShaderTable = pVars->getShaderTable().get();
        uint32_t recordSize = pVars->getRecordSize();
        D3D12_GPU_VIRTUAL_ADDRESS startAddress = pShaderTable->getGpuAddress();

        D3D12_DISPATCH_RAYS_DESC raytraceDesc = {};
        raytraceDesc.Width = width;
        raytraceDesc.Height = height;

        // RayGen is the first entry in the shader-table
        raytraceDesc.RayGenerationShaderRecord.StartAddress = startAddress + pVars->getRayGenRecordIndex() * recordSize;
        raytraceDesc.RayGenerationShaderRecord.SizeInBytes = recordSize;

        // Miss is the second entry in the shader-table
        raytraceDesc.MissShaderTable.StartAddress = startAddress + pVars->getFirstMissRecordIndex() * recordSize;
        raytraceDesc.MissShaderTable.StrideInBytes = recordSize;
        raytraceDesc.MissShaderTable.SizeInBytes = recordSize * pVars->getMissProgramsCount();

        raytraceDesc.HitGroupTable.StartAddress = startAddress + pVars->getFirstHitRecordIndex() * recordSize;
        raytraceDesc.HitGroupTable.StrideInBytes = recordSize;
        raytraceDesc.HitGroupTable.SizeInBytes = pVars->getShaderTable()->getSize() - (pVars->getFirstHitRecordIndex() * recordSize);

        // Currently, we need to set an empty root-signature. Some wizardry is required to make sure we restore the state
        const auto& pComputeVars = getComputeVars();
        setComputeVars(nullptr);
        ID3D12GraphicsCommandListPtr pCmdList = getLowLevelData()->getCommandList();
        pCmdList->SetComputeRootSignature(pVars->getGlobalVars()->getRootSignature()->getApiHandle().GetInterfacePtr());

        // Dispatch
        ID3D12CommandListRaytracingPrototypePtr pRtCmdList = pCmdList;
        pRtCmdList->DispatchRays(pState->getRtso()->getApiHandle().GetInterfacePtr(), &raytraceDesc);

        // Restore the vars
        setComputeVars(pComputeVars);
    }
}
