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
#include "D3D12RtProgramVarsHelper.h"

namespace Falcor
{
    uint32_t RtProgramVars::getProgramIdentifierSize()
    {
        return D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    }

    bool RtProgramVars::applyRtProgramVars(uint8_t* pRecord, const RtProgramVersion* pProgVersion, const RtStateObject* pRtso, ProgramVars* pVars, RtVarsContext* pContext)
    {
        MAKE_SMART_COM_PTR(ID3D12StateObjectProperties);
        ID3D12StateObjectPropertiesPtr pRtsoPtr = pRtso->getApiHandle();
        memcpy(pRecord, pRtsoPtr->GetShaderIdentifier(pProgVersion->getExportName().c_str()), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        pRecord += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        pContext->getRtVarsCmdList()->setRootParams(pProgVersion->getLocalRootSignature(), pRecord);
        return pVars->applyProgramVarsCommon<true>(pContext, true);
    }

    void RtVarsContext::apiInit()
    {
        mpList = RtVarsCmdList::create();

        ID3D12GraphicsCommandList* pList = mpList.get();
        mpLowLevelData->setCommandList(pList);
    }
}
