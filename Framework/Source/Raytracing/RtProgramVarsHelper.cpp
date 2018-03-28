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
#include "Framework.h"
#include "RtProgramVarsHelper.h"
#include "API/Device.h"
#include "API/LowLevel/DescriptorPool.h"
#include "API/D3D12/LowLevel/D3D12DescriptorData.h"
#include "API/D3D12/LowLevel/D3D12DescriptorHeap.h"

namespace Falcor
{
    RtVarsContext::SharedPtr RtVarsContext::create(CopyContext::SharedPtr pRtContext)
    {
        return SharedPtr(new RtVarsContext(pRtContext));
    }

    RtVarsContext::RtVarsContext(CopyContext::SharedPtr pRtContext) : mpRayTraceContext(pRtContext)
    {
        mpLowLevelData = LowLevelContextData::create(LowLevelContextData::CommandQueueType::Direct, nullptr);
        mpList = RtVarsCmdList::create();
        ID3D12GraphicsCommandList* pList = mpList.get();
        mpLowLevelData->setCommandList(pList);
    }

    RtVarsContext::~RtVarsContext()
    {
        // Release the low-level data before the list
        mpLowLevelData = nullptr;
        mpList = nullptr;
    }

    HRESULT RtVarsCmdList::QueryInterface(REFIID riid, void **ppvObject)
    {
        return gpDevice->getRenderContext()->getLowLevelData()->getCommandList()->QueryInterface(riid, ppvObject);
    }

    void RtVarsCmdList::SetGraphicsRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor)
    {
        uint32_t rootOffset = mpRootSignature->getElementByteOffset(RootParameterIndex);
        *(uint64_t*)(mpRootBase + rootOffset) = BaseDescriptor.ptr;
    }

    void RtVarsCmdList::SetGraphicsRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues)
    {
        assert(DestOffsetIn32BitValues == 0);
        uint32_t rootOffset = mpRootSignature->getElementByteOffset(RootParameterIndex);
        *(uint32_t*)(mpRootBase + rootOffset) = SrcData;
    }

    void RtVarsCmdList::SetGraphicsRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, const void *pSrcData, UINT DestOffsetIn32BitValues)
    {
        should_not_get_here();
    }

    void RtVarsCmdList::SetGraphicsRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation)
    {
        uint32_t rootOffset = mpRootSignature->getElementByteOffset(RootParameterIndex);
        assert((rootOffset % 8) == 0);
        *(D3D12_GPU_VIRTUAL_ADDRESS*)(mpRootBase + rootOffset) = BufferLocation;
    }

    void RtVarsCmdList::SetGraphicsRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation)
    {
        uint32_t rootOffset = mpRootSignature->getElementByteOffset(RootParameterIndex);
        assert((rootOffset % 8) == 0);
        *(D3D12_GPU_VIRTUAL_ADDRESS*)(mpRootBase + rootOffset) = BufferLocation;
    }

    void RtVarsCmdList::SetGraphicsRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation)
    {
        uint32_t rootOffset = mpRootSignature->getElementByteOffset(RootParameterIndex);
        assert((rootOffset % 8) == 0);
        *(D3D12_GPU_VIRTUAL_ADDRESS*)(mpRootBase + rootOffset) = BufferLocation;
    }
}
