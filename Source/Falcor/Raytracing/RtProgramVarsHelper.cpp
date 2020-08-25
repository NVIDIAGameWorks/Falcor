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
#include "RtProgramVarsHelper.h"
#include "Core/API/Device.h"

namespace Falcor
{
    RtVarsContext::SharedPtr RtVarsContext::create()
    {
        return SharedPtr(new RtVarsContext());
    }

    RtVarsContext::RtVarsContext()
        : CopyContext(LowLevelContextData::CommandQueueType::Direct, nullptr)
    {
        mpList = RtVarsCmdList::create();
        assert(mpList);
        ID3D12GraphicsCommandList* pList = mpList.get();
        mpLowLevelData->setCommandList(pList);
    }

    RtVarsContext::~RtVarsContext()
    {
        // Release the low-level data before the list
        mpLowLevelData = nullptr;
        mpList = nullptr;
    }

    bool RtVarsContext::resourceBarrier(const Resource* pResource, Resource::State newState, const ResourceViewInfo* pViewInfo)
    {
        return gpDevice->getRenderContext()->resourceBarrier(pResource, newState, pViewInfo);
    }

    void RtVarsContext::uavBarrier(const Resource* pResource)
    {
        gpDevice->getRenderContext()->uavBarrier(pResource);
    }

    HRESULT RtVarsCmdList::QueryInterface(REFIID riid, void **ppvObject)
    {
        if (riid == __uuidof(ID3D12CommandList))
        {
            *ppvObject = dynamic_cast<ID3D12CommandList*>(this);
            return S_OK;
        }
        else if (riid == __uuidof(ID3D12GraphicsCommandList4))
        {
            *ppvObject = dynamic_cast<ID3D12GraphicsCommandList4*>(this);
            return S_OK;
        }
        else if (riid == __uuidof(ID3D12GraphicsCommandList3))
        {
            *ppvObject = dynamic_cast<ID3D12GraphicsCommandList3*>(this);
            return S_OK;
        }
        else if (riid == __uuidof(ID3D12GraphicsCommandList2))
        {
            *ppvObject = dynamic_cast<ID3D12GraphicsCommandList2*>(this);
            return S_OK;
        }
        else if (riid == __uuidof(ID3D12GraphicsCommandList1))
        {
            *ppvObject = dynamic_cast<ID3D12GraphicsCommandList1*>(this);
            return S_OK;
        }
        else if (riid == __uuidof(ID3D12GraphicsCommandList))
        {
            *ppvObject = dynamic_cast<ID3D12GraphicsCommandList*>(this);
            return S_OK;
        }
        else
        {
            *ppvObject = nullptr;
            return E_NOINTERFACE;
        }
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
        assert(DestOffsetIn32BitValues == 0);
        uint32_t rootOffset = mpRootSignature->getElementByteOffset(RootParameterIndex);

        auto pSrcCursor = (uint32_t*)pSrcData;
        auto pDstCursor = (uint32_t*)(mpRootBase + rootOffset);
        for( UINT ii = 0; ii < Num32BitValuesToSet; ++ii )
        {
            *pDstCursor++ = *pSrcCursor++;
        }
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
