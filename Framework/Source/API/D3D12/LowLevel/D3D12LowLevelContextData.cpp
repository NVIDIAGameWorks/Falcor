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
#include "Framework.h"
#include "API/LowLevel/LowLevelContextData.h"
#include "API/Device.h"
#include "API/D3D12/D3D12ApiData.h"

namespace Falcor
{
    template<D3D12_COMMAND_LIST_TYPE type>
    static ID3D12CommandAllocatorPtr newCommandAllocator(void* pUserData)
    {
        ID3D12CommandAllocatorPtr pAllocator;
        if (FAILED(gpDevice->getApiHandle()->CreateCommandAllocator(type, IID_PPV_ARGS(&pAllocator))))
        {
            logError("Failed to create command allocator");
            return nullptr;
        }
        return pAllocator;
    }

    LowLevelContextData::SharedPtr LowLevelContextData::create(CommandQueueType type, CommandQueueHandle queue)
    {
        SharedPtr pThis = SharedPtr(new LowLevelContextData);
        pThis->mpFence = GpuFence::create();
        pThis->mpQueue = queue;
        pThis->mpApiData = new LowLevelContextApiData;

        // Create a command allocator
        D3D12_COMMAND_LIST_TYPE cmdListType = gpDevice->getApiCommandQueueType(type);
        switch (cmdListType)
        {
        case D3D12_COMMAND_LIST_TYPE_DIRECT:
            pThis->mpApiData->pAllocatorPool = FencedPool<CommandAllocatorHandle>::create(pThis->mpFence, newCommandAllocator<D3D12_COMMAND_LIST_TYPE_DIRECT>);
            break;
        case D3D12_COMMAND_LIST_TYPE_COMPUTE:
            pThis->mpApiData->pAllocatorPool = FencedPool<CommandAllocatorHandle>::create(pThis->mpFence, newCommandAllocator<D3D12_COMMAND_LIST_TYPE_COMPUTE>);
            break;
        case D3D12_COMMAND_LIST_TYPE_COPY:
            pThis->mpApiData->pAllocatorPool = FencedPool<CommandAllocatorHandle>::create(pThis->mpFence, newCommandAllocator<D3D12_COMMAND_LIST_TYPE_COPY>);
            break;
        default:
            should_not_get_here();
        }
        pThis->mpAllocator = pThis->mpApiData->pAllocatorPool->newObject();

        // Create a command list
        ID3D12Device* pDevice = gpDevice->getApiHandle().GetInterfacePtr();
        if (FAILED(pDevice->CreateCommandList(0, cmdListType, pThis->mpAllocator, nullptr, IID_PPV_ARGS(&pThis->mpList))))
        {
            logError("Failed to create command list for LowLevelContextData");
            return nullptr;
        }
        return pThis;
    }

    LowLevelContextData::~LowLevelContextData()
    {
        safe_delete(mpApiData);
    }

    void LowLevelContextData::reset()
    {
        mpFence->gpuSignal(mpQueue);
        mpAllocator = mpApiData->pAllocatorPool->newObject();
        d3d_call(mpList->Close());
        d3d_call(mpAllocator->Reset());
        d3d_call(mpList->Reset(mpAllocator, nullptr));
    }

    void LowLevelContextData::flush()
    {
        d3d_call(mpList->Close());
        ID3D12CommandList* pList = mpList.GetInterfacePtr();
        mpQueue->ExecuteCommandLists(1, &pList);
        mpFence->gpuSignal(mpQueue);
        mpList->Reset(mpAllocator, nullptr);
    }
}
