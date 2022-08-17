/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/LowLevelContextData.h"
#include "D3D12ApiData.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Core/Assert.h"

namespace Falcor
{
    template<D3D12_COMMAND_LIST_TYPE type>
    static ID3D12CommandAllocatorPtr newCommandAllocator(void* pUserData)
    {
        ID3D12CommandAllocatorPtr pAllocator;
        FALCOR_D3D_CALL(gpDevice->getApiHandle()->CreateCommandAllocator(type, IID_PPV_ARGS(&pAllocator)));
        return pAllocator;
    }

    template<typename ApiType>
    ApiType createCommandList(ID3D12Device* pDevice, D3D12_COMMAND_LIST_TYPE type, CommandAllocatorHandle allocator)
    {
        ApiType pList;
        HRESULT hr = pDevice->CreateCommandList(0, type, allocator, nullptr, IID_PPV_ARGS(&pList));
        return (FAILED(hr)) ? nullptr : pList;
    }

    LowLevelContextData::SharedPtr LowLevelContextData::create(CommandQueueType type, CommandQueueHandle queue)
    {
        return SharedPtr(new LowLevelContextData(type, queue));
    }

    LowLevelContextData::LowLevelContextData(CommandQueueType type, CommandQueueHandle queue)
        : mType(type)
        , mpQueue(queue)
    {
        mpFence = GpuFence::create();
        mpApiData.reset(new LowLevelContextApiData);
        FALCOR_ASSERT(mpFence && mpApiData);

        // Create a command allocator
        D3D12_COMMAND_LIST_TYPE cmdListType = gpDevice->getApiCommandQueueType(type);
        switch (cmdListType)
        {
        case D3D12_COMMAND_LIST_TYPE_DIRECT:
            mpApiData->pAllocatorPool = FencedPool<CommandAllocatorHandle>::create(mpFence, newCommandAllocator<D3D12_COMMAND_LIST_TYPE_DIRECT>);
            break;
        case D3D12_COMMAND_LIST_TYPE_COMPUTE:
            mpApiData->pAllocatorPool = FencedPool<CommandAllocatorHandle>::create(mpFence, newCommandAllocator<D3D12_COMMAND_LIST_TYPE_COMPUTE>);
            break;
        case D3D12_COMMAND_LIST_TYPE_COPY:
            mpApiData->pAllocatorPool = FencedPool<CommandAllocatorHandle>::create(mpFence, newCommandAllocator<D3D12_COMMAND_LIST_TYPE_COPY>);
            break;
        default:
            FALCOR_UNREACHABLE();
        }
        mpAllocator = mpApiData->pAllocatorPool->newObject();
        FALCOR_ASSERT(mpAllocator);

        FALCOR_D3D_CALL(gpDevice->getApiHandle()->CreateCommandList(0, cmdListType, mpAllocator, nullptr, IID_PPV_ARGS(&mpList)));
        FALCOR_ASSERT(mpList);
    }

    LowLevelContextData::~LowLevelContextData() = default;

    void LowLevelContextData::flush()
    {
        FALCOR_D3D_CALL(mpList->Close());
        ID3D12CommandList* pList = mpList.GetInterfacePtr();
        FALCOR_ASSERT(mpQueue);
        mpQueue->ExecuteCommandLists(1, &pList);
        mpFence->gpuSignal(mpQueue);
        mpAllocator = mpApiData->pAllocatorPool->newObject();
        FALCOR_D3D_CALL(mpAllocator->Reset());
        FALCOR_D3D_CALL(mpList->Reset(mpAllocator, nullptr));
    }

    const D3D12CommandListHandle& LowLevelContextData::getD3D12CommandList() const
    {
        return mpList;
    }

    const D3D12CommandQueueHandle& LowLevelContextData::getD3D12CommandQueue() const
    {
        return mpQueue;
    }
}
