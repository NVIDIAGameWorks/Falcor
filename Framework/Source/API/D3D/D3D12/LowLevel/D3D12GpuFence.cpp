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
#include "API/LowLevel/GpuFence.h"
#include "API/Device.h"

namespace Falcor
{
    struct FenceApiData
    {
        HANDLE eventHandle = INVALID_HANDLE_VALUE;
    };

    GpuFence::~GpuFence()
    {
        CloseHandle(mpApiData->eventHandle);
        safe_delete(mpApiData);
    }

    GpuFence::SharedPtr GpuFence::create()
    {
        SharedPtr pFence = SharedPtr(new GpuFence());
        pFence->mpApiData = new FenceApiData;
        pFence->mpApiData->eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        ID3D12Device* pDevice = gpDevice->getApiHandle().GetInterfacePtr();

        HRESULT hr = pDevice->CreateFence(pFence->mCpuValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence->mApiHandle));
        pFence->mCpuValue++;
        if(FAILED(hr))
        {
            d3dTraceHR("Failed to create a fence object", hr);
            return nullptr;
        }

        return pFence;
    }

    uint64_t GpuFence::gpuSignal(CommandQueueHandle pQueue)
    {
        d3d_call(pQueue->Signal(mApiHandle, mCpuValue));
        mCpuValue++;
        return mCpuValue - 1;
    }

    void GpuFence::syncGpu(CommandQueueHandle pQueue)
    {
        d3d_call(pQueue->Wait(mApiHandle, mCpuValue - 1));
    }

    void GpuFence::syncCpu()
    {
        uint64_t gpuVal = getGpuValue();
        if (gpuVal < mCpuValue - 1)
        {
            d3d_call(mApiHandle->SetEventOnCompletion(mCpuValue - 1, mpApiData->eventHandle));
            WaitForSingleObject(mpApiData->eventHandle, INFINITE);
        }
    }

    uint64_t GpuFence::getGpuValue() const
    {
        return mApiHandle->GetCompletedValue();
    }
}
