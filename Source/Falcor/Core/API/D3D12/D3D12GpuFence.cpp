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
#include "Core/API/GpuFence.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Core/Assert.h"

namespace Falcor
{
    struct FenceApiData
    {
        HANDLE eventHandle = INVALID_HANDLE_VALUE;
    };

    GpuFence::GpuFence() : mCpuValue(0) {}
    GpuFence::~GpuFence()
    {
        CloseHandle(mpApiData->eventHandle);
    }

    GpuFence::SharedPtr GpuFence::create(bool shared)
    {
        SharedPtr pFence = SharedPtr(new GpuFence());
        pFence->mpApiData.reset(new FenceApiData);
        pFence->mpApiData->eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (pFence->mpApiData->eventHandle == nullptr) throw RuntimeError("Failed to create an event object");

        FALCOR_ASSERT(gpDevice);
        ID3D12Device* pDevice = gpDevice->getApiHandle().GetInterfacePtr();
        HRESULT hr = pDevice->CreateFence(pFence->mCpuValue, shared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence->mApiHandle));
        if (FAILED(hr))
        {
            d3dTraceHR("Failed to create a fence object", hr);
            throw RuntimeError("Failed to create GPU fence");
        }

        pFence->mCpuValue++;
        return pFence;
    }

    uint64_t GpuFence::gpuSignal(CommandQueueHandle pQueue)
    {
        FALCOR_ASSERT(pQueue);
        FALCOR_D3D_CALL(pQueue->Signal(mApiHandle, mCpuValue));
        mCpuValue++;
        return mCpuValue - 1;
    }

    void GpuFence::syncGpu(CommandQueueHandle pQueue)
    {
        FALCOR_D3D_CALL(pQueue->Wait(mApiHandle, mCpuValue - 1));
    }

    void GpuFence::syncCpu(std::optional<uint64_t> val)
    {
        uint64_t syncVal = val ? val.value() : mCpuValue - 1;
        FALCOR_ASSERT(syncVal <= mCpuValue - 1);

        uint64_t gpuVal = getGpuValue();
        if (gpuVal < syncVal)
        {
            FALCOR_D3D_CALL(mApiHandle->SetEventOnCompletion(syncVal, mpApiData->eventHandle));
            WaitForSingleObject(mpApiData->eventHandle, INFINITE);
        }
    }

    uint64_t GpuFence::getGpuValue() const
    {
        return mApiHandle->GetCompletedValue();
    }

    void GpuFence::setGpuValue(uint64_t val)
    {
        mApiHandle->Signal(val);
    }

    SharedResourceApiHandle GpuFence::getSharedApiHandle() const
    {
        if (!mSharedApiHandle)
        {
            ID3D12DevicePtr pDevicePtr = gpDevice->getApiHandle();
            LPCWSTR name = NULL;
            SharedFenceApiHandle pHandle;

            HRESULT res = pDevicePtr->CreateSharedHandle(mApiHandle, 0, GENERIC_ALL, name, &pHandle);
            if (res == S_OK)
            {
                mSharedApiHandle = pHandle;
            }
            else
            {
                throw RuntimeError("Failed to create shared handle");
            }
        }
        return mSharedApiHandle;
    }

    const D3D12FenceHandle& GpuFence::getD3D12Handle() const
    {
        return mApiHandle;
    }
}
