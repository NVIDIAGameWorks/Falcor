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
#include "Core/API/GFX/GFXAPI.h"
#include "Core/Assert.h"
#include "Core/Errors.h"

namespace Falcor
{
    struct FenceApiData
    {
        Slang::ComPtr<gfx::IFence> gfxFence;
        D3D12FenceHandle d3d12Handle;
    };

    GpuFence::GpuFence() : mCpuValue(0) {}
    GpuFence::~GpuFence() = default;

    GpuFence::SharedPtr GpuFence::create(bool shared)
    {
        FALCOR_ASSERT(gpDevice);
        SharedPtr pFence = SharedPtr(new GpuFence());
        pFence->mpApiData.reset(new FenceApiData);
        gfx::IFence::Desc fenceDesc = {};
        if (SLANG_FAILED(gpDevice->getApiHandle()->createFence(fenceDesc, pFence->mpApiData->gfxFence.writeRef())))
        {
            throw RuntimeError("Failed to create a fence object");
        }
        pFence->mApiHandle = pFence->mpApiData->gfxFence;

        pFence->mCpuValue++;

#if FALCOR_HAS_D3D12
        gfx::InteropHandle nativeHandle = {};
        pFence->mpApiData->gfxFence->getNativeHandle(&nativeHandle);
        FALCOR_ASSERT(nativeHandle.api == gfx::InteropHandleAPI::D3D12);
        pFence->mpApiData->d3d12Handle = D3D12FenceHandle((ID3D12Fence*)nativeHandle.handleValue);
#endif
        return pFence;
    }

    uint64_t GpuFence::gpuSignal(CommandQueueHandle pQueue)
    {
        pQueue->executeCommandBuffers(0, nullptr, mpApiData->gfxFence.get(), mCpuValue);
        mCpuValue++;
        return mCpuValue - 1;
    }

    void GpuFence::syncGpu(CommandQueueHandle /*pQueue*/)
    {
    }

    void GpuFence::syncCpu(std::optional<uint64_t> val)
    {
        auto waitValue = val ? val.value() : mCpuValue - 1;
        gfx::IFence* gfxFence = mpApiData->gfxFence.get();
        uint64_t currentValue = 0;
        gfxFence->getCurrentValue(&currentValue);
        if (currentValue < waitValue)
        {
            gpDevice->getApiHandle()->waitForFences(1, &gfxFence, &waitValue, true, -1);
        }
    }

    uint64_t GpuFence::getGpuValue() const
    {
        uint64_t currentValue = 0;
        mpApiData->gfxFence->getCurrentValue(&currentValue);
        return currentValue;
    }

    void GpuFence::setGpuValue(uint64_t val)
    {
        mpApiData->gfxFence->setCurrentValue(val);
    }

    SharedResourceApiHandle GpuFence::getSharedApiHandle() const
    {
        gfx::InteropHandle sharedHandle;
        mpApiData->gfxFence->getSharedHandle(&sharedHandle);
        return (SharedResourceApiHandle)sharedHandle.handleValue;
    }

    const D3D12FenceHandle& GpuFence::getD3D12Handle() const
    {
#if FALCOR_HAS_D3D12
        return mpApiData->d3d12Handle;
#else
        throw RuntimeError("D3D12 is not available.");
#endif
    }
}
