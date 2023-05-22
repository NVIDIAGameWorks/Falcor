/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "GpuFence.h"
#include "Device.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/Assert.h"
#include "Core/Errors.h"

namespace Falcor
{
GpuFence::GpuFence(ref<Device> pDevice, bool shared) : mpDevice(pDevice), mCpuValue(1)
{
    FALCOR_ASSERT(mpDevice);
    gfx::IFence::Desc fenceDesc = {};
    fenceDesc.isShared = shared;
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createFence(fenceDesc, mGfxFence.writeRef()));
}

GpuFence::~GpuFence() = default;

ref<GpuFence> GpuFence::create(ref<Device> pDevice, bool shared)
{
    return ref<GpuFence>(new GpuFence(pDevice, shared));
}

uint64_t GpuFence::gpuSignal(CommandQueueHandle pQueue)
{
    pQueue->executeCommandBuffers(0, nullptr, mGfxFence, mCpuValue);
    mCpuValue++;
    return mCpuValue - 1;
}

void GpuFence::syncGpu(CommandQueueHandle pQueue)
{
    gfx::IFence* fences[1]{mGfxFence.get()};
    auto waitValue = mCpuValue - 1;
    FALCOR_GFX_CALL(pQueue->waitForFenceValuesOnDevice(std::size(fences), fences, &waitValue));
}

void GpuFence::syncCpu(std::optional<uint64_t> val)
{
    auto waitValue = val ? val.value() : mCpuValue - 1;
    uint64_t currentValue = 0;
    FALCOR_GFX_CALL(mGfxFence->getCurrentValue(&currentValue));
    if (currentValue < waitValue)
    {
        gfx::IFence* fences[1]{mGfxFence.get()};
        FALCOR_GFX_CALL(mpDevice->getGfxDevice()->waitForFences(std::size(fences), fences, &waitValue, true, -1));
    }
}

uint64_t GpuFence::getGpuValue() const
{
    uint64_t currentValue = 0;
    FALCOR_GFX_CALL(mGfxFence->getCurrentValue(&currentValue));
    return currentValue;
}

void GpuFence::setGpuValue(uint64_t val)
{
    FALCOR_GFX_CALL(mGfxFence->setCurrentValue(val));
}

SharedResourceApiHandle GpuFence::getSharedApiHandle() const
{
    gfx::InteropHandle sharedHandle;
    FALCOR_GFX_CALL(mGfxFence->getSharedHandle(&sharedHandle));
    return (SharedResourceApiHandle)sharedHandle.handleValue;
}

NativeHandle GpuFence::getNativeHandle() const
{
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(mGfxFence->getNativeHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(reinterpret_cast<ID3D12Fence*>(gfxNativeHandle.handleValue));
#endif
#if FALCOR_HAS_VULKAN
        // currently not supported
#endif
    return {};
}

void GpuFence::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

} // namespace Falcor
