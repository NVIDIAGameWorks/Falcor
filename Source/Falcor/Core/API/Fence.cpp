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
#include "Fence.h"
#include "Device.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/Error.h"

namespace Falcor
{
Fence::Fence(ref<Device> pDevice, FenceDesc desc) : mpDevice(pDevice), mDesc(desc)
{
    FALCOR_ASSERT(mpDevice);
    gfx::IFence::Desc gfxDesc = {};
    mSignaledValue = mDesc.initialValue;
    gfxDesc.isShared = mDesc.shared;
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createFence(gfxDesc, mGfxFence.writeRef()));
}

Fence::~Fence() = default;

uint64_t Fence::signal(uint64_t value)
{
    uint64_t signalValue = updateSignaledValue(value);
    FALCOR_GFX_CALL(mGfxFence->setCurrentValue(signalValue));
    return signalValue;
}

void Fence::wait(uint64_t value, uint64_t timeoutNs)
{
    uint64_t waitValue = value == kAuto ? mSignaledValue : value;
    uint64_t currentValue = getCurrentValue();
    if (currentValue >= waitValue)
        return;
    gfx::IFence* fences[] = {mGfxFence};
    uint64_t waitValues[] = {waitValue};
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->waitForFences(1, fences, waitValues, true, timeoutNs));
}

uint64_t Fence::getCurrentValue()
{
    uint64_t value;
    FALCOR_GFX_CALL(mGfxFence->getCurrentValue(&value));
    return value;
}

uint64_t Fence::updateSignaledValue(uint64_t value)
{
    mSignaledValue = value == kAuto ? mSignaledValue + 1 : value;
    return mSignaledValue;
}

SharedResourceApiHandle Fence::getSharedApiHandle() const
{
    gfx::InteropHandle sharedHandle;
    FALCOR_GFX_CALL(mGfxFence->getSharedHandle(&sharedHandle));
    return (SharedResourceApiHandle)sharedHandle.handleValue;
}

NativeHandle Fence::getNativeHandle() const
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

void Fence::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

} // namespace Falcor
