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
#include "ComputeStateObject.h"
#include "Device.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"

#if FALCOR_HAS_D3D12
#include "Shared/D3D12RootSignature.h"
#endif

namespace Falcor
{
bool ComputeStateObject::Desc::operator==(const ComputeStateObject::Desc& other) const
{
    bool b = true;
    b = b && (mpProgram == other.mpProgram);
    return b;
}

ComputeStateObject::~ComputeStateObject()
{
    mpDevice->releaseResource(mGfxPipelineState);
}

ComputeStateObject::ComputeStateObject(std::shared_ptr<Device> pDevice, const Desc& desc) : mpDevice(std::move(pDevice)), mDesc(desc)
{
    gfx::ComputePipelineStateDesc computePipelineDesc = {};
    computePipelineDesc.program = mDesc.mpProgram->getGfxProgram();
#if FALCOR_HAS_D3D12
    if (mDesc.mpD3D12RootSignatureOverride)
        mpDevice->requireD3D12();
    if (mpDevice->getType() == Device::Type::D3D12)
    {
        computePipelineDesc.d3d12RootSignatureOverride =
            mDesc.mpD3D12RootSignatureOverride ? (void*)mDesc.mpD3D12RootSignatureOverride->getApiHandle().GetInterfacePtr() : nullptr;
    }
#endif
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createComputePipelineState(computePipelineDesc, mGfxPipelineState.writeRef()));
}

ComputeStateObject::SharedPtr ComputeStateObject::create(Device* pDevice, const Desc& desc)
{
    return SharedPtr(new ComputeStateObject(pDevice->shared_from_this(), desc));
}

NativeHandle ComputeStateObject::getNativeHandle() const
{
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(mGfxPipelineState->getNativeHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(reinterpret_cast<ID3D12PipelineState*>(gfxNativeHandle.handleValue));
#endif
#if FALCOR_HAS_VULKAN
    if (mpDevice->getType() == Device::Type::Vulkan)
        return NativeHandle(reinterpret_cast<VkPipeline>(gfxNativeHandle.handleValue));
#endif
    return {};
}
} // namespace Falcor
