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
#include "Core/API/ComputeStateObject.h"
#include "Core/API/Device.h"
#include "Core/API/GFX/GFXAPI.h"

namespace Falcor
{
    void ComputeStateObject::apiInit()
    {
        gfx::ComputePipelineStateDesc computePipelineDesc = {};
        computePipelineDesc.program = mDesc.mpProgram->getApiHandle();
#if FALCOR_HAS_D3D12
        computePipelineDesc.d3d12RootSignatureOverride =
            mDesc.mpD3D12RootSignatureOverride ? (void*)mDesc.mpD3D12RootSignatureOverride->getApiHandle().GetInterfacePtr() : nullptr;
#endif
        FALCOR_GFX_CALL(gpDevice->getApiHandle()->createComputePipelineState(computePipelineDesc, mApiHandle.writeRef()));
    }

    const D3D12ComputeStateHandle& ComputeStateObject::getD3D12Handle()
    {
#if FALCOR_HAS_D3D12
        if (!mpD3D12Handle)
        {
            // Get back raw d3d12 pipeline state handle.
            gfx::InteropHandle handle = {};
            FALCOR_GFX_CALL(mApiHandle->getNativeHandle(&handle));
            FALCOR_ASSERT(handle.api == gfx::InteropHandleAPI::D3D12);
            mpD3D12Handle = D3D12ComputeStateHandle(reinterpret_cast<ID3D12PipelineState*>(handle.handleValue));
        }
        return mpD3D12Handle;
#else
        throw RuntimeError("D3D12 is not available.");
#endif
    }
}
