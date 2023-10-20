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
#pragma once
#include "D3D12DescriptorSet.h"
#include "Core/API/Buffer.h"
#include "Core/Macros.h"
#include "Core/Object.h"

#include <memory>

namespace Falcor
{
// GFX doesn't need constant buffer view.
// We provide a raw D3D12 implementation for applications
// that wish to use the raw D3D12DescriptorSet API.
class FALCOR_API D3D12ConstantBufferView : public Object
{
    FALCOR_OBJECT(D3D12ConstantBufferView)
public:
    static ref<D3D12ConstantBufferView> create(ref<Device> pDevice, uint64_t gpuAddress, uint32_t byteSize);
    static ref<D3D12ConstantBufferView> create(ref<Device> pDevice, ref<Buffer> pBuffer);
    static ref<D3D12ConstantBufferView> create(ref<Device> pDevice);

    /**
     * Get the D3D12 CPU descriptor handle representing this resource view.
     * Valid only when D3D12 is the underlying API.
     */
    D3D12_CPU_DESCRIPTOR_HANDLE getD3D12CpuHeapHandle() const;

private:
    D3D12ConstantBufferView(ref<Buffer> pBuffer, ref<D3D12DescriptorSet> pDescriptorSet)
        : mpBuffer(pBuffer), mpDescriptorSet(pDescriptorSet)
    {}

    ref<Buffer> mpBuffer;
    ref<D3D12DescriptorSet> mpDescriptorSet;
};
} // namespace Falcor
