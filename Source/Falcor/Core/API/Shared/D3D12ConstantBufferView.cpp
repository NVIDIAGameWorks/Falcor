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
#include "D3D12ConstantBufferView.h"
#include "Core/API/Device.h"
#include "Core/API/NativeHandleTraits.h"

namespace Falcor
{
ref<D3D12DescriptorSet> createCbvDescriptor(ref<Device> pDevice, const D3D12_CONSTANT_BUFFER_VIEW_DESC& desc)
{
    pDevice->requireD3D12();

    D3D12DescriptorSetLayout layout;
    layout.addRange(ShaderResourceType::Cbv, 0, 1);
    ref<D3D12DescriptorSet> handle = D3D12DescriptorSet::create(pDevice, pDevice->getD3D12CpuDescriptorPool(), layout);
    pDevice->getNativeHandle().as<ID3D12Device*>()->CreateConstantBufferView(&desc, handle->getCpuHandle(0));

    return handle;
}

ref<D3D12ConstantBufferView> D3D12ConstantBufferView::create(ref<Device> pDevice, uint64_t gpuAddress, uint32_t byteSize)
{
    FALCOR_ASSERT(pDevice);
    pDevice->requireD3D12();

    D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
    desc.BufferLocation = gpuAddress;
    desc.SizeInBytes = byteSize;

    return ref<D3D12ConstantBufferView>(new D3D12ConstantBufferView(nullptr, createCbvDescriptor(pDevice, desc)));
}

ref<D3D12ConstantBufferView> D3D12ConstantBufferView::create(ref<Device> pDevice, ref<Buffer> pBuffer)
{
    FALCOR_ASSERT(pDevice);
    pDevice->requireD3D12();

    FALCOR_ASSERT(pBuffer);
    D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
    desc.BufferLocation = pBuffer->getGpuAddress();
    desc.SizeInBytes = (uint32_t)pBuffer->getSize();

    return ref<D3D12ConstantBufferView>(new D3D12ConstantBufferView(pBuffer, createCbvDescriptor(pDevice, desc)));
}

ref<D3D12ConstantBufferView> D3D12ConstantBufferView::create(ref<Device> pDevice)
{
    FALCOR_ASSERT(pDevice);
    pDevice->requireD3D12();

    // GFX doesn't support constant buffer view.
    // We provide a raw D3D12 implementation for applications
    // that wish to use the raw D3D12DescriptorSet API.

    // Create a null view.
    D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
    return ref<D3D12ConstantBufferView>(new D3D12ConstantBufferView(nullptr, createCbvDescriptor(pDevice, desc)));
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12ConstantBufferView::getD3D12CpuHeapHandle() const
{
    return mpDescriptorSet->getCpuHandle(0);
}
} // namespace Falcor
