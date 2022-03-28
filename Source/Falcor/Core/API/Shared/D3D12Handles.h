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
#pragma once

#if FALCOR_D3D12_AVAILABLE
#define NOMINMAX
#include <d3d12.h>
#include <dxgi1_4.h>
#include <comdef.h>
#endif

#define FALCOR_MAKE_SMART_COM_PTR(_a) _COM_SMARTPTR_TYPEDEF(_a, __uuidof(_a))

namespace Falcor
{
#if FALCOR_D3D12_AVAILABLE
    FALCOR_MAKE_SMART_COM_PTR(IDXGISwapChain3);
    FALCOR_MAKE_SMART_COM_PTR(IDXGIDevice);
    FALCOR_MAKE_SMART_COM_PTR(IDXGIAdapter1);
    FALCOR_MAKE_SMART_COM_PTR(IDXGIFactory4);
    FALCOR_MAKE_SMART_COM_PTR(ID3DBlob);

    FALCOR_MAKE_SMART_COM_PTR(ID3D12StateObject);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12Device);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12GraphicsCommandList);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12Debug);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12CommandQueue);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12CommandAllocator);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12DescriptorHeap);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12Resource);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12Fence);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12PipelineState);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12RootSignature);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12QueryHeap);
    FALCOR_MAKE_SMART_COM_PTR(ID3D12CommandSignature);

    using D3D12DeviceHandle = ID3D12DevicePtr;
    using D3D12GraphicsStateHandle = ID3D12PipelineStatePtr;
    using D3D12ComputeStateHandle = ID3D12PipelineStatePtr;
    using D3D12RaytracingStateHandle = ID3D12PipelineStatePtr;
    using D3D12CommandListHandle = ID3D12GraphicsCommandListPtr;
    using D3D12CommandQueueHandle = ID3D12CommandQueuePtr;
    using D3D12DescriptorHeapHandle = ID3D12DescriptorHeapPtr;
    using D3D12RootSignatureHandle = ID3D12RootSignaturePtr;
    using D3D12ResourceHandle = ID3D12ResourcePtr;
    using D3D12FenceHandle = ID3D12FencePtr;

    using D3D12DescriptorCpuHandle = D3D12_CPU_DESCRIPTOR_HANDLE;
    using D3D12DescriptorGpuHandle = D3D12_GPU_DESCRIPTOR_HANDLE;
#else
    using D3D12DeviceHandle = void*;
    using D3D12GraphicsStateHandle = void*;
    using D3D12ComputeStateHandle = void*;
    using D3D12RaytracingStateHandle = void*;
    using D3D12CommandListHandle = void*;
    using D3D12CommandQueueHandle = void*;
    using D3D12DescriptorHeapHandle = void*;
    using D3D12RootSignatureHandle = void*;
    using D3D12ResourceHandle = void*;
    using D3D12FenceHandle = void*;

    using D3D12DescriptorCpuHandle = void*;
    using D3D12DescriptorGpuHandle = void*;
#endif

    using D3D12DescriptorSetApiHandle = void*;

} // namespace Falcor
