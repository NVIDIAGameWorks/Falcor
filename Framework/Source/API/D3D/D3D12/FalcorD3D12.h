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
#pragma once
#define NOMINMAX
#include <d3d12.h>

namespace Falcor
{
    class DescriptorSet;
    /*!
    *  \addtogroup Falcor
    *  @{
    */

    // Device
    MAKE_SMART_COM_PTR(ID3D12Device);
    MAKE_SMART_COM_PTR(ID3D12Debug);
    MAKE_SMART_COM_PTR(ID3D12CommandQueue);
    MAKE_SMART_COM_PTR(ID3D12CommandAllocator);
    MAKE_SMART_COM_PTR(ID3D12GraphicsCommandList);
    MAKE_SMART_COM_PTR(ID3D12DescriptorHeap);
    MAKE_SMART_COM_PTR(ID3D12Resource);
    MAKE_SMART_COM_PTR(ID3D12Fence);
    MAKE_SMART_COM_PTR(ID3D12PipelineState);
    MAKE_SMART_COM_PTR(ID3D12ShaderReflection);
    MAKE_SMART_COM_PTR(ID3D12RootSignature);
    MAKE_SMART_COM_PTR(ID3D12QueryHeap);
    MAKE_SMART_COM_PTR(ID3D12CommandSignature);
    MAKE_SMART_COM_PTR(IUnknown);
    
    using ApiObjectHandle = IUnknownPtr;

    using HeapCpuHandle = D3D12_CPU_DESCRIPTOR_HANDLE;
    using HeapGpuHandle = D3D12_GPU_DESCRIPTOR_HANDLE;

    class DescriptorHeapEntry;

	using WindowHandle = HWND;
	using DeviceHandle = ID3D12DevicePtr;
	using CommandListHandle = ID3D12GraphicsCommandListPtr;
	using CommandQueueHandle = ID3D12CommandQueuePtr;
    using ApiCommandQueueType = D3D12_COMMAND_LIST_TYPE;
    using CommandAllocatorHandle = ID3D12CommandAllocatorPtr;
    using CommandSignatureHandle = ID3D12CommandSignaturePtr;
    using FenceHandle = ID3D12FencePtr;
    using ResourceHandle = ID3D12ResourcePtr;
    using RtvHandle = std::shared_ptr<DescriptorSet>;
    using DsvHandle = std::shared_ptr<DescriptorSet>;
    using SrvHandle = std::shared_ptr<DescriptorSet>;
    using SamplerHandle = std::shared_ptr<DescriptorSet>;
    using UavHandle = std::shared_ptr<DescriptorSet>;
    using CbvHandle = std::shared_ptr<DescriptorSet>;
    using FboHandle = void*;
    using GpuAddress = D3D12_GPU_VIRTUAL_ADDRESS;
    using QueryHeapHandle = ID3D12QueryHeapPtr;

    using GraphicsStateHandle = ID3D12PipelineStatePtr;
    using ComputeStateHandle = ID3D12PipelineStatePtr;
    using ShaderHandle = D3D12_SHADER_BYTECODE;
    using RootSignatureHandle = ID3D12RootSignaturePtr;
    using DescriptorHeapHandle = ID3D12DescriptorHeapPtr;

    using VaoHandle = void*;
    using VertexShaderHandle = void*;
    using FragmentShaderHandle = void*;
    using DomainShaderHandle = void*;
    using HullShaderHandle = void*;
    using GeometryShaderHandle = void*;
    using ComputeShaderHandle = void*;
    using ProgramHandle = void*;
    using DepthStencilStateHandle = void*;
    using RasterizerStateHandle = void*;
    using BlendStateHandle = void*;
    using DescriptorSetApiHandle = void*;

    static const uint32_t kDefaultSwapChainBuffers = 3;

    inline constexpr uint32_t getMaxViewportCount() { return D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE; }
    /*! @} */
}

#pragma comment(lib, "d3d12.lib")

#define DEFAULT_API_MAJOR_VERSION 11
#define DEFAULT_API_MINOR_VERSION 1

#define UNSUPPORTED_IN_D3D12(msg_) {Falcor::logWarning(msg_ + std::string(" is not supported in D3D12. Ignoring call."));}
#define D3Dx(a) D3D12_##a
