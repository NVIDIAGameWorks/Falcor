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
#include "Core/Platform/PlatformHandles.h"
#include "Core/API/Shared/D3D12Handles.h"
#include <d3d12.h>
#include <memory>

namespace Falcor
{
    class D3D12DescriptorSet;
    class ShaderResourceView;

    FALCOR_MAKE_SMART_COM_PTR(IUnknown);

    using ApiObjectHandle = IUnknownPtr;

    class DescriptorHeapEntry;
    using DeviceHandle = ID3D12DevicePtr;
    using CommandListHandle = ID3D12GraphicsCommandListPtr;
    using CommandQueueHandle = ID3D12CommandQueuePtr;
    using ApiCommandQueueType = D3D12_COMMAND_LIST_TYPE;
    using CommandAllocatorHandle = ID3D12CommandAllocatorPtr;
    using CommandSignatureHandle = ID3D12CommandSignaturePtr;
    using FenceHandle = ID3D12FencePtr;
    using ResourceHandle = ID3D12ResourcePtr;
    using RtvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using DsvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using SrvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using SamplerHandle = std::shared_ptr<D3D12DescriptorSet>;
    using UavHandle = std::shared_ptr<D3D12DescriptorSet>;
    using CbvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using AccelerationStructureHandle = std::shared_ptr<ShaderResourceView>;
    using FboHandle = void*;
    using GpuAddress = D3D12_GPU_VIRTUAL_ADDRESS;
    using QueryHeapHandle = ID3D12QueryHeapPtr;
    using SharedResourceApiHandle = HANDLE;
    using SharedFenceApiHandle = HANDLE;

    using GraphicsStateHandle = ID3D12PipelineStatePtr;
    using ComputeStateHandle = ID3D12PipelineStatePtr;
    using RaytracingStateHandle = ID3D12StateObjectPtr;

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
}
