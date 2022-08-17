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
#include "Core/Macros.h"
#include "Core/Platform/PlatformHandles.h"
#include <slang.h>
#include <slang-gfx.h>
#include <slang-com-ptr.h>
#include <memory>

namespace Falcor
{
    using ApiObjectHandle = Slang::ComPtr<ISlangUnknown>;
    using DeviceHandle = Slang::ComPtr<gfx::IDevice>;

    using CommandListHandle = Slang::ComPtr<gfx::ICommandBuffer>;
    using CommandQueueHandle = Slang::ComPtr<gfx::ICommandQueue>;
    using ApiCommandQueueType = gfx::ICommandQueue::QueueType;
    using CommandAllocatorHandle = Slang::ComPtr<gfx::ITransientResourceHeap>;
    using CommandSignatureHandle = void*;
    using FenceHandle = Slang::ComPtr<gfx::IFence>;
    using ResourceHandle = Slang::ComPtr<gfx::IResource>;
    using RtvHandle = Slang::ComPtr<gfx::IResourceView>;
    using DsvHandle = Slang::ComPtr<gfx::IResourceView>;
    using SrvHandle = Slang::ComPtr<gfx::IResourceView>;

    class D3D12DescriptorSet;
    using CbvHandle = std::shared_ptr<D3D12DescriptorSet>;

    using SamplerHandle = Slang::ComPtr<gfx::ISamplerState>;
    using UavHandle = Slang::ComPtr<gfx::IResourceView>;
    using AccelerationStructureHandle = Slang::ComPtr<gfx::IAccelerationStructure>;
    using FboHandle = Slang::ComPtr<gfx::IFramebuffer>;
    using GpuAddress = uint64_t;
    using QueryHeapHandle = Slang::ComPtr<gfx::IQueryPool>;
#if FALCOR_WINDOWS
    using SharedResourceApiHandle = HANDLE;
    using SharedFenceApiHandle = HANDLE;
#elif FALCOR_LINUX
    using SharedResourceApiHandle = void*;
    using SharedFenceApiHandle = void*;
#endif

    using GraphicsStateHandle = Slang::ComPtr<gfx::IPipelineState>;
    using ComputeStateHandle = Slang::ComPtr<gfx::IPipelineState>;
    using RaytracingStateHandle = Slang::ComPtr<gfx::IPipelineState>;

    using VaoHandle = Slang::ComPtr<gfx::IInputLayout>;

    using ShaderHandle = Slang::ComPtr<slang::IComponentType>;

    using VertexShaderHandle = void*;
    using FragmentShaderHandle = void*;
    using DomainShaderHandle = void*;
    using HullShaderHandle = void*;
    using GeometryShaderHandle = void*;
    using ComputeShaderHandle = void*;
    using ProgramHandle = Slang::ComPtr<gfx::IShaderProgram>;
    using DepthStencilStateHandle = void*;
    using RasterizerStateHandle = void*;
    using BlendStateHandle = void*;
}
