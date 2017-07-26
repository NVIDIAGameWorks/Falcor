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
#include "Framework.h"
#include "D3D12Resource.h"

namespace Falcor
{
    const D3D12_HEAP_PROPERTIES kDefaultHeapProps =
    {
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        0,
        0
    };

    const D3D12_HEAP_PROPERTIES kUploadHeapProps =
    {
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        0,
        0,
    };

    const D3D12_HEAP_PROPERTIES kReadbackHeapProps =
    {
        D3D12_HEAP_TYPE_READBACK,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        0,
        0
    };

    D3D12_RESOURCE_FLAGS getD3D12ResourceFlags(Resource::BindFlags flags)
    {
        D3D12_RESOURCE_FLAGS d3d = D3D12_RESOURCE_FLAG_NONE;

        if (is_set(flags, Resource::BindFlags::UnorderedAccess))
        {
            d3d |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        if (is_set(flags, Resource::BindFlags::DepthStencil))
        {
            if (is_set(flags, Resource::BindFlags::ShaderResource) == false)
            {
                d3d |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
            }
            d3d |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        }

        if (is_set(flags, Resource::BindFlags::RenderTarget))
        {
            d3d |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        
        return d3d;
    }

    D3D12_RESOURCE_STATES getD3D12ResourceState(Resource::State s)
    {
        switch (s)
        {
        case Resource::State::Undefined:
        case Resource::State::Common:
            return D3D12_RESOURCE_STATE_COMMON;
        case Resource::State::ConstantBuffer:
        case Resource::State::VertexBuffer:
            return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        case Resource::State::CopyDest:
            return D3D12_RESOURCE_STATE_COPY_DEST;
        case Resource::State::CopySource:
            return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case Resource::State::DepthStencil:
            return D3D12_RESOURCE_STATE_DEPTH_WRITE; // If depth-writes are disabled, return D3D12_RESOURCE_STATE_DEPTH_WRITE
        case Resource::State::IndexBuffer:
            return D3D12_RESOURCE_STATE_INDEX_BUFFER;
        case Resource::State::IndirectArg:
            return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        case Resource::State::Predication:
            return D3D12_RESOURCE_STATE_PREDICATION;
        case Resource::State::Present:
            return D3D12_RESOURCE_STATE_PRESENT;
        case Resource::State::RenderTarget:
            return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case Resource::State::ResolveDest:
            return D3D12_RESOURCE_STATE_RESOLVE_DEST;
        case Resource::State::ResolveSource:
            return D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        case Resource::State::ShaderResource:
            return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE; // Need the shader usage mask in case the SRV is used by non-PS
        case Resource::State::StreamOut:
            return D3D12_RESOURCE_STATE_STREAM_OUT;
        case Resource::State::UnorderedAccess:
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case Resource::State::GenericRead:
            return D3D12_RESOURCE_STATE_GENERIC_READ;
        default:
            should_not_get_here();
            return D3D12_RESOURCE_STATE_GENERIC_READ;
        }
    }
}
