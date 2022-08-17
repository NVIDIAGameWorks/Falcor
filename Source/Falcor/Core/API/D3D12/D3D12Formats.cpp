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
#include "Core/API/Device.h"
#include "Core/API/Formats.h"
#include "Core/API/D3D12/D3D12API.h"

namespace Falcor
{
    ResourceBindFlags getFormatBindFlags(ResourceFormat format)
    {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support;
        support.Format = getDxgiFormat(format);
        FALCOR_D3D_CALL(gpDevice->getApiHandle()->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)));

        ResourceBindFlags flags = ResourceBindFlags::None;
        auto dxgi1 = support.Support1;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_BUFFER) flags |= ResourceBindFlags::Constant;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER) flags |= ResourceBindFlags::Vertex;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_IA_INDEX_BUFFER) flags |= ResourceBindFlags::Index;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_SO_BUFFER) flags |= ResourceBindFlags::StreamOutput;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_TEXTURE1D) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_TEXTURE3D) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_TEXTURECUBE) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE_COMPARISON) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_SHADER_GATHER) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_SHADER_GATHER_COMPARISON) flags |= ResourceBindFlags::ShaderResource;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) flags |= ResourceBindFlags::RenderTarget;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) flags |= ResourceBindFlags::DepthStencil;
        if (dxgi1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) flags |= ResourceBindFlags::UnorderedAccess;

        return flags;
    }
}
