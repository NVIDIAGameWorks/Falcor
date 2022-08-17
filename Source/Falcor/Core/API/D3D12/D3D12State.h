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
#include "Core/API/VAO.h"
#include "Core/API/GraphicsStateObject.h"
#include <d3d12.h>
#include <memory>
#include <vector>

namespace Falcor
{
    class BlendState;
    class RasterizerState;
    class DepthStencilState;
    class VertexLayout;
    class Sampler;

    // We need this because the D3D objects require a string. This ensures that once the descs are destroyed, the strings get destroyed as well
    struct InputLayoutDesc
    {
        std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
        std::vector<std::unique_ptr<char[]>> names; // Can't use strings directly because the vector size is unknown and vector reallocations will change the addresses we used in INPUT_ELEMENT_DESC
    };

    void initD3D12BlendDesc(const BlendState* pFalcorDesc, D3D12_BLEND_DESC& d3dDesc);
    void initD3D12RasterizerDesc(const RasterizerState* pState, D3D12_RASTERIZER_DESC& desc);
    void initD3DDepthStencilDesc(const DepthStencilState* pState, D3D12_DEPTH_STENCIL_DESC& desc);
    void initD3D12VertexLayout(const VertexLayout* pLayout, InputLayoutDesc& inputDesc);
    void initD3D12SamplerDesc(const Sampler* pSampler, D3D12_SAMPLER_DESC& desc);
    void initD3D12GraphicsStateDesc(const GraphicsStateObject::Desc& gsoDesc, D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, InputLayoutDesc& layoutDesc);

    inline D3D_PRIMITIVE_TOPOLOGY getD3DPrimitiveTopology(Vao::Topology topology)
    {
        switch (topology)
        {
        case Vao::Topology::PointList:
            return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        case Vao::Topology::LineList:
            return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        case Vao::Topology::LineStrip:
            return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        case Vao::Topology::TriangleList:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        case Vao::Topology::TriangleStrip:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        default:
            FALCOR_UNREACHABLE();
            return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
        }
    }

    inline D3D12_PRIMITIVE_TOPOLOGY_TYPE getD3DPrimitiveType(GraphicsStateObject::PrimitiveType type)
    {
        switch (type)
        {
        case GraphicsStateObject::PrimitiveType::Point:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        case GraphicsStateObject::PrimitiveType::Line:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        case GraphicsStateObject::PrimitiveType::Triangle:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        case GraphicsStateObject::PrimitiveType::Patch:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
        default:
            FALCOR_UNREACHABLE();
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
        }
    }
}
