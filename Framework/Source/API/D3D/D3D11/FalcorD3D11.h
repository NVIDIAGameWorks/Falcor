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
#include <d3d11_1.h>

namespace Falcor
{
    /*!
    *  \addtogroup Falcor
    *  @{
    */

    // Device
    MAKE_SMART_COM_PTR(ID3D11Device);
    MAKE_SMART_COM_PTR(ID3D11DeviceContext);
    MAKE_SMART_COM_PTR(ID3D11InputLayout);

    // Resource
    MAKE_SMART_COM_PTR(ID3D11RenderTargetView);
    MAKE_SMART_COM_PTR(ID3D11DepthStencilView);
    MAKE_SMART_COM_PTR(ID3D11UnorderedAccessView);
    MAKE_SMART_COM_PTR(ID3D11ShaderResourceView);
    MAKE_SMART_COM_PTR(ID3D11Buffer);
    MAKE_SMART_COM_PTR(ID3D11Resource);
    MAKE_SMART_COM_PTR(ID3D11Texture1D);
    MAKE_SMART_COM_PTR(ID3D11Texture2D);
    MAKE_SMART_COM_PTR(ID3D11Texture3D);

    // Shaders
    MAKE_SMART_COM_PTR(ID3D11VertexShader);
    MAKE_SMART_COM_PTR(ID3D11PixelShader);
    MAKE_SMART_COM_PTR(ID3D11DomainShader);
    MAKE_SMART_COM_PTR(ID3D11HullShader);
    MAKE_SMART_COM_PTR(ID3D11GeometryShader);
    MAKE_SMART_COM_PTR(ID3D11ComputeShader);
    MAKE_SMART_COM_PTR(ID3DBlob);

    // Reflection
    MAKE_SMART_COM_PTR(ID3D11ShaderReflection);
    MAKE_SMART_COM_PTR(ID3D11ShaderReflectionVariable);

    // State
    MAKE_SMART_COM_PTR(ID3D11DepthStencilState);
    MAKE_SMART_COM_PTR(ID3D11RasterizerState);
    MAKE_SMART_COM_PTR(ID3D11BlendState);
    MAKE_SMART_COM_PTR(ID3D11SamplerState);

    ID3D11DevicePtr getD3D11Device();
    ID3D11DeviceContextPtr getD3D11ImmediateContext();

    using TextureHandle             = ID3D11ResourcePtr;
    using BufferHandle              = ID3D11BufferPtr;
    using VaoHandle                 = uint32_t;
    using VertexShaderHandle        = ID3D11VertexShaderPtr;
    using FragmentShaderHandle      = ID3D11PixelShaderPtr;
    using DomainShaderHandle        = ID3D11DomainShaderPtr;
    using HullShaderHandle          = ID3D11HullShaderPtr;
    using GeometryShaderHandle      = ID3D11GeometryShaderPtr;
    using ComputeShaderHandle       = ID3D11ComputeShaderPtr;
    using ProgramHandle             = uint32_t;
    using DepthStencilStateHandle   = ID3D11DepthStencilStatePtr;
    using RasterizerStateHandle     = ID3D11RasterizerStatePtr;
    using BlendStateHandle          = ID3D11BlendStatePtr;
    using SamplerApiHandle          = ID3D11SamplerStatePtr;
    using ShaderResourceViewHandle  = ID3D11ShaderResourceViewPtr;

    inline constexpr uint32_t getMaxViewportCount() { return D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE; }

    /*! @} */
}

#pragma comment(lib, "d3d11.lib")

#define DEFAULT_API_MAJOR_VERSION 11
#define DEFAULT_API_MINOR_VERSION 1

#define UNSUPPORTED_IN_D3D11(msg_) {Falcor::Logger::log(Falcor::Logger::Level::Warning, msg_ + std::string(" is not supported in D3D11. Ignoring call."));}
