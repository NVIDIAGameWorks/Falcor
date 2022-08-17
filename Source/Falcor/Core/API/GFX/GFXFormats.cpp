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
#include "GFXFormats.h"
#include "Core/API/Device.h"
#include "Core/API/Formats.h"
#include "Core/API/GFX/GFXAPI.h"

namespace Falcor
{
    gfx::Format getGFXFormat(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::Alpha32Float:
            return gfx::Format::Unknown;
        case ResourceFormat::Alpha8Unorm:
            return gfx::Format::Unknown;
        case ResourceFormat::BC1Unorm:
            return gfx::Format::BC1_UNORM;
        case ResourceFormat::BC1UnormSrgb:
            return gfx::Format::BC1_UNORM_SRGB;
        case ResourceFormat::BC2Unorm:
            return gfx::Format::BC2_UNORM;
        case ResourceFormat::BC2UnormSrgb:
            return gfx::Format::BC2_UNORM_SRGB;
        case ResourceFormat::BC3Unorm:
            return gfx::Format::BC3_UNORM;
        case ResourceFormat::BC3UnormSrgb:
            return gfx::Format::BC3_UNORM_SRGB;
        case ResourceFormat::BC4Snorm:
            return gfx::Format::BC4_SNORM;
        case ResourceFormat::BC4Unorm:
            return gfx::Format::BC4_UNORM;
        case ResourceFormat::BC5Snorm:
            return gfx::Format::BC5_SNORM;
        case ResourceFormat::BC5Unorm:
            return gfx::Format::BC5_UNORM;
        case ResourceFormat::BC6HS16:
            return gfx::Format::BC6H_SF16;
        case ResourceFormat::BC6HU16:
            return gfx::Format::BC6H_UF16;
        case ResourceFormat::BC7Unorm:
            return gfx::Format::BC7_UNORM;
        case ResourceFormat::BC7UnormSrgb:
            return gfx::Format::BC7_UNORM_SRGB;
        case ResourceFormat::BGRA8Unorm:
            return gfx::Format::B8G8R8A8_UNORM;
        case ResourceFormat::BGRA8UnormSrgb:
            return gfx::Format::B8G8R8A8_UNORM_SRGB;
        case ResourceFormat::BGRX8Unorm:
            return gfx::Format::B8G8R8X8_UNORM;
        case ResourceFormat::BGRX8UnormSrgb:
            return gfx::Format::B8G8R8X8_UNORM_SRGB;
        case ResourceFormat::D16Unorm:
            return gfx::Format::D16_UNORM;
        case ResourceFormat::D24UnormS8:
            return gfx::Format::Unknown;
        case ResourceFormat::D32Float:
            return gfx::Format::D32_FLOAT;
        case ResourceFormat::D32FloatS8X24:
            return gfx::Format::Unknown;
        case ResourceFormat::R11G11B10Float:
            return gfx::Format::R11G11B10_FLOAT;
        case ResourceFormat::R16Float:
            return gfx::Format::R16_FLOAT;
        case ResourceFormat::R16Int:
            return gfx::Format::R16_SINT;
        case ResourceFormat::R16Snorm:
            return gfx::Format::R16_SNORM;
        case ResourceFormat::R16Uint:
            return gfx::Format::R16_UINT;
        case ResourceFormat::R16Unorm:
            return gfx::Format::R16_UNORM;
        case ResourceFormat::R24UnormX8:
            return gfx::Format::Unknown;
        case ResourceFormat::R32Float:
            return gfx::Format::R32_FLOAT;
        case ResourceFormat::R32FloatX32:
            return gfx::Format::Unknown;
        case ResourceFormat::R32Int:
            return gfx::Format::R32_SINT;
        case ResourceFormat::R32Uint:
            return gfx::Format::R32_UINT;
        case ResourceFormat::R5G6B5Unorm:
            return gfx::Format::B5G6R5_UNORM;
        case ResourceFormat::R8Int:
            return gfx::Format::R8_SINT;
        case ResourceFormat::R8Snorm:
            return gfx::Format::R8_SNORM;
        case ResourceFormat::R8Uint:
            return gfx::Format::R8_UINT;
        case ResourceFormat::R8Unorm:
            return gfx::Format::R8_UNORM;
        case ResourceFormat::RG16Float:
            return gfx::Format::R16G16_FLOAT;
        case ResourceFormat::RG16Int:
            return gfx::Format::R16G16_SINT;
        case ResourceFormat::RG16Snorm:
            return gfx::Format::R16G16_SNORM;
        case ResourceFormat::RG16Uint:
            return gfx::Format::R16G16_UINT;
        case ResourceFormat::RG16Unorm:
            return gfx::Format::R16G16_UNORM;
        case ResourceFormat::RG32Float:
            return gfx::Format::R32G32_FLOAT;
        case ResourceFormat::RG32Int:
            return gfx::Format::R32G32_SINT;
        case ResourceFormat::RG32Uint:
            return gfx::Format::R32G32_UINT;
        case ResourceFormat::RG8Int:
            return gfx::Format::R8G8_SINT;
        case ResourceFormat::RG8Snorm:
            return gfx::Format::R8G8_SNORM;
        case ResourceFormat::RG8Uint:
            return gfx::Format::R8G8_UINT;
        case ResourceFormat::RG8Unorm:
            return gfx::Format::R8G8_UNORM;
        case ResourceFormat::RGB10A2Uint:
            return gfx::Format::R10G10B10A2_UINT;
        case ResourceFormat::RGB10A2Unorm:
            return gfx::Format::R10G10B10A2_UNORM;
        case ResourceFormat::RGB16Float:
            return gfx::Format::Unknown;
        case ResourceFormat::RGB16Int:
            return gfx::Format::Unknown;
        case ResourceFormat::RGB16Snorm:
            return gfx::Format::Unknown;
        case ResourceFormat::RGB16Uint:
            return gfx::Format::Unknown;
        case ResourceFormat::RGB16Unorm:
            return gfx::Format::Unknown;
        case ResourceFormat::RGB32Float:
            return gfx::Format::R32G32B32_FLOAT;
        case ResourceFormat::RGB32Int:
            return gfx::Format::R32G32B32_SINT;
        case ResourceFormat::RGB32Uint:
            return gfx::Format::R32G32B32_UINT;
        case ResourceFormat::RGB5A1Unorm:
            return gfx::Format::B5G5R5A1_UNORM;
        case ResourceFormat::RGB9E5Float:
            return gfx::Format::R9G9B9E5_SHAREDEXP;
        case ResourceFormat::RGBA16Float:
            return gfx::Format::R16G16B16A16_FLOAT;
        case ResourceFormat::RGBA16Int:
            return gfx::Format::R16G16B16A16_SINT;
        case ResourceFormat::RGBA16Uint:
            return gfx::Format::R16G16B16A16_UINT;
        case ResourceFormat::RGBA16Unorm:
            return gfx::Format::R16G16B16A16_UNORM;
        case ResourceFormat::RGBA32Float:
            return gfx::Format::R32G32B32A32_FLOAT;
        case ResourceFormat::RGBA32Int:
            return gfx::Format::R32G32B32A32_SINT;
        case ResourceFormat::RGBA32Uint:
            return gfx::Format::R32G32B32A32_UINT;
        case ResourceFormat::RGBA8Int:
            return gfx::Format::R8G8B8A8_SINT;
        case ResourceFormat::RGBA8Snorm:
            return gfx::Format::R8G8B8A8_SNORM;
        case ResourceFormat::RGBA8Uint:
            return gfx::Format::R8G8B8A8_UINT;
        case ResourceFormat::RGBA8Unorm:
            return gfx::Format::R8G8B8A8_UNORM;
        case ResourceFormat::RGBA8UnormSrgb:
            return gfx::Format::R8G8B8A8_UNORM_SRGB;
        default:
            return gfx::Format::Unknown;
        }
    }

    ResourceBindFlags getFormatBindFlags(ResourceFormat format)
    {
        gfx::ResourceStateSet stateSet;
        FALCOR_GFX_CALL(gpDevice->getApiHandle()->getFormatSupportedResourceStates(getGFXFormat(format), &stateSet));

        ResourceBindFlags flags = ResourceBindFlags::None;
        if (stateSet.contains(gfx::ResourceState::ConstantBuffer))
        {
            flags |= ResourceBindFlags::Constant;
        }
        if (stateSet.contains(gfx::ResourceState::VertexBuffer))
        {
            flags |= ResourceBindFlags::Vertex;
        }
        if (stateSet.contains(gfx::ResourceState::IndexBuffer))
        {
            flags |= ResourceBindFlags::Index;
        }
        if (stateSet.contains(gfx::ResourceState::IndirectArgument))
        {
            flags |= ResourceBindFlags::IndirectArg;
        }
        if (stateSet.contains(gfx::ResourceState::StreamOutput))
        {
            flags |= ResourceBindFlags::StreamOutput;
        }
        if (stateSet.contains(gfx::ResourceState::ShaderResource))
        {
            flags |= ResourceBindFlags::ShaderResource;
        }
        if (stateSet.contains(gfx::ResourceState::RenderTarget))
        {
            flags |= ResourceBindFlags::RenderTarget;
        }
        if (stateSet.contains(gfx::ResourceState::DepthRead) || stateSet.contains(gfx::ResourceState::DepthWrite))
        {
            flags |= ResourceBindFlags::DepthStencil;
        }
        if (stateSet.contains(gfx::ResourceState::UnorderedAccess))
        {
            flags |= ResourceBindFlags::UnorderedAccess;
        }
        if (stateSet.contains(gfx::ResourceState::AccelerationStructure))
        {
            flags |= ResourceBindFlags::AccelerationStructure;
        }
        flags |= ResourceBindFlags::Shared;
        return flags;
    }

}
