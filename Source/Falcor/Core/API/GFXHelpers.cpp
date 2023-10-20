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
#include "GFXHelpers.h"
#include "Device.h"
#include "Formats.h"
#include "GFXAPI.h"

namespace Falcor
{
gfx::Format getGFXFormat(ResourceFormat format)
{
    switch (format)
    {
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
    case ResourceFormat::BGRA4Unorm:
        return gfx::Format::B4G4R4A4_UNORM;
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
    case ResourceFormat::D32Float:
        return gfx::Format::D32_FLOAT;
    case ResourceFormat::D32FloatS8Uint:
        return gfx::Format::D32_FLOAT_S8_UINT;
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
    case ResourceFormat::R32Float:
        return gfx::Format::R32_FLOAT;
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
    case ResourceFormat::RGBA16Snorm:
        return gfx::Format::R16G16B16A16_SNORM;
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

gfx::ResourceState getGFXResourceState(Resource::State state)
{
    switch (state)
    {
    case Resource::State::Undefined:
        return gfx::ResourceState::Undefined;
    case Resource::State::PreInitialized:
        return gfx::ResourceState::PreInitialized;
    case Resource::State::Common:
        return gfx::ResourceState::General;
    case Resource::State::VertexBuffer:
        return gfx::ResourceState::VertexBuffer;
    case Resource::State::ConstantBuffer:
        return gfx::ResourceState::ConstantBuffer;
    case Resource::State::IndexBuffer:
        return gfx::ResourceState::IndexBuffer;
    case Resource::State::RenderTarget:
        return gfx::ResourceState::RenderTarget;
    case Resource::State::UnorderedAccess:
        return gfx::ResourceState::UnorderedAccess;
    case Resource::State::DepthStencil:
        return gfx::ResourceState::DepthWrite;
    case Resource::State::ShaderResource:
        return gfx::ResourceState::ShaderResource;
    case Resource::State::StreamOut:
        return gfx::ResourceState::StreamOutput;
    case Resource::State::IndirectArg:
        return gfx::ResourceState::IndirectArgument;
    case Resource::State::CopyDest:
        return gfx::ResourceState::CopyDestination;
    case Resource::State::CopySource:
        return gfx::ResourceState::CopySource;
    case Resource::State::ResolveDest:
        return gfx::ResourceState::ResolveDestination;
    case Resource::State::ResolveSource:
        return gfx::ResourceState::ResolveSource;
    case Resource::State::Present:
        return gfx::ResourceState::Present;
    case Resource::State::GenericRead:
        return gfx::ResourceState::General;
    case Resource::State::Predication:
        return gfx::ResourceState::General;
    case Resource::State::PixelShader:
        return gfx::ResourceState::PixelShaderResource;
    case Resource::State::NonPixelShader:
        return gfx::ResourceState::NonPixelShaderResource;
    case Resource::State::AccelerationStructure:
        return gfx::ResourceState::AccelerationStructure;
    default:
        FALCOR_UNREACHABLE();
        return gfx::ResourceState::Undefined;
    }
}

void getGFXResourceState(ResourceBindFlags flags, gfx::ResourceState& defaultState, gfx::ResourceStateSet& allowedStates)
{
    defaultState = gfx::ResourceState::General;
    allowedStates = gfx::ResourceStateSet(defaultState);

    // setting up the following flags requires Slang gfx resourece states to have integral type
    if (is_set(flags, ResourceBindFlags::UnorderedAccess))
    {
        allowedStates.add(gfx::ResourceState::UnorderedAccess);
    }

    if (is_set(flags, ResourceBindFlags::ShaderResource))
    {
        allowedStates.add(gfx::ResourceState::ShaderResource);
    }

    if (is_set(flags, ResourceBindFlags::RenderTarget))
    {
        allowedStates.add(gfx::ResourceState::RenderTarget);
    }

    if (is_set(flags, ResourceBindFlags::DepthStencil))
    {
        allowedStates.add(gfx::ResourceState::DepthWrite);
    }

    if (is_set(flags, ResourceBindFlags::Vertex))
    {
        allowedStates.add(gfx::ResourceState::VertexBuffer);
        allowedStates.add(gfx::ResourceState::AccelerationStructureBuildInput);
    }
    if (is_set(flags, ResourceBindFlags::Index))
    {
        allowedStates.add(gfx::ResourceState::IndexBuffer);
        allowedStates.add(gfx::ResourceState::AccelerationStructureBuildInput);
    }
    if (is_set(flags, ResourceBindFlags::IndirectArg))
    {
        allowedStates.add(gfx::ResourceState::IndirectArgument);
    }
    if (is_set(flags, ResourceBindFlags::Constant))
    {
        allowedStates.add(gfx::ResourceState::ConstantBuffer);
    }
    if (is_set(flags, ResourceBindFlags::AccelerationStructure))
    {
        allowedStates.add(gfx::ResourceState::AccelerationStructure);
        allowedStates.add(gfx::ResourceState::ShaderResource);
        allowedStates.add(gfx::ResourceState::UnorderedAccess);
        defaultState = gfx::ResourceState::AccelerationStructure;
    }
    allowedStates.add(gfx::ResourceState::CopyDestination);
    allowedStates.add(gfx::ResourceState::CopySource);
}
} // namespace Falcor
