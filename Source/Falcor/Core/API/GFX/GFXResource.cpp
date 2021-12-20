/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    void Resource::apiSetName()
    {
    }

    SharedResourceApiHandle Resource::getSharedApiHandle() const
    {
        return SharedResourceApiHandle();
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
            return gfx::ResourceState::DepthRead;
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
            return gfx::ResourceState::ShaderResource;
        case Resource::State::NonPixelShader:
            return gfx::ResourceState::ShaderResource;
        case Resource::State::AccelerationStructure:
            return gfx::ResourceState::AccelerationStructure;
        default:
            should_not_get_here();
            return gfx::ResourceState::Undefined;
        }
    }
}
