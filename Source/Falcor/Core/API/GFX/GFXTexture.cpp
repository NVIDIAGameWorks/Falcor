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
#include "Core/API/Texture.h"
#include "Core/API/Device.h"
#include "Core/API/Formats.h"

#include "Core/API/GFX/GFXFormats.h"

namespace Falcor
{

    gfx::IResource::Type getResourceType(Texture::Type type)
    {
        switch (type)
        {
        case Texture::Type::Texture1D:
            return gfx::IResource::Type::Texture1D;

        case Texture::Type::Texture2D:
        case Texture::Type::Texture2DMultisample:
            return gfx::IResource::Type::Texture2D;
        case Texture::Type::TextureCube:
            return gfx::IResource::Type::TextureCube;

        case Texture::Type::Texture3D:
            return gfx::IResource::Type::TextureCube;
        default:
            should_not_get_here();
            return gfx::IResource::Type::Unknown;
        }
    }

    void getGfxResourceState(Resource::BindFlags flags, gfx::ResourceState &defaultState, gfx::ResourceStateSet &allowedStates)
    {
        defaultState = gfx::ResourceState::Undefined; // TODO: check what default state should be
        allowedStates = gfx::ResourceStateSet(defaultState);

        // setting up the following flags requires Slang gfx resourece states to have integral type

        bool uavRequired = is_set(flags, Resource::BindFlags::UnorderedAccess);

        if (uavRequired)
        {
            allowedStates.add(gfx::ResourceState::UnorderedAccess);
        }

        if (is_set(flags, Resource::BindFlags::ShaderResource))
        {
            allowedStates.add(gfx::ResourceState::ShaderResource);
        }

        if (is_set(flags, Resource::BindFlags::RenderTarget))
        {
            allowedStates.add(gfx::ResourceState::RenderTarget);
        }

    }

    uint64_t Texture::getTextureSizeInBytes() const
    {
        // get allocation info for resource description
        size_t outSizeBytes = 0, outAlignment = 0;

        Slang::ComPtr<gfx::IDevice> pDevicePtr = gpDevice->getApiHandle();
        gfx::ITextureResource* textureResource = static_cast<gfx::ITextureResource*>(getApiHandle().get());
        assert(textureResource);

        gfx::ITextureResource::Desc *desc = textureResource->getDesc();

        gpDevice->getApiHandle()->getTextureAllocationInfo(*desc, &outSizeBytes, &outAlignment);
        assert(outSizeBytes > 0);

        return outSizeBytes;

    }

    void Texture::apiInit(const void* pData, bool autoGenMips)
    {
        // create resource description
        gfx::ITextureResource::Desc desc;

        // base description

        // type
        desc.type = getResourceType(mType); // same as resource dimension in D3D12

        // default state and allowed states
        getGfxResourceState(mBindFlags, desc.defaultState, desc.allowedStates);

        // cpu access flags

        // TODO: check when cpuAccessFlags are required, might be buffer-only feature
        desc.cpuAccessFlags = gfx::AccessFlag::Write | gfx::AccessFlag::Read; // conservatively assume read/write access

        // texture resource specific description attributes
               
        // size
        desc.size.width = align_to(getFormatWidthCompressionRatio(mFormat), mWidth);
        desc.size.height = align_to(getFormatHeightCompressionRatio(mFormat), mHeight);
        desc.size.depth = mDepth; // relevant for Texture3D

        // array size
        if (mType == Texture::Type::TextureCube)
        {
            desc.arraySize = mArraySize * 6;
        }
        else
        {
            desc.arraySize = mArraySize;
        }

        // mip map levels
        desc.numMipLevels = mMipLevels;

        // format

        // TODO: check typeless formats in Slang-GFX
        // compare D3D12Texture.cpp L114
        desc.format = getGFXFormat(mFormat); // lookup can result in Unknown / unsupported format

        // sample description
        desc.sampleDesc.numSamples = mSampleCount;
        desc.sampleDesc.quality = 0;

        // clear value
        gfx::ClearValue clearValue;
        if ((mBindFlags & (Texture::BindFlags::RenderTarget | Texture::BindFlags::DepthStencil)) != Texture::BindFlags::None)
        {
            if ((mBindFlags & Texture::BindFlags::DepthStencil) != Texture::BindFlags::None)
            {
                clearValue.depthStencil.depth = 1.0f;
            }
        }
        desc.optimalClearValue = clearValue;

        // shared resource
        if (is_set(mBindFlags, Resource::BindFlags::Shared))
        {
            desc.isShared = true;
        }

        // validate description
        assert(desc.size.width > 0 && desc.size.height > 0);
        assert(desc.numMipLevels > 0 && desc.size.depth > 0 && desc.arraySize > 0 && desc.sampleDesc.numSamples > 0);

        // create resource
        Slang::ComPtr<gfx::ITextureResource> textureResource =
            gpDevice->getApiHandle()->createTextureResource(desc, nullptr);

        assert(textureResource);

        mApiHandle = textureResource;

        // upload init data through texture class
        if (pData)
        {
            uploadInitData(pData, autoGenMips);
        }
    }

    Texture::~Texture()
    {
    }
}
