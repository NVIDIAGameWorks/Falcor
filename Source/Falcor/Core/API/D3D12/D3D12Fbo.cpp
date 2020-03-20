/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#include "stdafx.h"
#include "Core/API/FBO.h"

namespace Falcor
{
    template<typename ViewType>
    ViewType getViewDimension(Resource::Type type, bool isArray);

    template<>
    D3D12_RTV_DIMENSION getViewDimension<D3D12_RTV_DIMENSION>(Resource::Type type, bool isTextureArray)
    {
        switch(type)
        {
        case Resource::Type::Texture1D:
            return (isTextureArray) ? D3D12_RTV_DIMENSION_TEXTURE1DARRAY : D3D12_RTV_DIMENSION_TEXTURE1D;
        case Resource::Type::Texture2D:
            return (isTextureArray) ? D3D12_RTV_DIMENSION_TEXTURE2DARRAY : D3D12_RTV_DIMENSION_TEXTURE2D;
        case Resource::Type::Texture3D:
            assert(isTextureArray == false);
            return D3D12_RTV_DIMENSION_TEXTURE3D;
        case Resource::Type::Texture2DMultisample:
            return (isTextureArray) ? D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY : D3D12_RTV_DIMENSION_TEXTURE2DMS;
        case Resource::Type::TextureCube:
            return D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        default:
            should_not_get_here();
            return D3D12_RTV_DIMENSION_UNKNOWN;
        }
    }

    template<>
    D3D12_DSV_DIMENSION getViewDimension<D3D12_DSV_DIMENSION>(Resource::Type type, bool isTextureArray)
    {
        switch(type)
        {
        case Resource::Type::Texture1D:
            return (isTextureArray) ? D3D12_DSV_DIMENSION_TEXTURE1DARRAY : D3D12_DSV_DIMENSION_TEXTURE1D;
        case Resource::Type::Texture2D:
            return (isTextureArray) ? D3D12_DSV_DIMENSION_TEXTURE2DARRAY : D3D12_DSV_DIMENSION_TEXTURE2D;
        case Resource::Type::Texture2DMultisample:
            return (isTextureArray) ? D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D12_DSV_DIMENSION_TEXTURE2DMS;
        case Resource::Type::TextureCube:
            return D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        default:
            should_not_get_here();
            return D3D12_DSV_DIMENSION_UNKNOWN;
        }
    }

    Fbo::Fbo()
    {
        mColorAttachments.resize(getMaxColorTargetCount());
    }

    Fbo::~Fbo() = default;

    const Fbo::ApiHandle& Fbo::getApiHandle() const
    {
        UNSUPPORTED_IN_D3D12("Fbo::getApiHandle()");
        return mApiHandle;
    }
    
    uint32_t Fbo::getMaxColorTargetCount()
    {
        return D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;
    }

    void Fbo::applyColorAttachment(uint32_t rtIndex)
    {
    }

    void Fbo::applyDepthAttachment()
    {
    }

    void Fbo::initApiHandle() const {}

    RenderTargetView::SharedPtr Fbo::getRenderTargetView(uint32_t rtIndex) const
    {
        const auto& rt = mColorAttachments[rtIndex];
        if(rt.pTexture)
        {
            return rt.pTexture->getRTV(rt.mipLevel, rt.firstArraySlice, rt.arraySize);
        }
        else
        {
            return RenderTargetView::getNullView();
        }
    }

    DepthStencilView::SharedPtr Fbo::getDepthStencilView() const
    {
        if(mDepthStencil.pTexture)
        {
            return mDepthStencil.pTexture->getDSV(mDepthStencil.mipLevel, mDepthStencil.firstArraySlice, mDepthStencil.arraySize);
        }
        else
        {
            return DepthStencilView::getNullView();
        }
    }
}
