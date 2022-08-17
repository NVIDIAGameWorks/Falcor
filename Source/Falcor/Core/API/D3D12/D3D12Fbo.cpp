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
#include "Core/API/FBO.h"
#include "Core/API/D3D12/D3D12API.h"

namespace Falcor
{
    struct FboData {};

    Fbo::Fbo()
    {
        mColorAttachments.resize(getMaxColorTargetCount());
    }

    Fbo::~Fbo() = default;

    const Fbo::ApiHandle& Fbo::getApiHandle() const
    {
        FALCOR_UNSUPPORTED_IN_D3D12("Fbo::getApiHandle()");
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
            // TODO: mColorAttachments doesn't contain enough information to fully determine the view dimension. Assume 2D for now.
            auto dimension = rt.arraySize > 1 ? RenderTargetView::Dimension::Texture2DArray : RenderTargetView::Dimension::Texture2D;
            return RenderTargetView::getNullView(dimension);
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
            // TODO: mDepthStencil doesn't contain enough information to fully determine the view dimension.  Assume 2D for now.
            auto dimension = mDepthStencil.arraySize > 1 ? DepthStencilView::Dimension::Texture2DArray : DepthStencilView::Dimension::Texture2D;
            return DepthStencilView::getNullView(dimension);
        }
    }
}
