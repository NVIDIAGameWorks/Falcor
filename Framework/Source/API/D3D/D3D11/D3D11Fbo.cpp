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
#include "API/FBO.h"
#include "API/Texture.h"
#include "glm/gtc/type_ptr.hpp"
#include <map>

namespace Falcor
{
    struct FboData
    {
        std::vector<ID3D11RenderTargetViewPtr> pRtv;
        ID3D11DepthStencilViewPtr pDsv;
    };

    Fbo::Fbo(bool initApiHandle)
    {
        mColorAttachments.resize(getMaxColorTargetCount());
        FboData* pFboData = new FboData;
        pFboData->pRtv.resize(getMaxColorTargetCount());
        mpPrivateData = pFboData;

        mApiHandle = -1;
    }

    Fbo::~Fbo()
    {
        delete (FboData*)mpPrivateData;
        mpPrivateData = nullptr;
    }

    uint32_t Fbo::getApiHandle() const
    {
        UNSUPPORTED_IN_D3D11("CFbo Api Handle");
        return mApiHandle;
    }

    uint32_t Fbo::getMaxColorTargetCount()
    {
        return D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT;
    }

    void Fbo::applyColorAttachment(uint32_t rtIndex)
    {
        FboData* pData = (FboData*)mpPrivateData;
        pData->pRtv[rtIndex] = nullptr;
        const auto pTexture = mColorAttachments[rtIndex].pTexture;
        if(pTexture)
        {
            ID3D11Resource* pResource = pTexture->getApiHandle();
            D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
            initializeRtvDesc(pTexture.get(), mColorAttachments[rtIndex].mipLevel, mColorAttachments[rtIndex].arraySlice, rtvDesc);
            ID3D11RenderTargetViewPtr pRTV;
            d3d_call(getD3D11Device()->CreateRenderTargetView(pResource, &rtvDesc, &pData->pRtv[rtIndex]));
        }
    }

    void Fbo::applyDepthAttachment()
    {
        FboData* pData = (FboData*)mpPrivateData;

        pData->pDsv = nullptr;
        const auto pDepth = mDepthStencil.pTexture;
        if(pDepth)
        {
            ID3D11Resource* pResource = pDepth->getApiHandle();
            D3D11_DEPTH_STENCIL_VIEW_DESC DsvDesc;
            initializeDsvDesc(pDepth.get(), mDepthStencil.mipLevel, mDepthStencil.arraySlice, DsvDesc);
            d3d_call(getD3D11Device()->CreateDepthStencilView(pResource, &DsvDesc, &pData->pDsv));
        }
    }

    bool Fbo::checkStatus() const
    {
        FboData* pData = (FboData*)mpPrivateData;

        if(mIsDirty)
        {
            mIsDirty = false;
            return calcAndValidateProperties();
        }

        return true;
    }
    
    void Fbo::clearColorTarget(uint32_t rtIndex, const glm::vec4& color) const
    {
        FboData* pFboData = (FboData*)mpPrivateData;
        const auto pTexture = mColorAttachments[rtIndex].pTexture;
        if(pTexture == nullptr)
        {
            logWarning("Trying to clear a color render-target, but the texture does not exist in CFbo.");
            return;
        }
        if(checkStatus())
        {
            getD3D11ImmediateContext()->ClearRenderTargetView(pFboData->pRtv[rtIndex], glm::value_ptr(color));
        }
    }

    void Fbo::clearColorTarget(uint32_t rtIndex, const glm::uvec4& color) const
    {
        UNSUPPORTED_IN_D3D11("unsigned int version of ClearColorTarget()");
    }

    void Fbo::clearColorTarget(uint32_t rtIndex, const glm::ivec4& color) const
    {
        UNSUPPORTED_IN_D3D11("int version of ClearColorTarget()");
    }

    void Fbo::captureToFile(uint32_t rtIndex, const std::string& filename, Bitmap::FileFormat fileFormat)
    {
        UNSUPPORTED_IN_D3D11("captureToPng()");
    }

    void Fbo::clearDepthStencil(float depth, uint8_t stencil, bool clearDepth, bool clearStencil) const
    {
        FboData* pFboData = (FboData*)mpPrivateData;
        const auto pTexture = mDepthStencil.pTexture;
        if(pTexture == nullptr)
        {
            logWarning("Trying to clear a depth buffer, but the texture does not exist in CFbo.");
            return;
        }

        if(!clearDepth && !clearStencil)
        {
            logWarning("Trying to clear a depth buffer, but both bClearDepth and bClearStencil are false.");
            return;
        }

        if(checkStatus())
        {
            uint32_t clearFlags = 0;
            clearFlags |= clearDepth ? D3D11_CLEAR_DEPTH : 0;
            clearFlags |= clearStencil ? D3D11_CLEAR_STENCIL : 0;
            getD3D11ImmediateContext()->ClearDepthStencilView(pFboData->pDsv, clearFlags, depth, stencil);
        }
    }
}

