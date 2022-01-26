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
#include "Core/API/FBO.h"

namespace Falcor
{
    struct FboPrivateData
    {
        bool mHandleDirty = true;
    };

    namespace
    {
        FboPrivateData* getFboPrivateData(void* privateData)
        {
            return static_cast<FboPrivateData*>(privateData);
        }

        void releaseFboHandleIfEmpty(FboHandle& apiHandle, Fbo::Attachment& depthStencil, std::vector<Fbo::Attachment>& colorAttachments)
        {
            if (depthStencil.pTexture)
            {
                return;
            }
            for (auto& attachment : colorAttachments)
            {
                if (attachment.pTexture)
                {
                    return;
                }
            }
            apiHandle = nullptr;
        }

    }

    Fbo::Fbo()
    {
        mpPrivateData = new FboPrivateData();
        mColorAttachments.resize(getMaxColorTargetCount());
    }

    Fbo::~Fbo()
    {
        delete getFboPrivateData(mpPrivateData);
    }

    const Fbo::ApiHandle& Fbo::getApiHandle() const
    {
        if (getFboPrivateData(mpPrivateData)->mHandleDirty)
        {
            initApiHandle();
        }
        return mApiHandle;
    }

    uint32_t Fbo::getMaxColorTargetCount()
    {
        return 8;
    }

    void Fbo::applyColorAttachment(uint32_t rtIndex)
    {
        getFboPrivateData(mpPrivateData)->mHandleDirty = true;
        releaseFboHandleIfEmpty(mApiHandle, mDepthStencil, mColorAttachments);
    }

    void Fbo::applyDepthAttachment()
    {
        getFboPrivateData(mpPrivateData)->mHandleDirty = true;
        releaseFboHandleIfEmpty(mApiHandle, mDepthStencil, mColorAttachments);
    }

    void Fbo::initApiHandle() const
    {
        getFboPrivateData(mpPrivateData)->mHandleDirty = false;

        gfx::IFramebufferLayout::Desc layoutDesc = {};
        std::vector<gfx::IFramebufferLayout::AttachmentLayout> attachmentLayouts;
        gfx::IFramebufferLayout::AttachmentLayout depthAttachmentLayout = {};
        gfx::IFramebuffer::Desc desc = {};
        if (mDepthStencil.pTexture)
        {
            auto texture = static_cast<gfx::ITextureResource*>(mDepthStencil.pTexture->getApiHandle().get());
            depthAttachmentLayout.format = texture->getDesc()->format;
            depthAttachmentLayout.sampleCount = texture->getDesc()->sampleDesc.numSamples;
            layoutDesc.depthStencil = &depthAttachmentLayout;
        }
        desc.depthStencilView = getDepthStencilView()->getApiHandle();
        desc.renderTargetCount = 0;
        std::vector<gfx::IResourceView*> renderTargetViews;
        for (uint32_t i = 0; i < static_cast<uint32_t>(mColorAttachments.size()); i++)
        {
            if (mColorAttachments[i].pTexture)
            {
                gfx::IFramebufferLayout::AttachmentLayout renderAttachmentLayout = {};
                auto texture = static_cast<gfx::ITextureResource*>(mColorAttachments[i].pTexture->getApiHandle().get());
                renderAttachmentLayout.format = texture->getDesc()->format;
                renderAttachmentLayout.sampleCount = texture->getDesc()->sampleDesc.numSamples;
                attachmentLayouts.push_back(renderAttachmentLayout);
                renderTargetViews.push_back(getRenderTargetView(i)->getApiHandle());
                desc.renderTargetCount = i + 1;
            }
        }
        desc.renderTargetViews = renderTargetViews.data();
        layoutDesc.renderTargetCount = desc.renderTargetCount;
        layoutDesc.renderTargets = attachmentLayouts.data();

        Slang::ComPtr<gfx::IFramebufferLayout> fboLayout;
        FALCOR_ASSERT(SLANG_SUCCEEDED(gpDevice->getApiHandle()->createFramebufferLayout(layoutDesc, fboLayout.writeRef())));

        desc.layout = fboLayout.get();
        FALCOR_ASSERT(SLANG_SUCCEEDED(gpDevice->getApiHandle()->createFramebuffer(desc, mApiHandle.writeRef())));
    }

    RenderTargetView::SharedPtr Fbo::getRenderTargetView(uint32_t rtIndex) const
    {
        const auto& rt = mColorAttachments[rtIndex];
        if (rt.pTexture)
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
        if (mDepthStencil.pTexture)
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
