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
#include "RenderContext.h"
#include "FBO.h"
#include "Texture.h"

namespace Falcor
{
    RenderContext::SharedPtr RenderContext::create(CommandQueueHandle queue)
    {
        return SharedPtr(new RenderContext(queue));
    }

    void RenderContext::clearFbo(const Fbo* pFbo, const float4& color, float depth, uint8_t stencil, FboAttachmentType flags)
    {
        bool hasDepthStencilTexture = pFbo->getDepthStencilTexture() != nullptr;
        ResourceFormat depthStencilFormat = hasDepthStencilTexture ? pFbo->getDepthStencilTexture()->getFormat() : ResourceFormat::Unknown;

        bool clearColor = (flags & FboAttachmentType::Color) != FboAttachmentType::None;
        bool clearDepth = hasDepthStencilTexture && ((flags & FboAttachmentType::Depth) != FboAttachmentType::None);
        bool clearStencil = hasDepthStencilTexture && ((flags & FboAttachmentType::Stencil) != FboAttachmentType::None) && isStencilFormat(depthStencilFormat);

        if (clearColor)
        {
            for (uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); i++)
            {
                if (pFbo->getColorTexture(i))
                {
                    clearRtv(pFbo->getRenderTargetView(i).get(), color);
                }
            }
        }

        if (clearDepth || clearStencil)
        {
            clearDsv(pFbo->getDepthStencilView().get(), depth, stencil, clearDepth, clearStencil);
        }
    }


    void RenderContext::clearTexture(Texture* pTexture, const float4& clearColor)
    {
        assert(pTexture);

        // Check that the format is either Unorm, Snorm or float
        auto format = pTexture->getFormat();
        auto fType = getFormatType(format);
        if (fType == FormatType::Sint || fType == FormatType::Uint || fType == FormatType::Unknown)
        {
            logWarning("RenderContext::clearTexture() - Unsupported texture format " + to_string(format) + ". The texture format must be a normalized or floating-point format");
            return;
        }

        auto bindFlags = pTexture->getBindFlags();
        // Select the right clear based on the texture's binding flags
        if (is_set(bindFlags, Resource::BindFlags::RenderTarget)) clearRtv(pTexture->getRTV().get(), clearColor);
        else if (is_set(bindFlags, Resource::BindFlags::UnorderedAccess)) clearUAV(pTexture->getUAV().get(), clearColor);
        else if (is_set(bindFlags, Resource::BindFlags::DepthStencil))
        {
            if (isStencilFormat(format) && (clearColor.y != 0))
            {
                logWarning("RenderContext::clearTexture() - when clearing a depth-stencil texture the stencil value(clearColor.y) must be 0. Received " + std::to_string(clearColor.y) + ". Forcing stencil to 0");
            }
            clearDsv(pTexture->getDSV().get(), clearColor.r, 0);
        }
        else
        {
            logWarning("Texture::clear() - The texture does not have a bind flag that allows us to clear!");
        }
    }

    bool RenderContext::applyGraphicsVars(GraphicsVars* pVars, RootSignature* pRootSignature)
    {
        bool bindRootSig = (pVars != mpLastBoundGraphicsVars);
        if (pVars->apply(this, bindRootSig, pRootSignature) == false)
        {
            logWarning("RenderContext::prepareForDraw() - applying GraphicsVars failed, most likely because we ran out of descriptors. Flushing the GPU and retrying");
            flush(true);
            if (!pVars->apply(this, bindRootSig, pRootSignature))
            {
                logError("RenderContext::applyGraphicsVars() - applying GraphicsVars failed, most likely because we ran out of descriptors");
                return false;
            }
        }
        return true;
    }

    void RenderContext::flush(bool wait)
    {
        ComputeContext::flush(wait);
        mpLastBoundGraphicsVars = nullptr;
    }
}

