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
#include "RenderContext.h"
#include "FBO.h"
#include "Texture.h"
#include "BlitContext.h"
#include "Utils/Logger.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"

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
        FALCOR_ASSERT(pTexture);

        // Check that the format is either Unorm, Snorm or float
        auto format = pTexture->getFormat();
        auto fType = getFormatType(format);
        if (fType == FormatType::Sint || fType == FormatType::Uint || fType == FormatType::Unknown)
        {
            logWarning("RenderContext::clearTexture() - Unsupported texture format {}. The texture format must be a normalized or floating-point format.", to_string(format));
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
                logWarning("RenderContext::clearTexture() - when clearing a depth-stencil texture the stencil value(clearColor.y) must be 0. Received {}. Forcing stencil to 0.", clearColor.y);
            }
            clearDsv(pTexture->getDSV().get(), clearColor.r, 0);
        }
        else
        {
            logWarning("Texture::clear() - The texture does not have a bind flag that allows us to clear!");
        }
    }

    void RenderContext::flush(bool wait)
    {
        ComputeContext::flush(wait);
        mpLastBoundGraphicsVars = nullptr;
    }

    void RenderContext::blit(const ShaderResourceView::SharedPtr& pSrc, const RenderTargetView::SharedPtr& pDst, uint4 srcRect, uint4 dstRect, Sampler::Filter filter)
    {
        const Sampler::ReductionMode componentsReduction[] = { Sampler::ReductionMode::Standard, Sampler::ReductionMode::Standard, Sampler::ReductionMode::Standard, Sampler::ReductionMode::Standard };
        const float4 componentsTransform[] = { float4(1.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 1.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 1.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 1.0f) };

        blit(pSrc, pDst, srcRect, dstRect, filter, componentsReduction, componentsTransform);
    }

    void RenderContext::blit(const ShaderResourceView::SharedPtr& pSrc, const RenderTargetView::SharedPtr& pDst, uint4 srcRect, uint4 dstRect, Sampler::Filter filter, const Sampler::ReductionMode componentsReduction[4], const float4 componentsTransform[4])
    {
        auto& blitData = getBlitContext();

        // Fetch textures from views.
        FALCOR_ASSERT(pSrc && pDst);
        auto pSrcResource = pSrc->getResource();
        auto pDstResource = pDst->getResource();
        if (pSrcResource->getType() == Resource::Type::Buffer || pDstResource->getType() == Resource::Type::Buffer)
        {
            throw ArgumentError("RenderContext::blit does not support buffers");
        }

        const Texture* pSrcTexture = dynamic_cast<const Texture*>(pSrcResource.get());
        const Texture* pDstTexture = dynamic_cast<const Texture*>(pDstResource.get());
        FALCOR_ASSERT(pSrcTexture != nullptr && pDstTexture != nullptr);

        // Clamp rectangles to the dimensions of the source/dest views.
        const uint32_t srcMipLevel = pSrc->getViewInfo().mostDetailedMip;
        const uint32_t dstMipLevel = pDst->getViewInfo().mostDetailedMip;
        const uint2 srcSize(pSrcTexture->getWidth(srcMipLevel), pSrcTexture->getHeight(srcMipLevel));
        const uint2 dstSize(pDstTexture->getWidth(dstMipLevel), pDstTexture->getHeight(dstMipLevel));

        srcRect.z = std::min(srcRect.z, srcSize.x);
        srcRect.w = std::min(srcRect.w, srcSize.y);
        dstRect.z = std::min(dstRect.z, dstSize.x);
        dstRect.w = std::min(dstRect.w, dstSize.y);

        if (srcRect.x >= srcRect.z || srcRect.y >= srcRect.w ||
            dstRect.x >= dstRect.z || dstRect.y >= dstRect.w)
        {
            logDebug("RenderContext::blit() called with out-of-bounds src/dst rectangle");
            return; // No blit necessary
        }

        // Determine the type of blit.
        const uint32_t sampleCount = pSrcTexture->getSampleCount();
        const bool complexBlit =
            !((componentsReduction[0] == Sampler::ReductionMode::Standard) && (componentsReduction[1] == Sampler::ReductionMode::Standard) && (componentsReduction[2] == Sampler::ReductionMode::Standard) && (componentsReduction[3] == Sampler::ReductionMode::Standard) &&
                (componentsTransform[0] == float4(1.0f, 0.0f, 0.0f, 0.0f)) && (componentsTransform[1] == float4(0.0f, 1.0f, 0.0f, 0.0f)) && (componentsTransform[2] == float4(0.0f, 0.0f, 1.0f, 0.0f)) && (componentsTransform[3] == float4(0.0f, 0.0f, 0.0f, 1.0f)));

        auto isFullView = [](const auto& view, const Texture* tex) {
            const auto& info = view->getViewInfo();
            return info.mostDetailedMip == 0 && info.firstArraySlice == 0 && info.mipCount == tex->getMipCount() && info.arraySize == tex->getArraySize();
        };
        const bool srcFullRect = srcRect.x == 0 && srcRect.y == 0 && srcRect.z == srcSize.x && srcRect.w == srcSize.y;
        const bool dstFullRect = dstRect.x == 0 && dstRect.y == 0 && dstRect.z == dstSize.x && dstRect.w == dstSize.y;

        const bool fullCopy =
            !complexBlit &&
            isFullView(pSrc, pSrcTexture) && srcFullRect &&
            isFullView(pDst, pDstTexture) && dstFullRect &&
            pSrcTexture->compareDesc(pDstTexture);

        // Take fast path to copy the entire resource if possible. This has many requirements;
        // the source/dest must have identical size/format/etc. and the views and rects must cover the full resources.
        if (fullCopy)
        {
            copyResource(pDstResource.get(), pSrcResource.get());
            return;
        }

        // At this point, we have to run a shader to perform the blit.
        // The implementation has some limitations. Check that all requirements are fullfilled.

        // Complex blit doesn't work with multi-sampled textures.
        if (complexBlit && sampleCount > 1) throw RuntimeError("RenderContext::blit() does not support sample count > 1 for complex blit");

        // Validate source format. Only single-sampled basic blit handles integer source format.
        // All variants support casting to integer destination format.
        if (isIntegerFormat(pSrcTexture->getFormat()))
        {
            if (sampleCount > 1) throw RuntimeError("RenderContext::blit() requires non-integer source format for multi-sampled textures");
            else if (complexBlit) throw RuntimeError("RenderContext::blit() requires non-integer source format for complex blit");
        }

        // Blit does not support texture arrays or mip maps.
        if (!(pSrc->getViewInfo().arraySize == 1 && pSrc->getViewInfo().mipCount == 1) ||
            !(pDst->getViewInfo().arraySize == 1 && pDst->getViewInfo().mipCount == 1))
        {
            throw RuntimeError("RenderContext::blit() does not support texture arrays or mip maps");
        }

        // Configure program.
        blitData.pPass->addDefine("SAMPLE_COUNT", std::to_string(sampleCount));
        blitData.pPass->addDefine("COMPLEX_BLIT", complexBlit ? "1" : "0");
        blitData.pPass->addDefine("SRC_INT", isIntegerFormat(pSrcTexture->getFormat()) ? "1" : "0");
        blitData.pPass->addDefine("DST_INT", isIntegerFormat(pDstTexture->getFormat()) ? "1" : "0");

        if (complexBlit)
        {
            FALCOR_ASSERT(sampleCount <= 1);

            Sampler::SharedPtr usedSampler[4];
            for (uint32_t i = 0; i < 4; i++)
            {
                FALCOR_ASSERT(componentsReduction[i] != Sampler::ReductionMode::Comparison);        // Comparison mode not supported.

                if (componentsReduction[i] == Sampler::ReductionMode::Min) usedSampler[i] = (filter == Sampler::Filter::Linear) ? blitData.pLinearMinSampler : blitData.pPointMinSampler;
                else if (componentsReduction[i] == Sampler::ReductionMode::Max) usedSampler[i] = (filter == Sampler::Filter::Linear) ? blitData.pLinearMaxSampler : blitData.pPointMaxSampler;
                else usedSampler[i] = (filter == Sampler::Filter::Linear) ? blitData.pLinearSampler : blitData.pPointSampler;
            }

            blitData.pPass->getVars()->setSampler("gSamplerR", usedSampler[0]);
            blitData.pPass->getVars()->setSampler("gSamplerG", usedSampler[1]);
            blitData.pPass->getVars()->setSampler("gSamplerB", usedSampler[2]);
            blitData.pPass->getVars()->setSampler("gSamplerA", usedSampler[3]);

            // Parameters for complex blit
            for (uint32_t i = 0; i < 4; i++)
            {
                if (blitData.prevComponentsTransform[i] != componentsTransform[i])
                {
                    blitData.pBlitParamsBuffer->setVariable(blitData.compTransVarOffset[i], componentsTransform[i]);
                    blitData.prevComponentsTransform[i] = componentsTransform[i];
                }
            }
        }
        else
        {
            blitData.pPass->getVars()->setSampler("gSampler", (filter == Sampler::Filter::Linear) ? blitData.pLinearSampler : blitData.pPointSampler);
        }

        float2 srcRectOffset(0.0f);
        float2 srcRectScale(1.0f);
        if (!srcFullRect)
        {
            srcRectOffset = float2(srcRect.x, srcRect.y) / float2(srcSize);
            srcRectScale = float2(srcRect.z - srcRect.x, srcRect.w - srcRect.y) / float2(srcSize);
        }

        GraphicsState::Viewport dstViewport(0.0f, 0.0f, (float)dstSize.x, (float)dstSize.y, 0.0f, 1.0f);
        if (!dstFullRect)
        {
            dstViewport = GraphicsState::Viewport((float)dstRect.x, (float)dstRect.y, (float)(dstRect.z - dstRect.x), (float)(dstRect.w - dstRect.y), 0.0f, 1.0f);
        }

        // Update buffer/state
        if (srcRectOffset != blitData.prevSrcRectOffset)
        {
            blitData.pBlitParamsBuffer->setVariable(blitData.offsetVarOffset, srcRectOffset);
            blitData.prevSrcRectOffset = srcRectOffset;
        }

        if (srcRectScale != blitData.prevSrcReftScale)
        {
            blitData.pBlitParamsBuffer->setVariable(blitData.scaleVarOffset, srcRectScale);
            blitData.prevSrcReftScale = srcRectScale;
        }

        Texture::SharedPtr pSharedTex = std::static_pointer_cast<Texture>(pDstResource);
        blitData.pFbo->attachColorTarget(pSharedTex, 0, pDst->getViewInfo().mostDetailedMip, pDst->getViewInfo().firstArraySlice, pDst->getViewInfo().arraySize);
        blitData.pPass->getVars()->setSrv(blitData.texBindLoc, pSrc);
        blitData.pPass->getState()->setViewport(0, dstViewport);
        blitData.pPass->execute(this, blitData.pFbo, false);

        // Release the resources we bound
        blitData.pPass->getVars()->setSrv(blitData.texBindLoc, nullptr);
    }
}
