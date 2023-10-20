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
#include "RenderContext.h"
#include "FBO.h"
#include "Texture.h"
#include "BlitContext.h"
#include "RtAccelerationStructure.h"
#include "GFXHelpers.h"
#include "GFXAPI.h"
#include "Core/State/GraphicsState.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <cstddef> // for offsetof

namespace Falcor
{
namespace
{
constexpr void checkViewportScissorBinaryCompatiblity()
{
    static_assert(offsetof(gfx::Viewport, originX) == offsetof(GraphicsState::Viewport, originX));
    static_assert(offsetof(gfx::Viewport, originY) == offsetof(GraphicsState::Viewport, originY));
    static_assert(offsetof(gfx::Viewport, extentX) == offsetof(GraphicsState::Viewport, width));
    static_assert(offsetof(gfx::Viewport, extentY) == offsetof(GraphicsState::Viewport, height));
    static_assert(offsetof(gfx::Viewport, minZ) == offsetof(GraphicsState::Viewport, minDepth));
    static_assert(offsetof(gfx::Viewport, maxZ) == offsetof(GraphicsState::Viewport, maxDepth));

    static_assert(offsetof(gfx::ScissorRect, minX) == offsetof(GraphicsState::Scissor, left));
    static_assert(offsetof(gfx::ScissorRect, minY) == offsetof(GraphicsState::Scissor, top));
    static_assert(offsetof(gfx::ScissorRect, maxX) == offsetof(GraphicsState::Scissor, right));
    static_assert(offsetof(gfx::ScissorRect, maxY) == offsetof(GraphicsState::Scissor, bottom));
}

void ensureFboAttachmentResourceStates(RenderContext* pCtx, Fbo* pFbo)
{
    if (pFbo)
    {
        for (uint32_t i = 0; i < pFbo->getMaxColorTargetCount(); i++)
        {
            auto pTexture = pFbo->getColorTexture(i);
            if (pTexture)
            {
                auto pRTV = pFbo->getRenderTargetView(i);
                pCtx->resourceBarrier(pTexture.get(), Resource::State::RenderTarget, &pRTV->getViewInfo());
            }
        }

        auto& pTexture = pFbo->getDepthStencilTexture();
        if (pTexture)
        {
            if (pTexture)
            {
                auto pDSV = pFbo->getDepthStencilView();
                pCtx->resourceBarrier(pTexture.get(), Resource::State::DepthStencil, &pDSV->getViewInfo());
            }
        }
    }
}

gfx::PrimitiveTopology getGFXPrimitiveTopology(Vao::Topology topology)
{
    switch (topology)
    {
    case Vao::Topology::Undefined:
        return gfx::PrimitiveTopology::TriangleList;
    case Vao::Topology::PointList:
        return gfx::PrimitiveTopology::PointList;
    case Vao::Topology::LineList:
        return gfx::PrimitiveTopology::LineList;
    case Vao::Topology::LineStrip:
        return gfx::PrimitiveTopology::LineStrip;
    case Vao::Topology::TriangleList:
        return gfx::PrimitiveTopology::TriangleList;
    case Vao::Topology::TriangleStrip:
        return gfx::PrimitiveTopology::TriangleStrip;
    default:
        FALCOR_UNREACHABLE();
        return gfx::PrimitiveTopology::TriangleList;
    }
}

gfx::AccelerationStructureCopyMode getGFXAcclerationStructureCopyMode(RenderContext::RtAccelerationStructureCopyMode mode)
{
    switch (mode)
    {
    case RenderContext::RtAccelerationStructureCopyMode::Clone:
        return gfx::AccelerationStructureCopyMode::Clone;
    case RenderContext::RtAccelerationStructureCopyMode::Compact:
        return gfx::AccelerationStructureCopyMode::Compact;
    default:
        FALCOR_UNREACHABLE();
        return gfx::AccelerationStructureCopyMode::Clone;
    }
}
} // namespace

RenderContext::RenderContext(Device* pDevice, gfx::ICommandQueue* pQueue) : ComputeContext(pDevice, pQueue)
{
    mpBlitContext = std::make_unique<BlitContext>(pDevice);
}

RenderContext::~RenderContext() {}

void RenderContext::clearFbo(const Fbo* pFbo, const float4& color, float depth, uint8_t stencil, FboAttachmentType flags)
{
    bool hasDepthStencilTexture = pFbo->getDepthStencilTexture() != nullptr;
    ResourceFormat depthStencilFormat = hasDepthStencilTexture ? pFbo->getDepthStencilTexture()->getFormat() : ResourceFormat::Unknown;

    bool clearColor = (flags & FboAttachmentType::Color) != FboAttachmentType::None;
    bool clearDepth = hasDepthStencilTexture && ((flags & FboAttachmentType::Depth) != FboAttachmentType::None);
    bool clearStencil =
        hasDepthStencilTexture && ((flags & FboAttachmentType::Stencil) != FboAttachmentType::None) && isStencilFormat(depthStencilFormat);

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
        logWarning(
            "RenderContext::clearTexture() - Unsupported texture format {}. The texture format must be a normalized or floating-point "
            "format.",
            to_string(format)
        );
        return;
    }

    auto bindFlags = pTexture->getBindFlags();
    // Select the right clear based on the texture's binding flags
    if (is_set(bindFlags, ResourceBindFlags::RenderTarget))
        clearRtv(pTexture->getRTV().get(), clearColor);
    else if (is_set(bindFlags, ResourceBindFlags::UnorderedAccess))
        clearUAV(pTexture->getUAV().get(), clearColor);
    else if (is_set(bindFlags, ResourceBindFlags::DepthStencil))
    {
        if (isStencilFormat(format) && (clearColor.y != 0))
        {
            logWarning(
                "RenderContext::clearTexture() - when clearing a depth-stencil texture the stencil value(clearColor.y) must be 0. Received "
                "{}. Forcing stencil to 0.",
                clearColor.y
            );
        }
        clearDsv(pTexture->getDSV().get(), clearColor.r, 0);
    }
    else
    {
        logWarning("Texture::clear() - The texture does not have a bind flag that allows us to clear!");
    }
}

void RenderContext::submit(bool wait)
{
    ComputeContext::submit(wait);
    mpLastBoundGraphicsVars = nullptr;
}

void RenderContext::blit(
    const ref<ShaderResourceView>& pSrc,
    const ref<RenderTargetView>& pDst,
    uint4 srcRect,
    uint4 dstRect,
    TextureFilteringMode filter
)
{
    const TextureReductionMode componentsReduction[] = {
        TextureReductionMode::Standard,
        TextureReductionMode::Standard,
        TextureReductionMode::Standard,
        TextureReductionMode::Standard,
    };
    const float4 componentsTransform[] = {
        float4(1.0f, 0.0f, 0.0f, 0.0f),
        float4(0.0f, 1.0f, 0.0f, 0.0f),
        float4(0.0f, 0.0f, 1.0f, 0.0f),
        float4(0.0f, 0.0f, 0.0f, 1.0f),
    };

    blit(pSrc, pDst, srcRect, dstRect, filter, componentsReduction, componentsTransform);
}

void RenderContext::blit(
    const ref<ShaderResourceView>& pSrc,
    const ref<RenderTargetView>& pDst,
    uint4 srcRect,
    uint4 dstRect,
    TextureFilteringMode filter,
    const TextureReductionMode componentsReduction[4],
    const float4 componentsTransform[4]
)
{
    FALCOR_ASSERT(mpBlitContext);
    auto& blitCtx = *mpBlitContext;

    // Fetch textures from views.
    FALCOR_ASSERT(pSrc && pDst);
    auto pSrcResource = pSrc->getResource();
    auto pDstResource = pDst->getResource();
    FALCOR_ASSERT(pSrcResource && pDstResource);
    if (pSrcResource->getType() == Resource::Type::Buffer || pDstResource->getType() == Resource::Type::Buffer)
    {
        FALCOR_THROW("RenderContext::blit does not support buffers");
    }

    const Texture* pSrcTexture = dynamic_cast<const Texture*>(pSrcResource);
    const Texture* pDstTexture = dynamic_cast<const Texture*>(pDstResource);
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

    if (srcRect.x >= srcRect.z || srcRect.y >= srcRect.w || dstRect.x >= dstRect.z || dstRect.y >= dstRect.w)
    {
        logDebug("RenderContext::blit() called with out-of-bounds src/dst rectangle");
        return; // No blit necessary
    }

    // Determine the type of blit.
    const uint32_t sampleCount = pSrcTexture->getSampleCount();
    const bool complexBlit =
        !((componentsReduction[0] == TextureReductionMode::Standard) && (componentsReduction[1] == TextureReductionMode::Standard) &&
          (componentsReduction[2] == TextureReductionMode::Standard) && (componentsReduction[3] == TextureReductionMode::Standard) &&
          all(componentsTransform[0] == float4(1.0f, 0.0f, 0.0f, 0.0f)) && all(componentsTransform[1] == float4(0.0f, 1.0f, 0.0f, 0.0f)) &&
          all(componentsTransform[2] == float4(0.0f, 0.0f, 1.0f, 0.0f)) && all(componentsTransform[3] == float4(0.0f, 0.0f, 0.0f, 1.0f)));

    auto isFullView = [](const auto& view, const Texture* tex)
    {
        const auto& info = view->getViewInfo();
        return info.mostDetailedMip == 0 && info.firstArraySlice == 0 && info.mipCount == tex->getMipCount() &&
               info.arraySize == tex->getArraySize();
    };
    const bool srcFullRect = srcRect.x == 0 && srcRect.y == 0 && srcRect.z == srcSize.x && srcRect.w == srcSize.y;
    const bool dstFullRect = dstRect.x == 0 && dstRect.y == 0 && dstRect.z == dstSize.x && dstRect.w == dstSize.y;

    const bool fullCopy = !complexBlit && isFullView(pSrc, pSrcTexture) && srcFullRect && isFullView(pDst, pDstTexture) && dstFullRect &&
                          pSrcTexture->compareDesc(pDstTexture);

    // Take fast path to copy the entire resource if possible. This has many requirements;
    // the source/dest must have identical size/format/etc. and the views and rects must cover the full resources.
    if (fullCopy)
    {
        copyResource(pDstResource, pSrcResource);
        return;
    }

    // At this point, we have to run a shader to perform the blit.
    // The implementation has some limitations. Check that all requirements are fullfilled.

    // Complex blit doesn't work with multi-sampled textures.
    if (complexBlit && sampleCount > 1)
        FALCOR_THROW("RenderContext::blit() does not support sample count > 1 for complex blit");

    // Validate source format. Only single-sampled basic blit handles integer source format.
    // All variants support casting to integer destination format.
    if (isIntegerFormat(pSrcTexture->getFormat()))
    {
        if (sampleCount > 1)
            FALCOR_THROW("RenderContext::blit() requires non-integer source format for multi-sampled textures");
        else if (complexBlit)
            FALCOR_THROW("RenderContext::blit() requires non-integer source format for complex blit");
    }

    // Blit does not support texture arrays or mip maps.
    if (!(pSrc->getViewInfo().arraySize == 1 && pSrc->getViewInfo().mipCount == 1) ||
        !(pDst->getViewInfo().arraySize == 1 && pDst->getViewInfo().mipCount == 1))
    {
        FALCOR_THROW("RenderContext::blit() does not support texture arrays or mip maps");
    }

    // Configure program.
    blitCtx.pPass->addDefine("SAMPLE_COUNT", std::to_string(sampleCount));
    blitCtx.pPass->addDefine("COMPLEX_BLIT", complexBlit ? "1" : "0");
    blitCtx.pPass->addDefine("SRC_INT", isIntegerFormat(pSrcTexture->getFormat()) ? "1" : "0");
    blitCtx.pPass->addDefine("DST_INT", isIntegerFormat(pDstTexture->getFormat()) ? "1" : "0");

    if (complexBlit)
    {
        FALCOR_ASSERT(sampleCount <= 1);

        ref<Sampler> usedSampler[4];
        for (uint32_t i = 0; i < 4; i++)
        {
            FALCOR_ASSERT(componentsReduction[i] != TextureReductionMode::Comparison); // Comparison mode not supported.

            if (componentsReduction[i] == TextureReductionMode::Min)
                usedSampler[i] = (filter == TextureFilteringMode::Linear) ? blitCtx.pLinearMinSampler : blitCtx.pPointMinSampler;
            else if (componentsReduction[i] == TextureReductionMode::Max)
                usedSampler[i] = (filter == TextureFilteringMode::Linear) ? blitCtx.pLinearMaxSampler : blitCtx.pPointMaxSampler;
            else
                usedSampler[i] = (filter == TextureFilteringMode::Linear) ? blitCtx.pLinearSampler : blitCtx.pPointSampler;
        }

        blitCtx.pPass->getVars()->setSampler("gSamplerR", usedSampler[0]);
        blitCtx.pPass->getVars()->setSampler("gSamplerG", usedSampler[1]);
        blitCtx.pPass->getVars()->setSampler("gSamplerB", usedSampler[2]);
        blitCtx.pPass->getVars()->setSampler("gSamplerA", usedSampler[3]);

        // Parameters for complex blit
        for (uint32_t i = 0; i < 4; i++)
        {
            if (any(blitCtx.prevComponentsTransform[i] != componentsTransform[i]))
            {
                blitCtx.pBlitParamsBuffer->setVariable(blitCtx.compTransVarOffset[i], componentsTransform[i]);
                blitCtx.prevComponentsTransform[i] = componentsTransform[i];
            }
        }
    }
    else
    {
        blitCtx.pPass->getVars()->setSampler(
            "gSampler", (filter == TextureFilteringMode::Linear) ? blitCtx.pLinearSampler : blitCtx.pPointSampler
        );
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
        dstViewport = GraphicsState::Viewport(
            (float)dstRect.x, (float)dstRect.y, (float)(dstRect.z - dstRect.x), (float)(dstRect.w - dstRect.y), 0.0f, 1.0f
        );
    }

    // Update buffer/state
    if (any(srcRectOffset != blitCtx.prevSrcRectOffset))
    {
        blitCtx.pBlitParamsBuffer->setVariable(blitCtx.offsetVarOffset, srcRectOffset);
        blitCtx.prevSrcRectOffset = srcRectOffset;
    }

    if (any(srcRectScale != blitCtx.prevSrcReftScale))
    {
        blitCtx.pBlitParamsBuffer->setVariable(blitCtx.scaleVarOffset, srcRectScale);
        blitCtx.prevSrcReftScale = srcRectScale;
    }

    ref<Texture> pSharedTex = pDstResource->asTexture();
    blitCtx.pFbo->attachColorTarget(
        pSharedTex, 0, pDst->getViewInfo().mostDetailedMip, pDst->getViewInfo().firstArraySlice, pDst->getViewInfo().arraySize
    );
    blitCtx.pPass->getVars()->setSrv(blitCtx.texBindLoc, pSrc);
    blitCtx.pPass->getState()->setViewport(0, dstViewport);
    blitCtx.pPass->execute(this, blitCtx.pFbo, false);

    // Release the resources we bound
    blitCtx.pFbo->attachColorTarget(nullptr, 0);
    blitCtx.pPass->getVars()->setSrv(blitCtx.texBindLoc, nullptr);
}

void RenderContext::clearRtv(const RenderTargetView* pRtv, const float4& color)
{
    resourceBarrier(pRtv->getResource(), Resource::State::RenderTarget);
    gfx::ClearValue clearValue = {};
    memcpy(clearValue.color.floatValues, &color, sizeof(float) * 4);
    auto encoder = getLowLevelData()->getResourceCommandEncoder();
    encoder->clearResourceView(pRtv->getGfxResourceView(), &clearValue, gfx::ClearResourceViewFlags::FloatClearValues);
    mCommandsPending = true;
}

void RenderContext::clearDsv(const DepthStencilView* pDsv, float depth, uint8_t stencil, bool clearDepth, bool clearStencil)
{
    resourceBarrier(pDsv->getResource(), Resource::State::DepthStencil);
    gfx::ClearValue clearValue = {};
    clearValue.depthStencil.depth = depth;
    clearValue.depthStencil.stencil = stencil;
    auto encoder = getLowLevelData()->getResourceCommandEncoder();
    gfx::ClearResourceViewFlags::Enum flags = gfx::ClearResourceViewFlags::None;
    if (clearDepth)
        flags = (gfx::ClearResourceViewFlags::Enum)((int)flags | gfx::ClearResourceViewFlags::ClearDepth);
    if (clearStencil)
        flags = (gfx::ClearResourceViewFlags::Enum)((int)flags | gfx::ClearResourceViewFlags::ClearStencil);
    encoder->clearResourceView(pDsv->getGfxResourceView(), &clearValue, flags);
    mCommandsPending = true;
}

void RenderContext::drawInstanced(
    GraphicsState* pState,
    ProgramVars* pVars,
    uint32_t vertexCount,
    uint32_t instanceCount,
    uint32_t startVertexLocation,
    uint32_t startInstanceLocation
)
{
    auto encoder = drawCallCommon(pState, pVars);
    FALCOR_GFX_CALL(encoder->drawInstanced(vertexCount, instanceCount, startVertexLocation, startInstanceLocation));
    mCommandsPending = true;
}

void RenderContext::draw(GraphicsState* pState, ProgramVars* pVars, uint32_t vertexCount, uint32_t startVertexLocation)
{
    auto encoder = drawCallCommon(pState, pVars);
    FALCOR_GFX_CALL(encoder->draw(vertexCount, startVertexLocation));
    mCommandsPending = true;
}

void RenderContext::drawIndexedInstanced(
    GraphicsState* pState,
    ProgramVars* pVars,
    uint32_t indexCount,
    uint32_t instanceCount,
    uint32_t startIndexLocation,
    int32_t baseVertexLocation,
    uint32_t startInstanceLocation
)
{
    auto encoder = drawCallCommon(pState, pVars);
    FALCOR_GFX_CALL(encoder->drawIndexedInstanced(indexCount, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation)
    );
    mCommandsPending = true;
}

void RenderContext::drawIndexed(
    GraphicsState* pState,
    ProgramVars* pVars,
    uint32_t indexCount,
    uint32_t startIndexLocation,
    int32_t baseVertexLocation
)
{
    auto encoder = drawCallCommon(pState, pVars);
    FALCOR_GFX_CALL(encoder->drawIndexed(indexCount, startIndexLocation, baseVertexLocation));
    mCommandsPending = true;
}

void RenderContext::drawIndirect(
    GraphicsState* pState,
    ProgramVars* pVars,
    uint32_t maxCommandCount,
    const Buffer* pArgBuffer,
    uint64_t argBufferOffset,
    const Buffer* pCountBuffer,
    uint64_t countBufferOffset
)
{
    resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
    auto encoder = drawCallCommon(pState, pVars);
    FALCOR_GFX_CALL(encoder->drawIndirect(
        maxCommandCount,
        pArgBuffer->getGfxBufferResource(),
        argBufferOffset,
        pCountBuffer ? pCountBuffer->getGfxBufferResource() : nullptr,
        countBufferOffset
    ));
    mCommandsPending = true;
}

void RenderContext::drawIndexedIndirect(
    GraphicsState* pState,
    ProgramVars* pVars,
    uint32_t maxCommandCount,
    const Buffer* pArgBuffer,
    uint64_t argBufferOffset,
    const Buffer* pCountBuffer,
    uint64_t countBufferOffset
)
{
    resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
    auto encoder = drawCallCommon(pState, pVars);
    FALCOR_GFX_CALL(encoder->drawIndexedIndirect(
        maxCommandCount,
        pArgBuffer->getGfxBufferResource(),
        argBufferOffset,
        pCountBuffer ? pCountBuffer->getGfxBufferResource() : nullptr,
        countBufferOffset
    ));
    mCommandsPending = true;
}

void RenderContext::raytrace(Program* pProgram, RtProgramVars* pVars, uint32_t width, uint32_t height, uint32_t depth)
{
    auto pRtso = pProgram->getRtso(pVars);

    pVars->prepareShaderTable(this, pRtso.get());
    pVars->prepareDescriptorSets(this);

    auto rtEncoder = mpLowLevelData->getRayTracingCommandEncoder();
    FALCOR_GFX_CALL(rtEncoder->bindPipelineWithRootObject(pRtso->getGfxPipelineState(), pVars->getShaderObject()));
    FALCOR_GFX_CALL(rtEncoder->dispatchRays(0, pVars->getShaderTable(), width, height, depth));
    mCommandsPending = true;
}

void RenderContext::resolveSubresource(const ref<Texture>& pSrc, uint32_t srcSubresource, const ref<Texture>& pDst, uint32_t dstSubresource)
{
    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    gfx::SubresourceRange srcRange = {};
    srcRange.baseArrayLayer = pSrc->getSubresourceArraySlice(srcSubresource);
    srcRange.layerCount = 1;
    srcRange.mipLevel = pSrc->getSubresourceMipLevel(srcSubresource);
    srcRange.mipLevelCount = 1;

    gfx::SubresourceRange dstRange = {};
    dstRange.baseArrayLayer = pDst->getSubresourceArraySlice(dstSubresource);
    dstRange.layerCount = 1;
    dstRange.mipLevel = pDst->getSubresourceMipLevel(dstSubresource);
    dstRange.mipLevelCount = 1;

    resourceEncoder->resolveResource(
        pSrc->getGfxTextureResource(),
        gfx::ResourceState::ResolveSource,
        srcRange,
        pDst->getGfxTextureResource(),
        gfx::ResourceState::ResolveDestination,
        dstRange
    );
    mCommandsPending = true;
}

void RenderContext::resolveResource(const ref<Texture>& pSrc, const ref<Texture>& pDst)
{
    resourceBarrier(pSrc.get(), Resource::State::ResolveSource);
    resourceBarrier(pDst.get(), Resource::State::ResolveDest);

    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();

    gfx::SubresourceRange srcRange = {};
    gfx::SubresourceRange dstRange = {};

    resourceEncoder->resolveResource(
        pSrc->getGfxTextureResource(),
        gfx::ResourceState::ResolveSource,
        srcRange,
        pDst->getGfxTextureResource(),
        gfx::ResourceState::ResolveDestination,
        dstRange
    );
    mCommandsPending = true;
}

void RenderContext::buildAccelerationStructure(
    const RtAccelerationStructure::BuildDesc& desc,
    uint32_t postBuildInfoCount,
    RtAccelerationStructurePostBuildInfoDesc* pPostBuildInfoDescs
)
{
    GFXAccelerationStructureBuildInputsTranslator translator = {};

    gfx::IAccelerationStructure::BuildDesc buildDesc = {};
    buildDesc.dest = desc.dest->getGfxAccelerationStructure();
    buildDesc.scratchData = desc.scratchData;
    buildDesc.source = desc.source ? desc.source->getGfxAccelerationStructure() : nullptr;
    buildDesc.inputs = translator.translate(desc.inputs);

    std::vector<gfx::AccelerationStructureQueryDesc> queryDescs(postBuildInfoCount);
    for (uint32_t i = 0; i < postBuildInfoCount; i++)
    {
        queryDescs[i].firstQueryIndex = pPostBuildInfoDescs[i].index;
        queryDescs[i].queryPool = pPostBuildInfoDescs[i].pool->getGFXQueryPool();
        queryDescs[i].queryType = getGFXAccelerationStructurePostBuildQueryType(pPostBuildInfoDescs[i].type);
    }
    auto rtEncoder = getLowLevelData()->getRayTracingCommandEncoder();
    rtEncoder->buildAccelerationStructure(buildDesc, (int)postBuildInfoCount, queryDescs.data());
    mCommandsPending = true;
}

void RenderContext::copyAccelerationStructure(
    RtAccelerationStructure* dest,
    RtAccelerationStructure* source,
    RenderContext::RtAccelerationStructureCopyMode mode
)
{
    auto rtEncoder = getLowLevelData()->getRayTracingCommandEncoder();
    rtEncoder->copyAccelerationStructure(
        dest->getGfxAccelerationStructure(), source->getGfxAccelerationStructure(), getGFXAcclerationStructureCopyMode(mode)
    );
    mCommandsPending = true;
}

gfx::IRenderCommandEncoder* RenderContext::drawCallCommon(GraphicsState* pState, ProgramVars* pVars)
{
    // Insert barriers for bound resources.
    pVars->prepareDescriptorSets(this);

    // Insert barriers for render targets.
    ensureFboAttachmentResourceStates(this, pState->getFbo().get());

    // Insert barriers for vertex/index buffers.
    auto pGso = pState->getGSO(pVars).get();
    if (pGso != mpLastBoundGraphicsStateObject)
    {
        auto pVao = pState->getVao().get();
        for (uint32_t i = 0; i < pVao->getVertexBuffersCount(); i++)
        {
            auto vertexBuffer = pVao->getVertexBuffer(i).get();
            resourceBarrier(vertexBuffer, Resource::State::VertexBuffer);
        }
        if (pVao->getIndexBuffer())
        {
            auto indexBuffer = pVao->getIndexBuffer().get();
            resourceBarrier(indexBuffer, Resource::State::IndexBuffer);
        }
    }

    bool isNewEncoder = false;
    auto encoder = getLowLevelData()->getRenderCommandEncoder(
        pGso->getGFXRenderPassLayout(), pState->getFbo() ? pState->getFbo()->getGfxFramebuffer() : nullptr, isNewEncoder
    );

    FALCOR_GFX_CALL(encoder->bindPipelineWithRootObject(pGso->getGfxPipelineState(), pVars->getShaderObject()));

    if (isNewEncoder || pGso != mpLastBoundGraphicsStateObject)
    {
        mpLastBoundGraphicsStateObject = pGso;
        auto pVao = pState->getVao().get();
        auto pVertexLayout = pVao->getVertexLayout().get();
        for (uint32_t i = 0; i < pVao->getVertexBuffersCount(); i++)
        {
            auto bufferLayout = pVertexLayout->getBufferLayout(i);
            auto vertexBuffer = pVao->getVertexBuffer(i).get();
            encoder->setVertexBuffer(i, pVao->getVertexBuffer(i)->getGfxBufferResource(), bufferLayout->getElementOffset(0));
        }
        if (pVao->getIndexBuffer())
        {
            auto indexBuffer = pVao->getIndexBuffer().get();
            encoder->setIndexBuffer(indexBuffer->getGfxBufferResource(), getGFXFormat(pVao->getIndexBufferFormat()));
        }
        encoder->setPrimitiveTopology(getGFXPrimitiveTopology(pVao->getPrimitiveTopology()));
        encoder->setViewports(
            (uint32_t)pState->getViewports().size(), reinterpret_cast<const gfx::Viewport*>(pState->getViewports().data())
        );
        encoder->setScissorRects(
            (uint32_t)pState->getScissors().size(), reinterpret_cast<const gfx::ScissorRect*>(pState->getScissors().data())
        );
    }

    return encoder;
}

FALCOR_SCRIPT_BINDING(CopyContext)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Resource)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Buffer)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Texture)

    pybind11::class_<CopyContext> copyContext(m, "CopyContext");

    copyContext.def("submit", &CopyContext::submit, "wait"_a = false);

#if FALCOR_HAS_CUDA
    copyContext.def(
        "wait_for_cuda",
        [](CopyContext& self, uint64_t stream = 0) { self.waitForCuda(reinterpret_cast<cudaStream_t>(stream)); },
        "stream"_a = 0
    );
    copyContext.def(
        "wait_for_falcor",
        [](CopyContext& self, uint64_t stream = 0) { self.waitForFalcor(reinterpret_cast<cudaStream_t>(stream)); },
        "stream"_a = 0
    );
#endif

    copyContext.def("uav_barrier", &CopyContext::uavBarrier, "resource"_a);
    copyContext.def("copy_resource", &CopyContext::copyResource, "dst"_a, "src"_a);
    copyContext.def("copy_subresource", &CopyContext::copySubresource, "dst"_a, "dst_subresource_idx"_a, "src"_a, "src_subresource_idx"_a);
    copyContext.def("copy_buffer_region", &CopyContext::copyBufferRegion, "dst"_a, "dst_offset"_a, "src"_a, "src_offset"_a, "num_bytes"_a);
}

FALCOR_SCRIPT_BINDING(ComputeContext)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(CopyContext)

    pybind11::class_<ComputeContext, CopyContext> computeContext(m, "ComputeContext");
}

FALCOR_SCRIPT_BINDING(RenderContext)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(ComputeContext)

    pybind11::class_<RenderContext, ComputeContext> renderContext(m, "RenderContext");
}

} // namespace Falcor
