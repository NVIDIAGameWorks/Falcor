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
#include "Core/API/RenderContext.h"
#include "GFXLowLevelContextApiData.h"
#include "GFXFormats.h"
#include "GFXRtAccelerationStructure.h"
#include "Core/API/BlitContext.h"
#include "Core/API/GFX/GFXAPI.h"
#include "Core/State/GraphicsState.h"
#include "Core/Program/ProgramVars.h"
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

        gfx::IRenderCommandEncoder* drawCallCommon(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars)
        {
            static GraphicsStateObject* spLastGso = nullptr;

            // Insert barriers for bound resources.
            pVars->prepareDescriptorSets(pContext);

            // Insert barriers for render targets.
            ensureFboAttachmentResourceStates(pContext, pState->getFbo().get());

            // Insert barriers for vertex/index buffers.
            auto pGso = pState->getGSO(pVars).get();
            if (pGso != spLastGso)
            {
                auto pVao = pState->getVao().get();
                for (uint32_t i = 0; i < pVao->getVertexBuffersCount(); i++)
                {
                    auto vertexBuffer = pVao->getVertexBuffer(i).get();
                    pContext->resourceBarrier(vertexBuffer, Resource::State::VertexBuffer);
                }
                if (pVao->getIndexBuffer())
                {
                    auto indexBuffer = pVao->getIndexBuffer().get();
                    pContext->resourceBarrier(indexBuffer, Resource::State::IndexBuffer);
                }
            }

            bool isNewEncoder = false;
            auto encoder = pContext->getLowLevelData()->getApiData()->getRenderCommandEncoder(
                pGso->getGFXRenderPassLayout(),
                pState->getFbo() ? pState->getFbo()->getApiHandle() : nullptr,
                isNewEncoder);

            FALCOR_GFX_CALL(encoder->bindPipelineWithRootObject(pGso->getApiHandle(), pVars->getShaderObject()));

            if (isNewEncoder || pGso != spLastGso)
            {
                spLastGso = pGso;
                auto pVao = pState->getVao().get();
                auto pVertexLayout = pVao->getVertexLayout().get();
                for (uint32_t i = 0; i < pVao->getVertexBuffersCount(); i++)
                {
                    auto bufferLayout = pVertexLayout->getBufferLayout(i);
                    auto vertexBuffer = pVao->getVertexBuffer(i).get();
                    encoder->setVertexBuffer(
                        i,
                        static_cast<gfx::IBufferResource*>(pVao->getVertexBuffer(i)->getApiHandle().get()),
                        bufferLayout->getElementOffset(0) + (uint32_t)vertexBuffer->getGpuAddressOffset());
                }
                if (pVao->getIndexBuffer())
                {
                    auto indexBuffer = pVao->getIndexBuffer().get();
                    encoder->setIndexBuffer(
                        static_cast<gfx::IBufferResource*>(indexBuffer->getApiHandle().get()),
                        getGFXFormat(pVao->getIndexBufferFormat()),
                        (uint32_t)indexBuffer->getGpuAddressOffset());
                }
                encoder->setPrimitiveTopology(getGFXPrimitiveTopology(pVao->getPrimitiveTopology()));
                encoder->setViewports((uint32_t)pState->getViewports().size(), reinterpret_cast<const gfx::Viewport*>(pState->getViewports().data()));
                encoder->setScissorRects((uint32_t)pState->getScissors().size(), reinterpret_cast<const gfx::ScissorRect*>(pState->getScissors().data()));
            }

            return encoder;
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

        struct RenderContextApiData
        {
            size_t refCount = 0;
            BlitContext blitData;

            static void init();
            static void release();
        };

        RenderContextApiData sApiData;

        void RenderContextApiData::init()
        {
            sApiData.blitData.init();
            sApiData.refCount++;
        }

        void RenderContextApiData::release()
        {
            sApiData.refCount--;
            if (sApiData.refCount == 0)
            {
                sApiData.blitData.release();
                sApiData = {};
            }
        }
    }

    RenderContext::RenderContext(CommandQueueHandle queue)
        : ComputeContext(LowLevelContextData::CommandQueueType::Direct, queue)
    {
        RenderContextApiData::init();
    }

    RenderContext::~RenderContext()
    {
        RenderContextApiData::release();
    }

    BlitContext& RenderContext::getBlitContext() { return sApiData.blitData; }

    void RenderContext::clearRtv(const RenderTargetView* pRtv, const float4& color)
    {
        resourceBarrier(pRtv->getResource().get(), Resource::State::RenderTarget);
        gfx::ClearValue clearValue = {};
        memcpy(clearValue.color.floatValues, &color, sizeof(float) * 4);
        auto encoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        encoder->clearResourceView(pRtv->getApiHandle(), &clearValue, gfx::ClearResourceViewFlags::FloatClearValues);
        mCommandsPending = true;
    }

    void RenderContext::clearDsv(const DepthStencilView* pDsv, float depth, uint8_t stencil, bool clearDepth, bool clearStencil)
    {
        resourceBarrier(pDsv->getResource().get(), Resource::State::DepthStencil);
        gfx::ClearValue clearValue = {};
        clearValue.depthStencil.depth = depth;
        clearValue.depthStencil.stencil = stencil;
        auto encoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::ClearResourceViewFlags::Enum flags = gfx::ClearResourceViewFlags::None;
        if (clearDepth) flags = (gfx::ClearResourceViewFlags::Enum)((int)flags | gfx::ClearResourceViewFlags::ClearDepth);
        if (clearStencil) flags = (gfx::ClearResourceViewFlags::Enum)((int)flags | gfx::ClearResourceViewFlags::ClearStencil);
        encoder->clearResourceView(pDsv->getApiHandle(), &clearValue, flags);
        mCommandsPending = true;
    }

    void RenderContext::drawInstanced(GraphicsState* pState, GraphicsVars* pVars, uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation)
    {
        auto encoder = drawCallCommon(this, pState, pVars);
        encoder->drawInstanced(vertexCount, instanceCount, startVertexLocation, startInstanceLocation);
        mCommandsPending = true;
    }

    void RenderContext::draw(GraphicsState* pState, GraphicsVars* pVars, uint32_t vertexCount, uint32_t startVertexLocation)
    {
        auto encoder = drawCallCommon(this, pState, pVars);
        encoder->draw(vertexCount, startVertexLocation);
        mCommandsPending = true;
    }

    void RenderContext::drawIndexedInstanced(GraphicsState* pState, GraphicsVars* pVars, uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation)
    {
        auto encoder = drawCallCommon(this, pState, pVars);
        encoder->drawIndexedInstanced(indexCount, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation);
        mCommandsPending = true;
    }

    void RenderContext::drawIndexed(GraphicsState* pState, GraphicsVars* pVars, uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation)
    {
        auto encoder = drawCallCommon(this, pState, pVars);
        encoder->drawIndexed(indexCount, startIndexLocation, baseVertexLocation);
        mCommandsPending = true;
    }

    void RenderContext::drawIndirect(GraphicsState* pState, GraphicsVars* pVars, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset)
    {
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        auto encoder = drawCallCommon(this, pState, pVars);
        encoder->drawIndirect(
            maxCommandCount,
            static_cast<gfx::IBufferResource*>(pArgBuffer->getApiHandle().get()),
            argBufferOffset,
            pCountBuffer ? static_cast<gfx::IBufferResource*>(pCountBuffer->getApiHandle().get()) : nullptr,
            countBufferOffset);
        mCommandsPending = true;
    }

    void RenderContext::drawIndexedIndirect(GraphicsState* pState, GraphicsVars* pVars, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset)
    {
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        auto encoder = drawCallCommon(this, pState, pVars);
        encoder->drawIndexedIndirect(
            maxCommandCount,
            static_cast<gfx::IBufferResource*>(pArgBuffer->getApiHandle().get()),
            argBufferOffset,
            pCountBuffer ? static_cast<gfx::IBufferResource*>(pCountBuffer->getApiHandle().get()) : nullptr,
            countBufferOffset);
        mCommandsPending = true;
    }

    void RenderContext::raytrace(RtProgram* pProgram, RtProgramVars* pVars, uint32_t width, uint32_t height, uint32_t depth)
    {
        auto pRtso = pProgram->getRtso(pVars);

        pVars->prepareShaderTable(this, pRtso.get());
        pVars->prepareDescriptorSets(this);

        auto rtEncoder = mpLowLevelData->getApiData()->getRayTracingCommandEncoder();
        FALCOR_GFX_CALL(rtEncoder->bindPipelineWithRootObject(pRtso->getApiHandle(), pVars->getShaderObject()));
        rtEncoder->dispatchRays(0, pVars->getShaderTable(), width, height, depth);
        mCommandsPending = true;
    }

    void RenderContext::resolveSubresource(const Texture::SharedPtr& pSrc, uint32_t srcSubresource, const Texture::SharedPtr& pDst, uint32_t dstSubresource)
    {
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
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
            static_cast<gfx::ITextureResource*>(pSrc->getApiHandle().get()),
            gfx::ResourceState::ResolveSource,
            srcRange,
            static_cast<gfx::ITextureResource*>(pDst->getApiHandle().get()),
            gfx::ResourceState::ResolveDestination,
            dstRange);
        mCommandsPending = true;
    }

    void RenderContext::resolveResource(const Texture::SharedPtr& pSrc, const Texture::SharedPtr& pDst)
    {
        resourceBarrier(pSrc.get(), Resource::State::ResolveSource);
        resourceBarrier(pDst.get(), Resource::State::ResolveDest);

        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();

        gfx::SubresourceRange srcRange = {};
        gfx::SubresourceRange dstRange = {};

        resourceEncoder->resolveResource(
            static_cast<gfx::ITextureResource*>(pSrc->getApiHandle().get()),
            gfx::ResourceState::ResolveSource,
            srcRange,
            static_cast<gfx::ITextureResource*>(pDst->getApiHandle().get()),
            gfx::ResourceState::ResolveDestination,
            dstRange);
        mCommandsPending = true;
    }

    void RenderContext::buildAccelerationStructure(const RtAccelerationStructure::BuildDesc& desc, uint32_t postBuildInfoCount, RtAccelerationStructurePostBuildInfoDesc* pPostBuildInfoDescs)
    {
        GFXAccelerationStructureBuildInputsTranslator translator = {};

        gfx::IAccelerationStructure::BuildDesc buildDesc = {};
        buildDesc.dest = desc.dest->getApiHandle();
        buildDesc.scratchData = desc.scratchData;
        buildDesc.source = desc.source ? desc.source->getApiHandle() : nullptr;
        buildDesc.inputs = translator.translate(desc.inputs);

        std::vector<gfx::AccelerationStructureQueryDesc> queryDescs(postBuildInfoCount);
        for (uint32_t i = 0; i < postBuildInfoCount; i++)
        {
            queryDescs[i].firstQueryIndex = pPostBuildInfoDescs[i].index;
            queryDescs[i].queryPool = pPostBuildInfoDescs[i].pool->getGFXQueryPool();
            queryDescs[i].queryType = getGFXAccelerationStructurePostBuildQueryType(pPostBuildInfoDescs[i].type);
        }
        auto rtEncoder = getLowLevelData()->getApiData()->getRayTracingCommandEncoder();
        rtEncoder->buildAccelerationStructure(buildDesc, (int)postBuildInfoCount, queryDescs.data());
        mCommandsPending = true;
    }

    void RenderContext::copyAccelerationStructure(RtAccelerationStructure* dest, RtAccelerationStructure* source, RenderContext::RtAccelerationStructureCopyMode mode)
    {
        auto rtEncoder = getLowLevelData()->getApiData()->getRayTracingCommandEncoder();
        rtEncoder->copyAccelerationStructure(dest->getApiHandle(), source->getApiHandle(), getGFXAcclerationStructureCopyMode(mode));
        mCommandsPending = true;
    }
}
