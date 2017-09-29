/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "API/RenderContext.h"
#include "API/LowLevel/DescriptorPool.h"
#include "API/Device.h"
#include "glm/gtc/type_ptr.hpp"
#include "VKState.h"

namespace Falcor
{
    VkImageAspectFlags getAspectFlagsFromFormat(ResourceFormat format);

    RenderContext::SharedPtr RenderContext::create(CommandQueueHandle queue)
    {
        SharedPtr pCtx = SharedPtr(new RenderContext());
        pCtx->mpLowLevelData = LowLevelContextData::create(LowLevelContextData::CommandQueueType::Direct, queue);
        if (pCtx->mpLowLevelData == nullptr)
        {
            return nullptr;
        }

        pCtx->bindDescriptorHeaps();

        if (spDrawCommandSig == nullptr)
        {
            initDrawCommandSignatures();
        }

        return pCtx;
    }

    RenderContext::~RenderContext() = default;

    template<typename ViewType, typename ClearType>
    void clearColorImageCommon(CopyContext* pCtx, const ViewType* pView, const ClearType& clearVal);

    void RenderContext::clearRtv(const RenderTargetView* pRtv, const glm::vec4& color)
    {
        clearColorImageCommon(this, pRtv, color);
        mCommandsPending = true;
    }

    void RenderContext::clearDsv(const DepthStencilView* pDsv, float depth, uint8_t stencil, bool clearDepth, bool clearStencil)
    {
        resourceBarrier(pDsv->getResource(), Resource::State::CopyDest);

        VkClearDepthStencilValue val;
        val.depth = depth;
        val.stencil = stencil;

        VkImageSubresourceRange range;
        const auto& viewInfo = pDsv->getViewInfo();
        range.baseArrayLayer = viewInfo.firstArraySlice;
        range.baseMipLevel = viewInfo.mostDetailedMip;
        range.layerCount = viewInfo.arraySize;
        range.levelCount = viewInfo.mipCount;
        range.aspectMask = clearDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0;
        range.aspectMask |= clearStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;

        vkCmdClearDepthStencilImage(mpLowLevelData->getCommandList(), pDsv->getResource()->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &val, 1, &range);
        mCommandsPending = true;
    }

    void setViewports(CommandListHandle cmdList, const std::vector<GraphicsState::Viewport>& viewports)
    {
        static_assert(offsetof(GraphicsState::Viewport, originX) == offsetof(VkViewport, x), "VP originX offset");
        static_assert(offsetof(GraphicsState::Viewport, originY) == offsetof(VkViewport, y), "VP originY offset");
        static_assert(offsetof(GraphicsState::Viewport, width) == offsetof(VkViewport, width), "VP width offset");
        static_assert(offsetof(GraphicsState::Viewport, height) == offsetof(VkViewport, height), "VP height offset");
        static_assert(offsetof(GraphicsState::Viewport, minDepth) == offsetof(VkViewport, minDepth), "VP minDepth offset");
        static_assert(offsetof(GraphicsState::Viewport, maxDepth) == offsetof(VkViewport, maxDepth), "VP maxDepth offset");

        vkCmdSetViewport(cmdList, 0, (uint32_t)viewports.size(), (VkViewport*)viewports.data());
    }

    void setScissors(CommandListHandle cmdList, const std::vector<GraphicsState::Scissor>& scissors)
    {
        std::vector<VkRect2D> vkScissors(scissors.size());
        for (size_t i = 0; i < scissors.size(); i++)
        {
            vkScissors[i].offset.x = scissors[i].left;
            vkScissors[i].offset.y = scissors[i].top;
            vkScissors[i].extent.width = scissors[i].right - scissors[i].left;
            vkScissors[i].extent.height = scissors[i].bottom - scissors[i].top;
        }
        vkCmdSetScissor(cmdList, 0, (uint32_t)scissors.size(), vkScissors.data());
    }

    static VkIndexType getVkIndexType(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::R16Uint:
            return VK_INDEX_TYPE_UINT16;
        case ResourceFormat::R32Uint:
            return VK_INDEX_TYPE_UINT32;
        default:
            should_not_get_here();
            return VK_INDEX_TYPE_MAX_ENUM;
        }
    }

    void setVao(CopyContext* pCtx, const Vao* pVao)
    {
        CommandListHandle cmdList = pCtx->getLowLevelData()->getCommandList();
        for (uint32_t i = 0; i < pVao->getVertexBuffersCount(); i++)
        {
            const Buffer* pVB = pVao->getVertexBuffer(i).get();
            VkDeviceSize offset = pVB->getGpuAddressOffset();
            VkBuffer handle = pVB->getApiHandle();
            vkCmdBindVertexBuffers(cmdList, i, 1, &handle, &offset);
            pCtx->resourceBarrier(pVB, Resource::State::VertexBuffer);
        }

        const Buffer* pIB = pVao->getIndexBuffer().get();
        if (pIB)
        {
            VkDeviceSize offset = pIB->getGpuAddressOffset();
            VkBuffer handle = pIB->getApiHandle();
            vkCmdBindIndexBuffer(cmdList, handle, offset, getVkIndexType(pVao->getIndexBufferFormat()));
            pCtx->resourceBarrier(pIB, Resource::State::IndexBuffer);
        }
    }

    void beginRenderPass(CommandListHandle cmdList, const Fbo* pFbo)
    {
        // Begin Render Pass
        const auto& fboHandle = pFbo->getApiHandle();
        VkRenderPassBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        beginInfo.renderPass = *fboHandle;
        beginInfo.framebuffer = *fboHandle;
        beginInfo.renderArea.offset = { 0, 0 };
        beginInfo.renderArea.extent = { pFbo->getWidth(), pFbo->getHeight() };

        // Only needed if attachments use VK_ATTACHMENT_LOAD_OP_CLEAR
        beginInfo.clearValueCount = 0;
        beginInfo.pClearValues = nullptr;

        vkCmdBeginRenderPass(cmdList, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
    }

    static void transitionFboResources(RenderContext* pCtx, const Fbo* pFbo)
    {
        // We are setting the entire RTV array to make sure everything that was previously bound is detached
        uint32_t colorTargets = Fbo::getMaxColorTargetCount();

        if (pFbo)
        {
            for (uint32_t i = 0; i < colorTargets; i++)
            {
                auto& pTexture = pFbo->getColorTexture(i);
                if (pTexture) pCtx->resourceBarrier(pTexture.get(), Resource::State::RenderTarget);
            }

            auto& pTexture = pFbo->getDepthStencilTexture();
            if (pTexture) pCtx->resourceBarrier(pTexture.get(), Resource::State::DepthStencil);
        }
    }

    static void endVkDraw(VkCommandBuffer cmdBuffer)
    {
        vkCmdEndRenderPass(cmdBuffer);
    }

    void RenderContext::prepareForDraw()
    {
        assert(mpGraphicsState);
        // Vao must be valid so at least primitive topology is known
        assert(mpGraphicsState->getVao().get());

        // Apply the vars. Must be first because applyGraphicsVars() might cause a flush
        if(mpGraphicsVars)
        {
            applyGraphicsVars();
        }

        GraphicsStateObject::SharedPtr pGSO = mpGraphicsState->getGSO(mpGraphicsVars.get());
        vkCmdBindPipeline(mpLowLevelData->getCommandList(), VK_PIPELINE_BIND_POINT_GRAPHICS, pGSO->getApiHandle());
        
        transitionFboResources(this, mpGraphicsState->getFbo().get());
        setViewports(mpLowLevelData->getCommandList(), mpGraphicsState->getViewports());
        setScissors(mpLowLevelData->getCommandList(), mpGraphicsState->getScissors());
        setVao(this, mpGraphicsState->getVao().get());
        beginRenderPass(mpLowLevelData->getCommandList(), mpGraphicsState->getFbo().get());
    }

    void RenderContext::drawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation)
    {
        prepareForDraw();
        vkCmdDraw(mpLowLevelData->getCommandList(), vertexCount, instanceCount, startVertexLocation, startInstanceLocation);
        endVkDraw(mpLowLevelData->getCommandList());
    }

    void RenderContext::draw(uint32_t vertexCount, uint32_t startVertexLocation)
    {
        drawInstanced(vertexCount, 1, startVertexLocation, 0);
    }

    void RenderContext::drawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation)
    {
        prepareForDraw();
        vkCmdDrawIndexed(mpLowLevelData->getCommandList(), indexCount, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation);
        endVkDraw(mpLowLevelData->getCommandList());
    }

    void RenderContext::drawIndexed(uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation)
    {
        drawIndexedInstanced(indexCount, 1, startIndexLocation, baseVertexLocation, 0);
    }

    void RenderContext::drawIndirect(const Buffer* pArgBuffer, uint64_t argBufferOffset)
    {
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        prepareForDraw();
        vkCmdDrawIndirect(mpLowLevelData->getCommandList(), pArgBuffer->getApiHandle(), argBufferOffset + pArgBuffer->getGpuAddressOffset(), 1, 0);
        endVkDraw(mpLowLevelData->getCommandList());
    }

    void RenderContext::drawIndexedIndirect(const Buffer* pArgBuffer, uint64_t argBufferOffset)
    {
        resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        prepareForDraw();
        vkCmdDrawIndexedIndirect(mpLowLevelData->getCommandList(), pArgBuffer->getApiHandle(), argBufferOffset + pArgBuffer->getGpuAddressOffset(), 1, 0);
        endVkDraw(mpLowLevelData->getCommandList());
    }

    void RenderContext::initDrawCommandSignatures()
    {
    }

    template<uint32_t offsetCount, typename ViewType>
    void initBlitData(const ViewType* pView, const uvec4& rect, VkImageSubresourceLayers& layer, VkOffset3D offset[offsetCount])
    {
        const Texture* pTex = dynamic_cast<const Texture*>(pView->getResource());

        layer.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // Can't blit depth texture
        const auto& viewInfo = pView->getViewInfo();
        layer.baseArrayLayer = viewInfo.firstArraySlice;
        layer.layerCount = viewInfo.arraySize;
        layer.mipLevel = viewInfo.mostDetailedMip;
        assert(pTex->getDepth(viewInfo.mostDetailedMip) == 1);

        offset[0].x =  (rect.x == -1) ? 0 : rect.x;
        offset[0].y = (rect.y == -1) ? 0 : rect.y;
        offset[0].z = 0;

        if(offsetCount > 1)
        {
            offset[1].x = (rect.z == -1) ? pTex->getWidth(viewInfo.mostDetailedMip) : rect.z;
            offset[1].y = (rect.w == -1) ? pTex->getHeight(viewInfo.mostDetailedMip) : rect.w;
            offset[1].z = 1;
        }
    }

    void RenderContext::blit(ShaderResourceView::SharedPtr pSrc, RenderTargetView::SharedPtr pDst, const uvec4& srcRect, const uvec4& dstRect, Sampler::Filter filter)
    {
        const Texture* pTexture = dynamic_cast<const Texture*>(pSrc->getResource());
        resourceBarrier(pSrc->getResource(), Resource::State::CopySource);
        resourceBarrier(pDst->getResource(), Resource::State::CopyDest);

        if (pTexture && pTexture->getSampleCount() > 1)
        {
            // Resolve
            VkImageResolve resolve;
            initBlitData<1>(pSrc.get(), srcRect, resolve.srcSubresource, &resolve.srcOffset);
            initBlitData<1>(pDst.get(), dstRect, resolve.dstSubresource, &resolve.dstOffset);
            const auto& viewInfo = pSrc->getViewInfo();
            resolve.extent.width = pTexture->getWidth(viewInfo.mostDetailedMip);
            resolve.extent.height = pTexture->getHeight(viewInfo.mostDetailedMip);
            resolve.extent.depth = 1;
            
            vkCmdResolveImage(mpLowLevelData->getCommandList(), pSrc->getResource()->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pDst->getResource()->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolve);
        }
        else
        {
            VkImageBlit blt;
            initBlitData<2>(pSrc.get(), srcRect, blt.srcSubresource, blt.srcOffsets);
            initBlitData<2>(pDst.get(), dstRect, blt.dstSubresource, blt.dstOffsets);

            vkCmdBlitImage(mpLowLevelData->getCommandList(), pSrc->getResource()->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pDst->getResource()->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blt, getVkFilter(filter));
        }
    }
}
