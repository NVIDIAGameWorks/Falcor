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
#include "API/CopyContext.h"
#include "API/Buffer.h"
#include "API/Texture.h"
#include <cstring>

namespace Falcor
{
    VkImageAspectFlags getAspectFlagsFromFormat(ResourceFormat format, bool ignoreStencil = false)
    {
        VkImageAspectFlags flags = 0;
        if (isDepthFormat(format))      flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (ignoreStencil == false)
        {
            if (isStencilFormat(format))    flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        if (isDepthStencilFormat(format) == false) flags |= VK_IMAGE_ASPECT_COLOR_BIT;
        return flags;
    }

    static uint32_t getMipLevelPackedDataSize(const Texture* pTexture, uint32_t w, uint32_t h, uint32_t d, ResourceFormat format)
    {
        uint32_t perW = getFormatWidthCompressionRatio(format);
        uint32_t bw = align_to(perW, w) / perW;

        uint32_t perH = getFormatHeightCompressionRatio(format);
        uint32_t bh = align_to(perH, h) / perH;

        uint32_t size = bh * bw * d * getFormatBytesPerBlock(format);
        return size;
    }

    VkImageLayout getImageLayout(Resource::State state)
    {
        switch (state)
        {
        case Resource::State::Undefined:
            return VK_IMAGE_LAYOUT_UNDEFINED;
        case Resource::State::PreInitialized:
            return VK_IMAGE_LAYOUT_PREINITIALIZED;
        case Resource::State::Common:
        case Resource::State::UnorderedAccess:
            return VK_IMAGE_LAYOUT_GENERAL;
        case Resource::State::RenderTarget:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        case Resource::State::DepthStencil:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case Resource::State::ShaderResource:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case Resource::State::ResolveDest:
        case Resource::State::CopyDest:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case Resource::State::ResolveSource:
        case Resource::State::CopySource:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            break;
        case Resource::State::Present:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        default:
            should_not_get_here();
            return VkImageLayout(-1);
        }
    }

    static VkAccessFlagBits getAccessMask(Resource::State state)
    {
        switch (state)
        {
        case Resource::State::Undefined:
        case Resource::State::Present:
        case Resource::State::Common:
        case Resource::State::PreInitialized:
            return VkAccessFlagBits(0);
        case Resource::State::VertexBuffer:
            return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        case Resource::State::ConstantBuffer:
            return VK_ACCESS_UNIFORM_READ_BIT;
        case Resource::State::IndexBuffer:
            return VK_ACCESS_INDEX_READ_BIT;
        case Resource::State::RenderTarget:
            return VkAccessFlagBits(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
        case Resource::State::UnorderedAccess:
            return VK_ACCESS_SHADER_WRITE_BIT;
        case Resource::State::DepthStencil:
            return VkAccessFlagBits(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        case Resource::State::ShaderResource:
            return VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
        case Resource::State::IndirectArg:
            return VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        case Resource::State::ResolveDest:
        case Resource::State::CopyDest:
            return VK_ACCESS_TRANSFER_WRITE_BIT;
        case Resource::State::ResolveSource:
        case Resource::State::CopySource:
            return VK_ACCESS_TRANSFER_READ_BIT;
        default:
            should_not_get_here();
            return VkAccessFlagBits(-1);
        }
    }

    static VkPipelineStageFlags getShaderStageMask(Resource::State state, bool src)
    {
        switch (state)
        {
        case Resource::State::Undefined:
        case Resource::State::PreInitialized:
        case Resource::State::Common:
            assert(src);
            return src ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : (VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        case Resource::State::VertexBuffer:
        case Resource::State::IndexBuffer:
            return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        case Resource::State::UnorderedAccess:
        case Resource::State::ConstantBuffer:
        case Resource::State::ShaderResource:
            return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT; // #OPTME Assume the worst
        case Resource::State::RenderTarget:
            return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        case Resource::State::DepthStencil:
            return src ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT : VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        case Resource::State::IndirectArg:
            return VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        case Resource::State::CopyDest:
        case Resource::State::CopySource:
        case Resource::State::ResolveDest:
        case Resource::State::ResolveSource:
            return VK_PIPELINE_STAGE_TRANSFER_BIT;
        case Resource::State::Present:
            return src ? (VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        default:
            should_not_get_here();
            return VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        }
    }

    void CopyContext::bindDescriptorHeaps()
    {
    }

    static void initTexAccessParams(const Texture* pTexture, uint32_t subresourceIndex, VkBufferImageCopy& vkCopy, Buffer::SharedPtr& pStaging, const void* pSrcData, const uint3& offset, const uint3& size, size_t& dataSize)
    {
        assert(isDepthStencilFormat(pTexture->getFormat()) == false); // #VKTODO Nothing complicated here, just that Vulkan doesn't support writing to both depth and stencil, which may be confusing to the user
        uint32_t mipLevel = pTexture->getSubresourceMipLevel(subresourceIndex);

        vkCopy = {};
        vkCopy.bufferRowLength = 0;
        vkCopy.bufferImageHeight = 0;
        vkCopy.imageSubresource.aspectMask = getAspectFlagsFromFormat(pTexture->getFormat());
        vkCopy.imageSubresource.baseArrayLayer = pTexture->getSubresourceArraySlice(subresourceIndex);
        vkCopy.imageSubresource.layerCount = 1;
        vkCopy.imageSubresource.mipLevel = mipLevel;
        vkCopy.imageOffset = { (int32_t)offset.x, (int32_t)offset.y, (int32_t)offset.z };
        vkCopy.imageExtent.width = (size.x == -1) ? pTexture->getWidth(mipLevel) - offset.x : size.x;
        vkCopy.imageExtent.height = (size.y == -1) ? pTexture->getHeight(mipLevel) - offset.y : size.y;
        vkCopy.imageExtent.depth = (size.z == -1) ? pTexture->getDepth(mipLevel) - offset.z : size.z;

        dataSize = getMipLevelPackedDataSize(pTexture, vkCopy.imageExtent.width, vkCopy.imageExtent.height, vkCopy.imageExtent.depth, pTexture->getFormat());

        // Upload the data to a staging buffer
        pStaging = Buffer::create(dataSize, Buffer::BindFlags::None, pSrcData ? Buffer::CpuAccess::Write : Buffer::CpuAccess::Read, pSrcData);
        vkCopy.bufferOffset = pStaging->getGpuAddressOffset();
    }

    static void updateTextureSubresource(CopyContext* pCtx, const Texture* pTexture, uint32_t subresourceIndex, const void* pData, const uint3& offset, const uint3& size)
    {
        VkBufferImageCopy vkCopy;
        Buffer::SharedPtr pStaging;
        size_t dataSize;
        initTexAccessParams(pTexture, subresourceIndex, vkCopy, pStaging, pData, offset, size, dataSize);

        // Execute the copy
        pCtx->resourceBarrier(pTexture, Resource::State::CopyDest);
        pCtx->resourceBarrier(pStaging.get(), Resource::State::CopySource);
        vkCmdCopyBufferToImage(pCtx->getLowLevelData()->getCommandList(), pStaging->getApiHandle(), pTexture->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &vkCopy);
    }

    void CopyContext::updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData, const uint3& offset, const uint3& size)
    {
        bool copyRegion = (offset != uint3(0)) || (size != uint3(-1));
        assert(subresourceCount == 1 || (copyRegion == false));

        mCommandsPending = true;
        const uint8_t* pSubResData = (uint8_t*)pData;
        for (uint32_t i = 0; i < subresourceCount; i++)
        {
            uint32_t subresource = i + firstSubresource;
            updateTextureSubresource(this, pTexture, subresource, pSubResData, offset, size);
            uint32_t mipLevel = pTexture->getSubresourceMipLevel(subresource);
            uint32_t offset = getMipLevelPackedDataSize(pTexture, pTexture->getWidth(mipLevel), pTexture->getHeight(mipLevel), pTexture->getDepth(mipLevel), pTexture->getFormat());
            pSubResData += offset;
        }
    }

    CopyContext::ReadTextureTask::SharedPtr CopyContext::ReadTextureTask::create(CopyContext* pCtx, const Texture* pTexture, uint32_t subresourceIndex)
    {
        SharedPtr pThis = SharedPtr(new ReadTextureTask);
        pThis->mpContext = pCtx;

        VkBufferImageCopy vkCopy;
        initTexAccessParams(pTexture, subresourceIndex, vkCopy, pThis->mpBuffer, nullptr, {}, uint3(-1, -1, -1), pThis->mDataSize);

        // Execute the copy
        pCtx->resourceBarrier(pTexture, Resource::State::CopySource);
        pCtx->resourceBarrier(pThis->mpBuffer.get(), Resource::State::CopyDest);
        vkCmdCopyImageToBuffer(pCtx->getLowLevelData()->getCommandList(), pTexture->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pThis->mpBuffer->getApiHandle(), 1, &vkCopy);

        // Create a fence and signal
        pThis->mpFence = GpuFence::create();
        pCtx->flush(false);
        pThis->mpFence->gpuSignal(pCtx->getLowLevelData()->getCommandQueue());

        return pThis;
    }

    std::vector<uint8_t> CopyContext::ReadTextureTask::getData()
    {
        mpFence->syncCpu();
        // Map and read the results
        std::vector<uint8> result(mDataSize);
        uint8* pData = reinterpret_cast<uint8*>(mpBuffer->map(Buffer::MapType::Read));
        std::memcpy(result.data(), pData, mDataSize);
        return result;
    }

    void CopyContext::uavBarrier(const Resource* pResource)
    {
        UNSUPPORTED_IN_VULKAN("uavBarrier");
    }

    void CopyContext::apiSubresourceBarrier(const Texture* pTexture, Resource::State newState, Resource::State oldState, uint32_t arraySlice, uint32_t mipLevel)
    {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.newLayout = getImageLayout(newState);
        barrier.oldLayout = getImageLayout(oldState);
        barrier.image = pTexture->getApiHandle();
        barrier.subresourceRange.aspectMask = getAspectFlagsFromFormat(pTexture->getFormat());
        barrier.subresourceRange.baseArrayLayer = arraySlice;
        barrier.subresourceRange.baseMipLevel = mipLevel;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.srcAccessMask = getAccessMask(oldState);
        barrier.dstAccessMask = getAccessMask(newState);

        vkCmdPipelineBarrier(mpLowLevelData->getCommandList(), getShaderStageMask(oldState, true), getShaderStageMask(newState, false), 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    bool CopyContext::textureBarrier(const Texture* pTexture, Resource::State newState)
    {
        assert(pTexture->getApiHandle().getType() == VkResourceType::Image);

        VkImageLayout srcLayout = getImageLayout(pTexture->getGlobalState());
        VkImageLayout dstLayout = getImageLayout(newState);

        if (srcLayout != dstLayout)
        {
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = srcLayout;
            barrier.newLayout = dstLayout;
            barrier.image = pTexture->getApiHandle();
            barrier.subresourceRange.aspectMask = getAspectFlagsFromFormat(pTexture->getFormat());
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = pTexture->getArraySize();
            barrier.subresourceRange.levelCount = pTexture->getMipCount();
            barrier.srcAccessMask = getAccessMask(pTexture->getGlobalState());
            barrier.dstAccessMask = getAccessMask(newState);

            VkPipelineStageFlags srcStageMask = getShaderStageMask(pTexture->getGlobalState(), true);
            VkPipelineStageFlags dstStageMask = getShaderStageMask(newState, false);
            vkCmdPipelineBarrier(mpLowLevelData->getCommandList(), srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            pTexture->setGlobalState(newState);
            mCommandsPending = true;
            return true;
        }
        return false;
    }

    bool CopyContext::bufferBarrier(const Buffer* pBuffer, Resource::State newState)
    {
        assert(pBuffer);
        assert(pBuffer->getApiHandle().getType() == VkResourceType::Buffer);

        VkPipelineStageFlags srcStageMask = getShaderStageMask(pBuffer->getGlobalState(), true);
        VkPipelineStageFlags dstStageMask = getShaderStageMask(newState, false);

        if (srcStageMask != dstStageMask)
        {
            VkBufferMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = getAccessMask(pBuffer->getGlobalState());
            barrier.dstAccessMask = getAccessMask(newState);
            barrier.buffer = pBuffer->getApiHandle();
            barrier.offset = pBuffer->getGpuAddressOffset();
            barrier.size = pBuffer->getSize();

            vkCmdPipelineBarrier(mpLowLevelData->getCommandList(), srcStageMask, dstStageMask, 0, 0, nullptr, 1, &barrier, 0, nullptr);

            pBuffer->setGlobalState(newState);
            mCommandsPending = true;
            return true;
        }
        return false;
    }

    void CopyContext::copyResource(const Resource* pDst, const Resource* pSrc)
    {
        const Buffer* pDstBuffer = dynamic_cast<const Buffer*>(pDst);
        if (pDstBuffer)
        {
            const Buffer* pSrcBuffer = dynamic_cast<const Buffer*>(pSrc);
            assert(pSrcBuffer && (pSrcBuffer->getSize() == pDstBuffer->getSize()));
            copyBufferRegion(pDstBuffer, 0, pSrcBuffer, 0, pSrcBuffer->getSize());
        }
        else
        {
            const Texture* pSrcTex = dynamic_cast<const Texture*>(pSrc);
            const Texture* pDstTex = dynamic_cast<const Texture*>(pDst);
            assert(pSrcTex && pDstTex);
            assert((pSrcTex->getArraySize() == pDstTex->getArraySize()) && (pSrcTex->getMipCount() == pDstTex->getMipCount()));

            uint32_t mipCount = pSrcTex->getMipCount();
            std::vector<VkImageCopy> regions(mipCount);
            VkImageAspectFlags srcAspect = getAspectFlagsFromFormat(pSrcTex->getFormat());
            VkImageAspectFlags dstAspect = getAspectFlagsFromFormat(pDstTex->getFormat());
            uint32_t arraySize = pSrcTex->getArraySize();
            for (uint32_t i = 0; i < mipCount; i++)
            {
                regions[i] = {};
                regions[i].srcSubresource.aspectMask = srcAspect;
                regions[i].srcSubresource.baseArrayLayer = 0;
                regions[i].srcSubresource.layerCount = arraySize;
                regions[i].srcSubresource.mipLevel = i;

                regions[i].dstSubresource = regions[i].srcSubresource;
                regions[i].dstSubresource.aspectMask = dstAspect;

                regions[i].extent.width = pSrcTex->getWidth(i);
                regions[i].extent.height = pSrcTex->getHeight(i);
                regions[i].extent.depth = pSrcTex->getDepth(i);
            }

            resourceBarrier(pDst, Resource::State::CopyDest);
            resourceBarrier(pSrc, Resource::State::CopySource);
            vkCmdCopyImage(mpLowLevelData->getCommandList(), pSrc->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pDst->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipCount, regions.data());
        }
        mCommandsPending = true;
    }

    void initRegionSubresourceData(const Texture* pTex, uint32_t subresource, VkImageSubresourceLayers& data)
    {
        data.aspectMask = getAspectFlagsFromFormat(pTex->getFormat());
        data.baseArrayLayer = pTex->getSubresourceArraySlice(subresource);
        data.layerCount = 1;
        data.mipLevel = pTex->getSubresourceMipLevel(subresource);
    }

    void CopyContext::copySubresource(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);
        VkImageCopy region = {};
        initRegionSubresourceData(pSrc, srcSubresourceIdx, region.srcSubresource);
        initRegionSubresourceData(pDst, dstSubresourceIdx, region.dstSubresource);
        uint32_t mipLevel = region.dstSubresource.mipLevel;

        region.extent.width = pDst->getWidth(mipLevel);
        region.extent.height = pDst->getHeight(mipLevel);
        region.extent.depth = pDst->getDepth(mipLevel);
        vkCmdCopyImage(mpLowLevelData->getCommandList(), pSrc->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pDst->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        mCommandsPending = true;
    }

    void CopyContext::copyBufferRegion(const Buffer* pDst, uint64_t dstOffset, const Buffer* pSrc, uint64_t srcOffset, uint64_t numBytes)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);
        VkBufferCopy region;
        region.srcOffset = pSrc->getGpuAddressOffset() + srcOffset;
        region.dstOffset = pDst->getGpuAddressOffset() + dstOffset;
        region.size = numBytes;

        vkCmdCopyBuffer(mpLowLevelData->getCommandList(), pSrc->getApiHandle(), pDst->getApiHandle(), 1, &region);
        mCommandsPending = true;
    }

    void CopyContext::copySubresourceRegion(const Texture* pDst, uint32_t dstSubresource, const Texture* pSrc, uint32_t srcSubresource, const uint3& dstOffset, const uint3& srcOffset, const uint3& size)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        VkImageCopy region = {};
        // Source subresource
        region.srcSubresource.layerCount = 1;
        region.srcSubresource.baseArrayLayer = pSrc->getSubresourceArraySlice(srcSubresource);
        region.srcSubresource.mipLevel = pSrc->getSubresourceMipLevel(srcSubresource);
        region.srcSubresource.aspectMask = getAspectFlagsFromFormat(pSrc->getFormat());

        // Dst subresource
        region.dstSubresource.layerCount = 1;
        region.dstSubresource.baseArrayLayer = pDst->getSubresourceArraySlice(dstSubresource);
        region.dstSubresource.mipLevel = pDst->getSubresourceMipLevel(dstSubresource);
        region.dstSubresource.aspectMask = getAspectFlagsFromFormat(pDst->getFormat());

        region.dstOffset = { (int32_t)dstOffset.x, (int32_t)dstOffset.y, (int32_t)dstOffset.z };
        region.srcOffset = { (int32_t)srcOffset.x, (int32_t)srcOffset.y, (int32_t)srcOffset.z };

        uint32_t mipLevel = region.srcSubresource.mipLevel;

        region.extent.width = (size.x == -1) ? pSrc->getWidth(mipLevel) - srcOffset.x : size.x;
        region.extent.height = (size.y == -1) ? pSrc->getHeight(mipLevel) - srcOffset.y : size.y;
        region.extent.depth = (size.z == -1) ? pSrc->getDepth(mipLevel) - srcOffset.z : size.z;

        vkCmdCopyImage(mpLowLevelData->getCommandList(), pSrc->getApiHandle(), getImageLayout(Resource::State::CopySource), pDst->getApiHandle(), getImageLayout(Resource::State::CopyDest), 1, &region);

        mCommandsPending = true;
    }
}
