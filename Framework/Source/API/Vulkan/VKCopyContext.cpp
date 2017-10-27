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
#include "API/CopyContext.h"
#include "API/Buffer.h"
#include "API/Texture.h"
#include <cstring>

namespace Falcor
{
    VkImageAspectFlags getAspectFlagsFromFormat(ResourceFormat format)
    {
        VkImageAspectFlags flags = 0;
        if (isDepthFormat(format))      flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (isStencilFormat(format))    flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
        if (isDepthStencilFormat(format) == false) flags |= VK_IMAGE_ASPECT_COLOR_BIT;
        return flags;
    }

    static uint32_t getMipLevelPackedDataSize(const Texture* pTexture, uint32_t mipLevel)
    {
        assert(mipLevel < pTexture->getMipCount());
        ResourceFormat format = pTexture->getFormat();

        uint32_t w = pTexture->getWidth(mipLevel);
        uint32_t perW = getFormatWidthCompressionRatio(format);
        uint32_t bw = align_to(perW, w) / perW;

        uint32_t h = pTexture->getHeight(mipLevel);
        uint32_t perH = getFormatHeightCompressionRatio(format);
        uint32_t bh = align_to(perH, h) / perH;

        uint32_t d = pTexture->getDepth(mipLevel);

        uint32_t size = bh * bw * d * getFormatBytesPerBlock(format);
        return size;
    }

    static VkImageLayout getImageLayout(Resource::State state)
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
        case Resource::State::ResolveDest:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;            
        case Resource::State::DepthStencil:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case Resource::State::ShaderResource:
        case Resource::State::ResolveSource:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case Resource::State::CopyDest:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case Resource::State::CopySource:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            break;
        case Resource::State::Present:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        default:
            should_not_get_here();
            return VkImageLayout (-1);
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
        case Resource::State::CopyDest:
            return VK_ACCESS_TRANSFER_WRITE_BIT;
        case Resource::State::CopySource:
            return VK_ACCESS_TRANSFER_READ_BIT;
        case Resource::State::ResolveDest:
            return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        case Resource::State::ResolveSource:
            return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
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

    void CopyContext::updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData)
    {
        mCommandsPending = true;
        const uint8_t* pSubResData = (uint8_t*)pData;
        for (uint32_t i = 0; i < subresourceCount; i++)
        {
            uint32_t subresource = i + firstSubresource;
            updateTextureSubresource(pTexture, subresource, pSubResData);
            uint32_t mipLevel = pTexture->getSubresourceMipLevel(subresource);
            uint32_t offset = getMipLevelPackedDataSize(pTexture, mipLevel);
            pSubResData += offset;
        }
    }

    void initTexAccessParams(const Texture* pTexture, uint32_t subresourceIndex, VkBufferImageCopy& vkCopy, Buffer::SharedPtr& pStaging, const void* pSrcData, size_t& dataSize)
    {
        assert(isDepthStencilFormat(pTexture->getFormat()) == false); // #VKTODO Nothing complicated here, just that Vulkan doesn't support writing to both depth and stencil, which may be confusing to the user
        uint32_t mipLevel = pTexture->getSubresourceMipLevel(subresourceIndex);
        dataSize = getMipLevelPackedDataSize(pTexture, mipLevel);

        // Upload the data to a staging buffer
        pStaging = Buffer::create(dataSize, Buffer::BindFlags::None, pSrcData ? Buffer::CpuAccess::Write : Buffer::CpuAccess::Read, pSrcData);

        vkCopy = {};
        vkCopy.bufferOffset = pStaging->getGpuAddressOffset();
        vkCopy.bufferRowLength = 0;
        vkCopy.bufferImageHeight = 0;
        vkCopy.imageSubresource.aspectMask = getAspectFlagsFromFormat(pTexture->getFormat());
        vkCopy.imageSubresource.baseArrayLayer = pTexture->getSubresourceArraySlice(subresourceIndex);
        vkCopy.imageSubresource.layerCount = 1;
        vkCopy.imageSubresource.mipLevel = mipLevel;
        vkCopy.imageExtent.width = pTexture->getWidth(mipLevel);
        vkCopy.imageExtent.height = pTexture->getHeight(mipLevel);
        vkCopy.imageExtent.depth = pTexture->getDepth(mipLevel);
    }
    
    void CopyContext::updateTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex, const void* pData)
    {
        mCommandsPending = true;
        VkBufferImageCopy vkCopy;
        Buffer::SharedPtr pStaging;
        size_t dataSize;
        initTexAccessParams(pTexture, subresourceIndex, vkCopy, pStaging, pData, dataSize);

        // Execute the copy
        resourceBarrier(pTexture, Resource::State::CopyDest);
        resourceBarrier(pStaging.get(), Resource::State::CopySource);
        vkCmdCopyBufferToImage(mpLowLevelData->getCommandList(), pStaging->getApiHandle(), pTexture->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &vkCopy);
    }

    std::vector<uint8> CopyContext::readTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex)
    {
        mCommandsPending = true;
        VkBufferImageCopy vkCopy;
        Buffer::SharedPtr pStaging;
        size_t dataSize = 0;
        initTexAccessParams(pTexture, subresourceIndex, vkCopy, pStaging, nullptr, dataSize);

        // Execute the copy
        resourceBarrier(pTexture, Resource::State::CopySource);
        resourceBarrier(pStaging.get(), Resource::State::CopyDest);
        vkCmdCopyImageToBuffer(mpLowLevelData->getCommandList(), pTexture->getApiHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pStaging->getApiHandle(), 1, &vkCopy);

        flush(true);

        // Map and read the results
        std::vector<uint8> result(dataSize);
        uint8* pData = reinterpret_cast<uint8*>(pStaging->map(Buffer::MapType::Read));
        std::memcpy(result.data(), pData, dataSize);

        return result;
    }

    void CopyContext::resourceBarrier(const Resource* pResource, Resource::State newState)
    {
        if (pResource->getState() != newState)
        {
            if(pResource->getApiHandle().getType() == VkResourceType::Image)
            {
                const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
                VkImageMemoryBarrier barrier = {};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.newLayout = getImageLayout(newState);
                barrier.oldLayout = getImageLayout(pResource->mState);
                barrier.image = pResource->getApiHandle();
                barrier.subresourceRange.aspectMask = getAspectFlagsFromFormat(pTexture->getFormat());
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.layerCount = pTexture->getArraySize();
                barrier.subresourceRange.levelCount = pTexture->getMipCount();
                barrier.srcAccessMask = getAccessMask(pResource->mState);
                barrier.dstAccessMask = getAccessMask(newState);

                vkCmdPipelineBarrier(mpLowLevelData->getCommandList(), getShaderStageMask(pResource->mState, true), getShaderStageMask(newState, false), 0, 0, nullptr, 0, nullptr, 1, &barrier);
            }
            else
            {
                const Buffer* pBuffer = dynamic_cast<const Buffer*>(pResource);
                assert(pBuffer);
                VkBufferMemoryBarrier barrier = {};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = getAccessMask(pResource->mState);
                barrier.dstAccessMask = getAccessMask(newState);
                barrier.buffer = pBuffer->getApiHandle();
                barrier.offset = pBuffer->getGpuAddressOffset();
                barrier.size = pBuffer->getSize();

                vkCmdPipelineBarrier(mpLowLevelData->getCommandList(), getShaderStageMask(pResource->mState, true), getShaderStageMask(newState, false), 0, 0, nullptr, 1, &barrier, 0, nullptr);
            }


            pResource->mState = newState;
            mCommandsPending = true;
        }
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
}
