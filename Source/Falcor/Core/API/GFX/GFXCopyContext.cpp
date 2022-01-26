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
#include "Core/API/CopyContext.h"
#include "GFXLowLevelContextApiData.h"
#include "GFXFormats.h"
#include "GFXResource.h"

namespace Falcor
{
    void CopyContext::bindDescriptorHeaps()
    {
    }

    void CopyContext::updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData, const uint3& offset, const uint3& size)
    {
        resourceBarrier(pTexture, Resource::State::CopyDest);

        bool copyRegion = (offset != uint3(0)) || (size != uint3(-1));
        FALCOR_ASSERT(subresourceCount == 1 || (copyRegion == false));
        uint8_t* dataPtr = (uint8_t*)pData;
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::ITextureResource::Offset3D gfxOffset = { offset.x, offset.y, offset.z };
        gfx::ITextureResource::Size gfxSize = { (int)size.x, (int)size.y, (int)size.z };
        gfx::FormatInfo formatInfo = {};
        gfx::gfxGetFormatInfo(getGFXFormat(pTexture->getFormat()), &formatInfo);
        for (uint32_t index = firstSubresource; index < firstSubresource + subresourceCount; index++)
        {
            gfx::SubresourceRange subresourceRange = {};
            subresourceRange.baseArrayLayer = pTexture->getSubresourceArraySlice(index);
            subresourceRange.mipLevel = pTexture->getSubresourceMipLevel(index);
            subresourceRange.layerCount = 1;
            subresourceRange.mipLevelCount = 1;
            if (!copyRegion)
            {
                gfxSize.width = pTexture->getWidth(subresourceRange.mipLevel);
                gfxSize.height = pTexture->getHeight(subresourceRange.mipLevel);
                gfxSize.depth = pTexture->getDepth(subresourceRange.mipLevel);
            }
            gfx::ITextureResource::SubresourceData data = {};
            data.data = dataPtr;
            data.strideY = (int64_t)(gfxSize.width + formatInfo.blockWidth - 1) / formatInfo.blockWidth * formatInfo.blockSizeInBytes;
            data.strideZ = data.strideY * ((gfxSize.height + formatInfo.blockHeight - 1) / formatInfo.blockHeight);
            dataPtr += data.strideZ;
            resourceEncoder->uploadTextureData(static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get()), subresourceRange, gfxOffset, gfxSize, &data, 1);
        }
    }

    CopyContext::ReadTextureTask::SharedPtr CopyContext::ReadTextureTask::create(CopyContext* pCtx, const Texture* pTexture, uint32_t subresourceIndex)
    {
        SharedPtr pThis = SharedPtr(new ReadTextureTask);
        pThis->mpContext = pCtx;
        //Get footprint
        gfx::ITextureResource* srcTexture = static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get());
        gfx::FormatInfo formatInfo;
        gfx::gfxGetFormatInfo(srcTexture->getDesc()->format, &formatInfo);

        auto mipLevel = pTexture->getSubresourceMipLevel(subresourceIndex);
        pThis->mActualRowSize = (pTexture->getWidth(mipLevel) + formatInfo.blockWidth - 1) / formatInfo.blockWidth * formatInfo.blockSizeInBytes;
        pThis->mRowSize = align_to(pThis->mActualRowSize, gfx::ITextureResource::kTexturePitchAlignment);
        uint64_t rowCount =  (pTexture->getHeight(mipLevel) + formatInfo.blockHeight - 1) / formatInfo.blockHeight;
        uint64_t size = pTexture->getDepth(mipLevel) * rowCount * pThis->mRowSize;

        //Create buffer
        pThis->mpBuffer = Buffer::create(size, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);

        //Copy from texture to buffer
        pCtx->resourceBarrier(pTexture, Resource::State::CopySource);
        auto encoder = pCtx->getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::SubresourceRange srcSubresource = {};
        srcSubresource.baseArrayLayer = pTexture->getSubresourceArraySlice(subresourceIndex);
        srcSubresource.mipLevel = mipLevel;
        srcSubresource.layerCount = 1;
        srcSubresource.mipLevelCount = 1;
        encoder->copyTextureToBuffer(
            static_cast<gfx::IBufferResource*>(pThis->mpBuffer->getApiHandle().get()),
            0,
            size,
            srcTexture,
            srcSubresource,
            gfx::ITextureResource::Offset3D(0, 0, 0),
            gfx::ITextureResource::Size{ (int)pTexture->getWidth(mipLevel), (int)pTexture->getHeight(mipLevel), (int)pTexture->getDepth(mipLevel) });
        pCtx->setPendingCommands(true);

        // Create a fence and signal
        pThis->mpFence = GpuFence::create();
        pCtx->flush(false);
        pThis->mpFence->gpuSignal(pCtx->getLowLevelData()->getCommandQueue());
        pThis->mRowCount = (uint32_t)rowCount;
        pThis->mDepth = pTexture->getDepth(mipLevel);
        return pThis;
    }

    std::vector<uint8_t> CopyContext::ReadTextureTask::getData()
    {
        mpFence->syncCpu();
        // Get buffer data
        std::vector<uint8_t> result;
        result.resize((size_t)mRowCount * mActualRowSize);
        uint8_t* pData = reinterpret_cast<uint8_t*>(mpBuffer->map(Buffer::MapType::Read));

        for (uint32_t z = 0; z < mDepth; z++)
        {
            const uint8_t* pSrcZ = pData + z * (size_t)mRowSize * mRowCount;
            uint8_t* pDstZ = result.data() + z * (size_t)mActualRowSize * mRowCount;
            for (uint32_t y = 0; y < mRowCount; y++)
            {
                const uint8_t* pSrc = pSrcZ + y * (size_t)mRowSize;
                uint8_t* pDst = pDstZ + y * (size_t)mActualRowSize;
                memcpy(pDst, pSrc, mActualRowSize);
            }
        }

        mpBuffer->unmap();
        return result;
    }

    bool CopyContext::textureBarrier(const Texture* pTexture, Resource::State newState)
    {
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        bool recorded = false;
        if (pTexture->getGlobalState() != newState)
        {
            gfx::ITextureResource* textureResource = static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get());
            resourceEncoder->textureBarrier(1, &textureResource, getGFXResourceState(pTexture->getGlobalState()), getGFXResourceState(newState));
            mCommandsPending = true;
            recorded = true;
        }
        pTexture->setGlobalState(newState);
        return recorded;
    }

    bool CopyContext::bufferBarrier(const Buffer* pBuffer, Resource::State newState)
    {
        FALCOR_ASSERT(pBuffer);
        if (pBuffer->getCpuAccess() != Buffer::CpuAccess::None) return false;
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        bool recorded = false;
        if (pBuffer->getGlobalState() != newState)
        {
            gfx::IBufferResource* bufferResource = static_cast<gfx::IBufferResource*>(pBuffer->getApiHandle().get());
            resourceEncoder->bufferBarrier(1, &bufferResource, getGFXResourceState(pBuffer->getGlobalState()), getGFXResourceState(newState));
            mCommandsPending = true;
            recorded = true;
        }
        pBuffer->setGlobalState(newState);
        return recorded;
    }

    void CopyContext::apiSubresourceBarrier(const Texture* pTexture, Resource::State newState, Resource::State oldState, uint32_t arraySlice, uint32_t mipLevel)
    {
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        if (pTexture->getGlobalState() != newState)
        {
            gfx::ITextureResource* textureResource = static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get());
            gfx::SubresourceRange subresourceRange = {};
            subresourceRange.baseArrayLayer = arraySlice;
            subresourceRange.mipLevel = mipLevel;
            subresourceRange.layerCount = 1;
            subresourceRange.mipLevelCount = 1;
            resourceEncoder->textureSubresourceBarrier(textureResource, subresourceRange, getGFXResourceState(pTexture->getGlobalState()), getGFXResourceState(newState));
            mCommandsPending = true;
        }
    }

    void CopyContext::uavBarrier(const Resource* pResource)
    {
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();

        if (pResource->getType() == Resource::Type::Buffer)
        {
            gfx::IBufferResource* bufferResource = static_cast<gfx::IBufferResource*>(pResource->getApiHandle().get());
            resourceEncoder->bufferBarrier(1, &bufferResource, gfx::ResourceState::UnorderedAccess, gfx::ResourceState::UnorderedAccess);
        }
        else
        {
            gfx::ITextureResource* textureResource = static_cast<gfx::ITextureResource*>(pResource->getApiHandle().get());
            resourceEncoder->textureBarrier(1, &textureResource, gfx::ResourceState::UnorderedAccess, gfx::ResourceState::UnorderedAccess);
        }
        mCommandsPending = true;
    }

    void CopyContext::copyResource(const Resource* pDst, const Resource* pSrc)
    {
        // Copy from texture to texture or from buffer to buffer.
        FALCOR_ASSERT(pDst->getType() == pSrc->getType());

        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();

        if (pDst->getType() == Resource::Type::Buffer)
        {
            FALCOR_ASSERT(pSrc->getSize() <= pDst->getSize());

            gfx::IBufferResource* srcBuffer = static_cast<gfx::IBufferResource*>(pSrc->getApiHandle().get());
            gfx::IBufferResource* dstBuffer = static_cast<gfx::IBufferResource*>(pDst->getApiHandle().get());

            resourceEncoder->copyBuffer(dstBuffer, 0, srcBuffer, 0, pSrc->getSize());
        }
        else
        {
            gfx::ITextureResource* dstTexture = static_cast<gfx::ITextureResource*>(pDst->getApiHandle().get());
            gfx::ITextureResource* srcTexture = static_cast<gfx::ITextureResource*>(pSrc->getApiHandle().get());
            gfx::SubresourceRange subresourceRange = {};
            resourceEncoder->copyTexture(dstTexture, subresourceRange, gfx::ITextureResource::Offset3D(0, 0, 0),
                srcTexture, subresourceRange, gfx::ITextureResource::Offset3D(0, 0, 0), gfx::ITextureResource::Size{ 0,0,0 });
        }
        mCommandsPending = true;
    }

    void CopyContext::copySubresource(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx)
    {
        copySubresourceRegion(pDst, dstSubresourceIdx, pSrc, srcSubresourceIdx, uint3(0), uint3(0), uint3(-1));
    }

    void CopyContext::copyBufferRegion(const Buffer* pDst, uint64_t dstOffset, const Buffer* pSrc, uint64_t srcOffset, uint64_t numBytes)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::IBufferResource* dstBuffer = static_cast<gfx::IBufferResource*>(pDst->getApiHandle().get());
        gfx::IBufferResource* srcBuffer = static_cast<gfx::IBufferResource*>(pSrc->getApiHandle().get());

        resourceEncoder->copyBuffer(dstBuffer, dstOffset, srcBuffer, pSrc->getGpuAddressOffset() + srcOffset, numBytes);
        mCommandsPending = true;
    }

    void CopyContext::copySubresourceRegion(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx, const uint3& dstOffset, const uint3& srcOffset, const uint3& size)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::ITextureResource* dstTexture = static_cast<gfx::ITextureResource*>(pDst->getApiHandle().get());
        gfx::ITextureResource* srcTexture = static_cast<gfx::ITextureResource*>(pSrc->getApiHandle().get());

        gfx::SubresourceRange dstSubresource = {};
        dstSubresource.baseArrayLayer = pDst->getSubresourceArraySlice(dstSubresourceIdx);
        dstSubresource.layerCount = 1;
        dstSubresource.mipLevel = pDst->getSubresourceMipLevel(dstSubresourceIdx);
        dstSubresource.mipLevelCount = 1;

        gfx::SubresourceRange srcSubresource = {};
        srcSubresource.baseArrayLayer = pSrc->getSubresourceArraySlice(srcSubresourceIdx);
        srcSubresource.layerCount = 1;
        srcSubresource.mipLevel = pSrc->getSubresourceMipLevel(srcSubresourceIdx);
        srcSubresource.mipLevelCount = 1;

        gfx::ITextureResource::Size copySize = { (int)size.x, (int)size.y, (int)size.z };

        if (size.x == glm::uint(-1))
        {
            copySize.width = pSrc->getWidth(srcSubresource.mipLevel) - srcOffset.x;
            copySize.height = pSrc->getHeight(srcSubresource.mipLevel) - srcOffset.y;
            copySize.depth = pSrc->getDepth(srcSubresource.mipLevel) - srcOffset.z;
        }

        resourceEncoder->copyTexture(
            dstTexture,
            dstSubresource,
            gfx::ITextureResource::Offset3D(dstOffset.x, dstOffset.y, dstOffset.z),
            srcTexture,
            srcSubresource,
            gfx::ITextureResource::Offset3D(srcOffset.x, srcOffset.y, srcOffset.z),
            copySize);
        mCommandsPending = true;
    }
}
