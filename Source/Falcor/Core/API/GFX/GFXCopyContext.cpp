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
#include "Core/API/CopyContext.h"
#include "GFXLowLevelContextApiData.h"
#include "GFXFormats.h"
#include "GFXResource.h"
#include "Core/API/Device.h"
#include "Core/API/Texture.h"
#if FALCOR_HAS_D3D12
#include "Core/API/Shared/D3D12DescriptorPool.h"
#include "Core/API/Shared/D3D12DescriptorData.h"
#endif
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"

namespace Falcor
{
    void CopyContext::bindDescriptorHeaps()
    {
    }

    void CopyContext::bindCustomGPUDescriptorPool()
    {
#if FALCOR_HAS_D3D12
        const D3D12DescriptorPool* pGpuPool = gpDevice->getD3D12GpuDescriptorPool().get();
        const D3D12DescriptorPool::ApiData* pData = pGpuPool->getApiData();
        ID3D12DescriptorHeap* pHeaps[D3D12DescriptorPool::ApiData::kHeapCount];
        uint32_t heapCount = 0;
        for (uint32_t i = 0; i < std::size(pData->pHeaps); i++)
        {
            if (pData->pHeaps[i])
            {
                pHeaps[heapCount] = pData->pHeaps[i]->getApiHandle();
                heapCount++;
            }
        }
        mpLowLevelData->getD3D12CommandList()->SetDescriptorHeaps(heapCount, pHeaps);
#endif
    }

    void CopyContext::unbindCustomGPUDescriptorPool()
    {
#if FALCOR_HAS_D3D12
        ComPtr<gfx::ICommandBufferD3D12> d3d12CommandBuffer;
        mpLowLevelData->getApiData()->pCommandBuffer->queryInterface(SlangUUID SLANG_UUID_ICommandBufferD3D12, reinterpret_cast<void**>(d3d12CommandBuffer.writeRef()));
        d3d12CommandBuffer->invalidateDescriptorHeapBinding();
#endif
    }

    void CopyContext::updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData, const uint3& offset, const uint3& size)
    {
        resourceBarrier(pTexture, Resource::State::CopyDest);

        bool copyRegion = (offset != uint3(0)) || (size != uint3(-1));
        FALCOR_ASSERT(subresourceCount == 1 || (copyRegion == false));
        uint8_t* dataPtr = (uint8_t*)pData;
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::ITextureResource::Offset3D gfxOffset = { static_cast<gfx::GfxIndex>(offset.x), static_cast<gfx::GfxIndex>(offset.y), static_cast<gfx::GfxIndex>(offset.z) };
        gfx::ITextureResource::Extents gfxSize = { static_cast<gfx::GfxCount>(size.x), static_cast<gfx::GfxCount>(size.y), static_cast<gfx::GfxCount>(size.z) };
        gfx::FormatInfo formatInfo = {};
        gfx::gfxGetFormatInfo(getGFXFormat(pTexture->getFormat()), &formatInfo);
        for (uint32_t index = firstSubresource; index < firstSubresource + subresourceCount; index++)
        {
            gfx::SubresourceRange subresourceRange = {};
            subresourceRange.baseArrayLayer = static_cast<gfx::GfxIndex>(pTexture->getSubresourceArraySlice(index));
            subresourceRange.mipLevel = static_cast<gfx::GfxIndex>(pTexture->getSubresourceMipLevel(index));
            subresourceRange.layerCount = 1;
            subresourceRange.mipLevelCount = 1;
            if (!copyRegion)
            {
                gfxSize.width = align_to(formatInfo.blockWidth, static_cast<gfx::GfxCount>(pTexture->getWidth(subresourceRange.mipLevel)));
                gfxSize.height = align_to(formatInfo.blockHeight, static_cast<gfx::GfxCount>(pTexture->getHeight(subresourceRange.mipLevel)));
                gfxSize.depth = static_cast<gfx::GfxCount>(pTexture->getDepth(subresourceRange.mipLevel));
            }
            gfx::ITextureResource::SubresourceData data = {};
            data.data = dataPtr;
            data.strideY = static_cast<int64_t>(gfxSize.width) / formatInfo.blockWidth * formatInfo.blockSizeInBytes;
            data.strideZ = data.strideY * (gfxSize.height / formatInfo.blockHeight);
            dataPtr += data.strideZ * gfxSize.depth;
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
        pThis->mActualRowSize = uint32_t((pTexture->getWidth(mipLevel) + formatInfo.blockWidth - 1) / formatInfo.blockWidth * formatInfo.blockSizeInBytes);
        size_t rowAlignment = 1;
        gpDevice->getApiHandle()->getTextureRowAlignment(&rowAlignment);
        pThis->mRowSize = align_to(static_cast<uint32_t>(rowAlignment), pThis->mActualRowSize);
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
            pThis->mpBuffer->getGpuAddressOffset(),
            size,
            pThis->mRowSize,
            srcTexture,
            gfx::ResourceState::CopySource,
            srcSubresource,
            gfx::ITextureResource::Offset3D(0, 0, 0),
            gfx::ITextureResource::Extents{ static_cast<gfx::GfxIndex>(pTexture->getWidth(mipLevel)), static_cast<gfx::GfxIndex>(pTexture->getHeight(mipLevel)), static_cast<gfx::GfxIndex>(pTexture->getDepth(mipLevel)) });
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
        bool recorded = false;
        if (pBuffer->getGlobalState() != newState)
        {
            auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
            gfx::IBufferResource* bufferResource = static_cast<gfx::IBufferResource*>(pBuffer->getApiHandle().get());
            resourceEncoder->bufferBarrier(1, &bufferResource, getGFXResourceState(pBuffer->getGlobalState()), getGFXResourceState(newState));
            pBuffer->setGlobalState(newState);
            mCommandsPending = true;
            recorded = true;
        }
        return recorded;
    }

    void CopyContext::apiSubresourceBarrier(const Texture* pTexture, Resource::State newState, Resource::State oldState, uint32_t arraySlice, uint32_t mipLevel)
    {
        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        auto subresourceState = pTexture->getSubresourceState(arraySlice, mipLevel);
        if (subresourceState != newState)
        {
            gfx::ITextureResource* textureResource = static_cast<gfx::ITextureResource*>(pTexture->getApiHandle().get());
            gfx::SubresourceRange subresourceRange = {};
            subresourceRange.baseArrayLayer = arraySlice;
            subresourceRange.mipLevel = mipLevel;
            subresourceRange.layerCount = 1;
            subresourceRange.mipLevelCount = 1;
            resourceEncoder->textureSubresourceBarrier(textureResource, subresourceRange, getGFXResourceState(subresourceState), getGFXResourceState(newState));
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

            resourceEncoder->copyBuffer(
                dstBuffer,
                static_cast<const Buffer*>(pDst)->getGpuAddressOffset(),
                srcBuffer,
                static_cast<const Buffer*>(pSrc)->getGpuAddressOffset(),
                pSrc->getSize());
        }
        else
        {
            gfx::ITextureResource* dstTexture = static_cast<gfx::ITextureResource*>(pDst->getApiHandle().get());
            gfx::ITextureResource* srcTexture = static_cast<gfx::ITextureResource*>(pSrc->getApiHandle().get());
            gfx::SubresourceRange subresourceRange = {};
            resourceEncoder->copyTexture(dstTexture, gfx::ResourceState::CopyDestination, subresourceRange, gfx::ITextureResource::Offset3D(0, 0, 0),
                srcTexture, gfx::ResourceState::CopySource, subresourceRange, gfx::ITextureResource::Offset3D(0, 0, 0), gfx::ITextureResource::Extents{ 0,0,0 });
        }
        mCommandsPending = true;
    }

    void CopyContext::copySubresource(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx)
    {
        copySubresourceRegion(pDst, dstSubresourceIdx, pSrc, srcSubresourceIdx, uint3(0), uint3(0), uint3(-1));
    }

    void CopyContext::updateBuffer(const Buffer* pBuffer, const void* pData, size_t offset, size_t numBytes)
    {
        if (numBytes == 0)
        {
            numBytes = pBuffer->getSize() - offset;
        }

        if (pBuffer->adjustSizeOffsetParams(numBytes, offset) == false)
        {
            logWarning("CopyContext::updateBuffer() - size and offset are invalid. Nothing to update.");
            return;
        }

        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        resourceEncoder->uploadBufferData(
            static_cast<gfx::IBufferResource*>(pBuffer->getApiHandle().get()),
            pBuffer->getGpuAddressOffset() + offset,
            numBytes,
            (void*)pData);

        mCommandsPending = true;
    }

    void CopyContext::copyBufferRegion(const Buffer* pDst, uint64_t dstOffset, const Buffer* pSrc, uint64_t srcOffset, uint64_t numBytes)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        auto resourceEncoder = getLowLevelData()->getApiData()->getResourceCommandEncoder();
        gfx::IBufferResource* dstBuffer = static_cast<gfx::IBufferResource*>(pDst->getApiHandle().get());
        gfx::IBufferResource* srcBuffer = static_cast<gfx::IBufferResource*>(pSrc->getApiHandle().get());

        resourceEncoder->copyBuffer(dstBuffer, pDst->getGpuAddressOffset() + dstOffset, srcBuffer, pSrc->getGpuAddressOffset() + srcOffset, numBytes);
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

        gfx::ITextureResource::Extents copySize = { (int)size.x, (int)size.y, (int)size.z };

        if (size.x == glm::uint(-1))
        {
            copySize.width = pSrc->getWidth(srcSubresource.mipLevel) - srcOffset.x;
            copySize.height = pSrc->getHeight(srcSubresource.mipLevel) - srcOffset.y;
            copySize.depth = pSrc->getDepth(srcSubresource.mipLevel) - srcOffset.z;
        }

        resourceEncoder->copyTexture(
            dstTexture,
            gfx::ResourceState::CopyDestination,
            dstSubresource,
            gfx::ITextureResource::Offset3D(dstOffset.x, dstOffset.y, dstOffset.z),
            srcTexture,
            gfx::ResourceState::CopySource,
            srcSubresource,
            gfx::ITextureResource::Offset3D(srcOffset.x, srcOffset.y, srcOffset.z),
            copySize);
        mCommandsPending = true;
    }
}
