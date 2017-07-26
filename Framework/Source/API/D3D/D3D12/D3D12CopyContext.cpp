/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "API/Device.h"
#include "API/Buffer.h"
#include <queue>
#include "D3D12Resource.h"
#include "LowLevel/D3D12DescriptorData.h"

namespace Falcor
{
    void CopyContext::bindDescriptorHeaps()
    {
        const DescriptorPool* pGpuPool = gpDevice->getGpuDescriptorPool().get();
        const DescriptorPool::ApiData* pData = pGpuPool->getApiData();
        ID3D12DescriptorHeap* pHeaps[arraysize(pData->pHeaps)];
        uint32_t heapCount = 0;
        for (uint32_t i = 0; i < arraysize(pData->pHeaps); i++)
        {
            if (pData->pHeaps[i])
            {
                pHeaps[heapCount] = pData->pHeaps[i]->getApiHandle();
                heapCount++;
            }
        }
        mpLowLevelData->getCommandList()->SetDescriptorHeaps(heapCount, pHeaps);
    }
    
    void copySubresourceData(const D3D12_SUBRESOURCE_DATA& srcData, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& dstFootprint, uint8_t* pDstStart, uint64_t rowSize, uint64_t rowsToCopy)
    {
        const uint8_t* pSrc = (uint8_t*)srcData.pData;
        uint8_t* pDst = pDstStart + dstFootprint.Offset;
        const D3D12_SUBRESOURCE_FOOTPRINT& dstData = dstFootprint.Footprint;

        for (uint32_t z = 0; z < dstData.Depth; z++)
        {
            uint8_t* pDstSlice = pDst + rowsToCopy * dstData.RowPitch * z;
            const uint8_t* pSrcSlice = pSrc + srcData.SlicePitch * z;

            for (uint32_t y = 0; y < rowsToCopy; y++)
            {
                const uint8_t* pSrcRow = pSrcSlice + srcData.RowPitch * y;
                uint8_t* pDstRow = pDstSlice + dstData.RowPitch* y;
                memcpy(pDstRow, pSrcRow, rowSize);
            }
        }
    }
    
    void CopyContext::updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData)
    {
        mCommandsPending = true;

        uint32_t arraySize = (pTexture->getType() == Texture::Type::TextureCube) ? pTexture->getArraySize() * 6 : pTexture->getArraySize();
        assert(firstSubresource + subresourceCount <= arraySize * pTexture->getMipCount());

        ID3D12Device* pDevice = gpDevice->getApiHandle();

        // Get the footprint
        D3D12_RESOURCE_DESC texDesc = pTexture->getApiHandle()->GetDesc();
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprint(subresourceCount);
        std::vector<uint32_t> rowCount(subresourceCount);
        std::vector<uint64_t> rowSize(subresourceCount);
        uint64_t size;
        pDevice->GetCopyableFootprints(&texDesc, firstSubresource, subresourceCount, 0, footprint.data(), rowCount.data(), rowSize.data(), &size);

        // Allocate a buffer on the upload heap
        Buffer::SharedPtr pBuffer = Buffer::create(size, Buffer::BindFlags::None, Buffer::CpuAccess::Write, nullptr);
        // Map the buffer
        uint8_t* pDst = (uint8_t*)pBuffer->map(Buffer::MapType::WriteDiscard);
        ID3D12ResourcePtr pResource = pBuffer->getApiHandle();

        // Get the offset from the beginning of the resource
        uint64_t offset = pBuffer->getGpuAddressOffset();
        resourceBarrier(pTexture, Resource::State::CopyDest);

        const uint8_t* pSrc = (uint8_t*)pData;
        for (uint32_t s = 0; s < subresourceCount; s++)
        {
            uint32_t physicalWidth = footprint[s].Footprint.Width / getFormatWidthCompressionRatio(pTexture->getFormat());
            uint32_t physicalHeight = footprint[s].Footprint.Height / getFormatHeightCompressionRatio(pTexture->getFormat());

            D3D12_SUBRESOURCE_DATA src;
            src.pData = pSrc;
            src.RowPitch = physicalWidth * getFormatBytesPerBlock(pTexture->getFormat());
            src.SlicePitch = src.RowPitch * physicalHeight;
            copySubresourceData(src, footprint[s], pDst, rowSize[s], rowCount[s]);
            pSrc = (uint8_t*)pSrc + footprint[s].Footprint.Depth * src.SlicePitch;

            // Dispatch a command
            footprint[s].Offset += offset;
            uint32_t subresource = s + firstSubresource;
            D3D12_TEXTURE_COPY_LOCATION dstLoc = { pTexture->getApiHandle(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, subresource };
            D3D12_TEXTURE_COPY_LOCATION srcLoc = { pResource, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint[s] };
            mpLowLevelData->getCommandList()->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        }

        pBuffer->unmap();
    }

    void CopyContext::updateTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex, const void* pData)
    {
        mCommandsPending = true;
        updateTextureSubresources(pTexture, subresourceIndex, 1, pData);
    }

    std::vector<uint8> CopyContext::readTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex)
    {
        //Get footprint
        D3D12_RESOURCE_DESC texDesc = pTexture->getApiHandle()->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        uint32_t rowCount;
        uint64_t rowSize;
        uint64_t size;
        ID3D12Device* pDevice = gpDevice->getApiHandle();
        pDevice->GetCopyableFootprints(&texDesc, subresourceIndex, 1, 0, &footprint, &rowCount, &rowSize, &size);

        //Create buffer 
        Buffer::SharedPtr pBuffer = Buffer::create(size, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);

        //Copy from texture to buffer
        D3D12_TEXTURE_COPY_LOCATION srcLoc = { pTexture->getApiHandle(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, subresourceIndex };
        D3D12_TEXTURE_COPY_LOCATION dstLoc = { pBuffer->getApiHandle(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint };
        resourceBarrier(pTexture, Resource::State::CopySource);
        mpLowLevelData->getCommandList()->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        flush(true);

        //Get buffer data
        std::vector<uint8> result;
		uint32_t actualRowSize = footprint.Footprint.Width * getFormatBytesPerBlock(pTexture->getFormat());
        result.resize(rowCount * actualRowSize);
        uint8* pData = reinterpret_cast<uint8*>(pBuffer->map(Buffer::MapType::Read));

        for(uint32_t z = 0 ; z < footprint.Footprint.Depth ; z++)
        {
            const uint8_t* pSrcZ = pData + z * footprint.Footprint.RowPitch * rowCount;
            uint8_t* pDstZ = result.data() + z * actualRowSize * rowCount;
            for (uint32_t y = 0; y < rowCount; y++)
            {
                const uint8_t* pSrc = pSrcZ + y *  footprint.Footprint.RowPitch;
                uint8_t* pDst = pDstZ + y * actualRowSize;
                memcpy(pDst, pSrc, actualRowSize);
            }
        }

        pBuffer->unmap();
        return result;
    }
    
    void CopyContext::resourceBarrier(const Resource* pResource, Resource::State newState)
    {
        // If the resource is a buffer with CPU access, no need to do anything
        const Buffer* pBuffer = dynamic_cast<const Buffer*>(pResource);
        if (pBuffer && pBuffer->getCpuAccess() != Buffer::CpuAccess::None) return;

        if (pResource->getState() != newState)
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = pResource->getApiHandle();
            barrier.Transition.StateBefore = getD3D12ResourceState(pResource->getState());
            barrier.Transition.StateAfter = getD3D12ResourceState(newState);
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;   // OPTME: Need to do that only for the subresources we will actually use

            mpLowLevelData->getCommandList()->ResourceBarrier(1, &barrier);
            mCommandsPending = true;
            pResource->mState = newState;
        }
    }

    void CopyContext::copyResource(const Resource* pDst, const Resource* pSrc)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);
        mpLowLevelData->getCommandList()->CopyResource(pDst->getApiHandle(), pSrc->getApiHandle());
        mCommandsPending = true;
    }

    void CopyContext::copySubresource(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        D3D12_TEXTURE_COPY_LOCATION pSrcCopyLoc;
        D3D12_TEXTURE_COPY_LOCATION pDstCopyLoc;

        pDstCopyLoc.pResource = pDst->getApiHandle();
        pDstCopyLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        pDstCopyLoc.SubresourceIndex = dstSubresourceIdx;

        pSrcCopyLoc.pResource = pSrc->getApiHandle();
        pSrcCopyLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        pSrcCopyLoc.SubresourceIndex = srcSubresourceIdx;

        mpLowLevelData->getCommandList()->CopyTextureRegion(&pDstCopyLoc, 0, 0, 0, &pSrcCopyLoc, NULL);
        mCommandsPending = true;
    }

    void CopyContext::copyBufferRegion(const Buffer* pDst, uint64_t dstOffset, const Buffer* pSrc, uint64_t srcOffset, uint64_t numBytes)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);
        mpLowLevelData->getCommandList()->CopyBufferRegion(pDst->getApiHandle(), dstOffset, pSrc->getApiHandle(), pSrc->getGpuAddressOffset() + srcOffset, numBytes);    
        mCommandsPending = true;
    }
}