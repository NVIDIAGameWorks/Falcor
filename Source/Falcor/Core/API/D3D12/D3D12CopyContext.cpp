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
#include "D3D12Resource.h"
#include "Core/API/Device.h"
#include "Core/API/Texture.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Core/API/Shared/D3D12DescriptorData.h"
#include "Utils/Math/Common.h"

namespace Falcor
{
    void CopyContext::bindDescriptorHeaps()
    {
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
                uint8_t* pDstRow = pDstSlice + dstData.RowPitch * y;
                memcpy(pDstRow, pSrcRow, rowSize);
            }
        }
    }

    void CopyContext::updateTextureSubresources(const Texture* pTexture, uint32_t firstSubresource, uint32_t subresourceCount, const void* pData, const uint3& offset, const uint3& size)
    {
        bool copyRegion = (offset != uint3(0)) || (size != uint3(-1));
        FALCOR_ASSERT(subresourceCount == 1 || (copyRegion == false));

        mCommandsPending = true;

        uint32_t arraySize = (pTexture->getType() == Texture::Type::TextureCube) ? pTexture->getArraySize() * 6 : pTexture->getArraySize();
        FALCOR_ASSERT(firstSubresource + subresourceCount <= arraySize * pTexture->getMipCount());

        // Get the footprint
        D3D12_RESOURCE_DESC texDesc = pTexture->getApiHandle()->GetDesc();
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprint(subresourceCount);
        std::vector<uint32_t> rowCount(subresourceCount);
        std::vector<uint64_t> rowSize(subresourceCount);
        uint64_t bufferSize;

        if (copyRegion)
        {
            footprint[0].Offset = 0;
            footprint[0].Footprint.Format = getDxgiFormat(pTexture->getFormat());
            uint32_t mipLevel = pTexture->getSubresourceMipLevel(firstSubresource);
            footprint[0].Footprint.Width = (size.x == -1) ? pTexture->getWidth(mipLevel) - offset.x : size.x;
            footprint[0].Footprint.Height = (size.y == -1) ? pTexture->getHeight(mipLevel) - offset.y : size.y;
            footprint[0].Footprint.Depth = (size.z == -1) ? pTexture->getDepth(mipLevel) - offset.z : size.z;
            footprint[0].Footprint.RowPitch = align_to((uint32_t)D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, footprint[0].Footprint.Width * getFormatBytesPerBlock(pTexture->getFormat()));
            rowCount[0] = footprint[0].Footprint.Height;
            rowSize[0] = footprint[0].Footprint.RowPitch;
            bufferSize = rowSize[0] * rowCount[0] * footprint[0].Footprint.Depth;
        }
        else
        {
            ID3D12Device* pDevice = gpDevice->getApiHandle();
            pDevice->GetCopyableFootprints(&texDesc, firstSubresource, subresourceCount, 0, footprint.data(), rowCount.data(), rowSize.data(), &bufferSize);
        }

        // Allocate a buffer on the upload heap
        Buffer::SharedPtr pBuffer = Buffer::create(bufferSize, Buffer::BindFlags::None, Buffer::CpuAccess::Write, nullptr);
        // Map the buffer
        uint8_t* pDst = (uint8_t*)pBuffer->map(Buffer::MapType::WriteDiscard);
        ID3D12ResourcePtr pResource = pBuffer->getApiHandle();

        // Get the offset from the beginning of the resource
        uint64_t vaOffset = pBuffer->getGpuAddressOffset();
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
            footprint[s].Offset += vaOffset;
            uint32_t subresource = s + firstSubresource;
            D3D12_TEXTURE_COPY_LOCATION dstLoc = { pTexture->getApiHandle(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, subresource };
            D3D12_TEXTURE_COPY_LOCATION srcLoc = { pResource, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint[s] };

            mpLowLevelData->getCommandList()->CopyTextureRegion(&dstLoc, offset.x, offset.y, offset.z, &srcLoc, nullptr);
        }

        pBuffer->unmap();
    }

    CopyContext::ReadTextureTask::SharedPtr CopyContext::ReadTextureTask::create(CopyContext* pCtx, const Texture* pTexture, uint32_t subresourceIndex)
    {
        SharedPtr pThis = SharedPtr(new ReadTextureTask);
        pThis->mpContext = pCtx;
        //Get footprint
        D3D12_RESOURCE_DESC texDesc = pTexture->getApiHandle()->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint = pThis->mFootprint;
        uint64_t rowSize;
        uint64_t size;
        ID3D12Device* pDevice = gpDevice->getApiHandle();
        pDevice->GetCopyableFootprints(&texDesc, subresourceIndex, 1, 0, &footprint, &pThis->mRowCount, &rowSize, &size);

        //Create buffer
        pThis->mpBuffer = Buffer::create(size, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);

        //Copy from texture to buffer
        D3D12_TEXTURE_COPY_LOCATION srcLoc = { pTexture->getApiHandle(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, subresourceIndex };
        D3D12_TEXTURE_COPY_LOCATION dstLoc = { pThis->mpBuffer->getApiHandle(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint };
        pCtx->resourceBarrier(pTexture, Resource::State::CopySource);
        pCtx->getLowLevelData()->getCommandList()->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        pCtx->setPendingCommands(true);

        // Create a fence and signal
        pThis->mpFence = GpuFence::create();
        pCtx->flush(false);
        pThis->mpFence->gpuSignal(pCtx->getLowLevelData()->getCommandQueue());
        pThis->mTextureFormat = pTexture->getFormat();

        return pThis;
    }

    std::vector<uint8_t> CopyContext::ReadTextureTask::getData()
    {
        mpFence->syncCpu();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint = mFootprint;

        // Calculate row size. GPU pitch can be different because it is aligned to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
        FALCOR_ASSERT(footprint.Footprint.Width % getFormatWidthCompressionRatio(mTextureFormat) == 0); // Should divide evenly
        uint32_t actualRowSize = (footprint.Footprint.Width / getFormatWidthCompressionRatio(mTextureFormat)) * getFormatBytesPerBlock(mTextureFormat);

        // Get buffer data
        std::vector<uint8_t> result;
        result.resize(mRowCount * actualRowSize * footprint.Footprint.Depth);
        uint8_t* pData = reinterpret_cast<uint8_t*>(mpBuffer->map(Buffer::MapType::Read));

        for (uint32_t z = 0; z < footprint.Footprint.Depth; z++)
        {
            const uint8_t* pSrcZ = pData + z * footprint.Footprint.RowPitch * mRowCount;
            uint8_t* pDstZ = result.data() + z * actualRowSize * mRowCount;
            for (uint32_t y = 0; y < mRowCount; y++)
            {
                const uint8_t* pSrc = pSrcZ + y * footprint.Footprint.RowPitch;
                uint8_t* pDst = pDstZ + y * actualRowSize;
                memcpy(pDst, pSrc, actualRowSize);
            }
        }

        mpBuffer->unmap();
        return result;
    }

    static void d3d12ResourceBarrier(const Resource* pResource, Resource::State newState, Resource::State oldState, uint32_t subresourceIndex, ID3D12GraphicsCommandList* pCmdList)
    {
        D3D12_RESOURCE_BARRIER barrier;
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = pResource->getApiHandle();
        barrier.Transition.StateBefore = getD3D12ResourceState(oldState);
        barrier.Transition.StateAfter = getD3D12ResourceState(newState);
        barrier.Transition.Subresource = subresourceIndex;

        // Check that resource has required bind flags for before/after state to be supported
        D3D12_RESOURCE_STATES beforeOrAfterState = barrier.Transition.StateBefore | barrier.Transition.StateAfter;
        if (beforeOrAfterState & D3D12_RESOURCE_STATE_RENDER_TARGET)
        {
            FALCOR_ASSERT(is_set(pResource->getBindFlags(), Resource::BindFlags::RenderTarget));
        }

        if (beforeOrAfterState & (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE))
        {
            FALCOR_ASSERT(is_set(pResource->getBindFlags(), Resource::BindFlags::ShaderResource));
        }

        if (beforeOrAfterState & D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            FALCOR_ASSERT(is_set(pResource->getBindFlags(), Resource::BindFlags::UnorderedAccess));
        }

        pCmdList->ResourceBarrier(1, &barrier);
    }

    static bool d3d12GlobalResourceBarrier(const Resource* pResource, Resource::State newState, ID3D12GraphicsCommandList* pCmdList)
    {
        if (pResource->getGlobalState() != newState)
        {
            d3d12ResourceBarrier(pResource, newState, pResource->getGlobalState(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, pCmdList);
            return true;
        }
        return false;
    }

    bool CopyContext::textureBarrier(const Texture* pTexture, Resource::State newState)
    {
        FALCOR_ASSERT(pTexture);
        bool recorded = d3d12GlobalResourceBarrier(pTexture, newState, mpLowLevelData->getCommandList());
        pTexture->setGlobalState(newState);
        mCommandsPending = mCommandsPending || recorded;
        return recorded;
    }

    bool CopyContext::bufferBarrier(const Buffer* pBuffer, Resource::State newState)
    {
        FALCOR_ASSERT(pBuffer);
        if (pBuffer->getCpuAccess() != Buffer::CpuAccess::None) return false;
        bool recorded = d3d12GlobalResourceBarrier(pBuffer, newState, mpLowLevelData->getCommandList());
        pBuffer->setGlobalState(newState);
        mCommandsPending = mCommandsPending || recorded;
        return recorded;
    }

    void CopyContext::apiSubresourceBarrier(const Texture* pTexture, Resource::State newState, Resource::State oldState, uint32_t arraySlice, uint32_t mipLevel)
    {
        uint32_t subresourceIndex = pTexture->getSubresourceIndex(arraySlice, mipLevel);
        d3d12ResourceBarrier(pTexture, newState, oldState, subresourceIndex, mpLowLevelData->getCommandList());
    }

    void CopyContext::uavBarrier(const Resource* pResource)
    {
        D3D12_RESOURCE_BARRIER barrier;
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.UAV.pResource = pResource->getApiHandle();

        // Check that resource has required bind flags for UAV barrier to be supported
        static const Resource::BindFlags reqFlags = Resource::BindFlags::UnorderedAccess | Resource::BindFlags::AccelerationStructure;
        FALCOR_ASSERT(is_set(pResource->getBindFlags(), reqFlags));
        mpLowLevelData->getCommandList()->ResourceBarrier(1, &barrier);
        mCommandsPending = true;
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

        mCommandsPending = true;

        // Allocate a buffer on the upload heap
        Buffer::SharedPtr pUploadBuffer = Buffer::create(numBytes, Buffer::BindFlags::None, Buffer::CpuAccess::Write, pData);

        copyBufferRegion(pBuffer, offset, pUploadBuffer.get(), 0, numBytes);
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

    void CopyContext::copySubresourceRegion(const Texture* pDst, uint32_t dstSubresource, const Texture* pSrc, uint32_t srcSubresource, const uint3& dstOffset, const uint3& srcOffset, const uint3& size)
    {
        resourceBarrier(pDst, Resource::State::CopyDest);
        resourceBarrier(pSrc, Resource::State::CopySource);

        D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = dstSubresource;
        dstLoc.pResource = pDst->getApiHandle();

        D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = srcSubresource;
        srcLoc.pResource = pSrc->getApiHandle();

        D3D12_BOX box;
        box.left = srcOffset.x;
        box.top = srcOffset.y;
        box.front = srcOffset.z;
        uint32_t mipLevel = pSrc->getSubresourceMipLevel(dstSubresource);
        box.right = (size.x == -1) ? pSrc->getWidth(mipLevel) - box.left : size.x;
        box.bottom = (size.y == -1) ? pSrc->getHeight(mipLevel) - box.top : size.y;
        box.back = (size.z == -1) ? pSrc->getDepth(mipLevel) - box.front : size.z;

        mpLowLevelData->getCommandList()->CopyTextureRegion(&dstLoc, dstOffset.x, dstOffset.y, dstOffset.z, &srcLoc, &box);

        mCommandsPending = true;
    }
}
