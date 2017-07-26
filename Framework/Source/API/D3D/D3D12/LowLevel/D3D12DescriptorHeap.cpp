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
#include "D3D12DescriptorHeap.h"
#include "API/Device.h"

namespace Falcor
{
    D3D12DescriptorHeap::D3D12DescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t chunkCount) : mChunkCount(chunkCount), mType (type)
    {
		ID3D12DevicePtr pDevice = gpDevice->getApiHandle();
        mDescriptorSize = pDevice->GetDescriptorHandleIncrementSize(type);
    }

    D3D12DescriptorHeap::~D3D12DescriptorHeap() = default;

    D3D12DescriptorHeap::SharedPtr D3D12DescriptorHeap::create(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t descCount, bool shaderVisible)
    {
		ID3D12DevicePtr pDevice = gpDevice->getApiHandle();

        uint32_t chunkCount = (descCount + kDescPerChunk - 1) / kDescPerChunk;
        D3D12DescriptorHeap::SharedPtr pHeap = SharedPtr(new D3D12DescriptorHeap(type, chunkCount));
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};

        desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.Type = type;
        desc.NumDescriptors = chunkCount * kDescPerChunk;
        if(FAILED(pDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pHeap->mApiHandle))))
        {
            logError("Can't create descriptor heap");
            return nullptr;
        }

        pHeap->mCpuHeapStart = pHeap->mApiHandle->GetCPUDescriptorHandleForHeapStart();
        pHeap->mGpuHeapStart = pHeap->mApiHandle->GetGPUDescriptorHandleForHeapStart();
        return pHeap;
    }

    template<typename HandleType>
    HandleType getHandleCommon(HandleType base, uint32_t index, uint32_t descSize)
    {
        base.ptr += descSize * index;
        return base;
    }

    D3D12DescriptorHeap::CpuHandle D3D12DescriptorHeap::getCpuHandle(uint32_t index) const
    {
        return getHandleCommon(mCpuHeapStart, index, mDescriptorSize);
    }

    D3D12DescriptorHeap::GpuHandle D3D12DescriptorHeap::getGpuHandle(uint32_t index) const
    {
        return getHandleCommon(mGpuHeapStart, index, mDescriptorSize);
    }

    D3D12DescriptorHeap::Allocation::SharedPtr D3D12DescriptorHeap::allocateDescriptors(uint32_t count)
    {
        if (setupCurrentChunk(count) == false) return nullptr;

        if (mpCurrentChunk->chunkCount * kDescPerChunk - mpCurrentChunk->currentDesc < count)
        {
            return nullptr;
        }

        Allocation::SharedPtr pAlloc = Allocation::create(shared_from_this(), mpCurrentChunk->getCurrentAbsoluteIndex(), count, mpCurrentChunk);

        // Update the chunk
        mpCurrentChunk->allocCount++;
        mpCurrentChunk->currentDesc += count;
        return pAlloc;
    }

    bool D3D12DescriptorHeap::setupCurrentChunk(uint32_t descCount)
    {
        if (mpCurrentChunk)
        {
            // Check if the current chunk has enough space
            if (mpCurrentChunk->chunkCount * kDescPerChunk - mpCurrentChunk->currentDesc >= descCount) return true;

            if (mpCurrentChunk->allocCount == 0)
            {
                mpCurrentChunk->reset();
                return true;
            }
        }

        // Need a new chunk
        uint32_t chunkCount = (descCount + kDescPerChunk - 1) / kDescPerChunk;

        // TODO: Optimize it for the case that chunkCount > 1 - mFreeChunks can be sorted by offset to find contiguous chunks
        if (chunkCount == 1 && (mFreeChunks.empty() == false))
        {
            mpCurrentChunk = mFreeChunks.front();
            mFreeChunks.pop();
            mpCurrentChunk->reset();
            return true;
        }

        // No free chunks. Allocate
        if (mAllocatedChunks + chunkCount > mChunkCount)
        {
            return false;
        }

        mpCurrentChunk = Chunk::SharedPtr(new Chunk(mAllocatedChunks, chunkCount));
        mAllocatedChunks += chunkCount;
        return true;
    }
    
    void D3D12DescriptorHeap::releaseChunk(Chunk::SharedPtr pChunk)
    {
        pChunk->allocCount--;
        if(pChunk->allocCount == 0 && (pChunk != mpCurrentChunk))
        {
            mFreeChunks.push(pChunk);
        }
    }

    D3D12DescriptorHeap::Allocation::SharedPtr D3D12DescriptorHeap::Allocation::create(D3D12DescriptorHeap::SharedPtr pHeap, uint32_t baseIndex, uint32_t descCount, std::shared_ptr<Chunk> pChunk)
    {
        return SharedPtr(new Allocation(pHeap, baseIndex, descCount, pChunk));
    }

    D3D12DescriptorHeap::Allocation::Allocation(D3D12DescriptorHeap::SharedPtr pHeap, uint32_t baseIndex, uint32_t descCount, std::shared_ptr<Chunk> pChunk) :
        mpHeap(pHeap), mBaseIndex(baseIndex), mDescCount(descCount), mpChunk(pChunk) {}

    D3D12DescriptorHeap::Allocation::~Allocation()
    {
        mpHeap->releaseChunk(mpChunk);
    }
}
