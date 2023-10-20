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
#include "D3D12DescriptorHeap.h"
#include "Core/API/Device.h"
#include "Core/API/NativeHandleTraits.h"

namespace Falcor
{
D3D12DescriptorHeap::D3D12DescriptorHeap(Device* pDevice, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t chunkCount, bool shaderVisible)
    : mMaxChunkCount(chunkCount), mType(type), mShaderVisible(shaderVisible)
{
    ID3D12Device* pD3D12Device = pDevice->getNativeHandle().as<ID3D12Device*>();

    mDescriptorSize = pD3D12Device->GetDescriptorHandleIncrementSize(type);
}

D3D12DescriptorHeap::~D3D12DescriptorHeap() = default;

ref<D3D12DescriptorHeap> D3D12DescriptorHeap::create(
    Device* pDevice,
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    uint32_t descCount,
    bool shaderVisible
)
{
    FALCOR_ASSERT(pDevice);
    pDevice->requireD3D12();
    ID3D12Device* pD3D12Device = pDevice->getNativeHandle().as<ID3D12Device*>();

    uint32_t chunkCount = (descCount + kDescPerChunk - 1) / kDescPerChunk;
    ref<D3D12DescriptorHeap> pHeap = ref<D3D12DescriptorHeap>(new D3D12DescriptorHeap(pDevice, type, chunkCount, shaderVisible));
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};

    desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    desc.Type = type;
    desc.NumDescriptors = chunkCount * kDescPerChunk;
    if (FAILED(pD3D12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pHeap->mApiHandle))))
    {
        FALCOR_THROW("Failed to create descriptor heap");
    }

    pHeap->mCpuHeapStart = pHeap->mApiHandle->GetCPUDescriptorHandleForHeapStart();
    if (shaderVisible)
        pHeap->mGpuHeapStart = pHeap->mApiHandle->GetGPUDescriptorHandleForHeapStart();

    return pHeap;
}

D3D12DescriptorHeap::GpuHandle D3D12DescriptorHeap::getBaseGpuHandle() const
{
    if (!mShaderVisible)
        FALCOR_THROW("getBaseGpuHandle() - heap must be shader visible.");
    return mGpuHeapStart;
}

template<typename HandleType>
HandleType getHandleCommon(HandleType base, uint32_t index, uint32_t descSize)
{
    base.ptr += descSize * index;
    return base;
}

D3D12DescriptorHeap::CpuHandle D3D12DescriptorHeap::getCpuHandle(uint32_t index) const
{
    return getHandleCommon(getBaseCpuHandle(), index, mDescriptorSize);
}

D3D12DescriptorHeap::GpuHandle D3D12DescriptorHeap::getGpuHandle(uint32_t index) const
{
    return getHandleCommon(getBaseGpuHandle(), index, mDescriptorSize);
}

D3D12DescriptorHeap::Allocation::SharedPtr D3D12DescriptorHeap::allocateDescriptors(uint32_t count)
{
    if (setupCurrentChunk(count) == false)
        return nullptr;

    if (mpCurrentChunk->chunkCount * kDescPerChunk - mpCurrentChunk->currentDesc < count)
    {
        return nullptr;
    }

    Allocation::SharedPtr pAlloc =
        Allocation::create(ref<D3D12DescriptorHeap>(this), mpCurrentChunk->getCurrentAbsoluteIndex(), count, mpCurrentChunk);

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
        if (mpCurrentChunk->getRemainingDescs() >= descCount)
            return true;

        if (mpCurrentChunk->allocCount == 0)
        {
            // Chunk is empty, doesn't necessarily mean it has enough space, need to check
            if (mpCurrentChunk->chunkCount * kDescPerChunk >= descCount)
            {
                mpCurrentChunk->reset();
                return true;
            }
        }
    }

    // Need a new chunk
    uint32_t chunkCount = (descCount + kDescPerChunk - 1) / kDescPerChunk;

    if (chunkCount == 1 && mFreeChunks.empty() == false)
    {
        mpCurrentChunk = mFreeChunks.back();
        mFreeChunks.pop_back();
        return true;
    }
    else if (chunkCount > 1 && mFreeLargeChunks.empty() == false)
    {
        // Find the smallest chunk big enough for the allocation
        auto it = std::lower_bound(mFreeLargeChunks.begin(), mFreeLargeChunks.end(), chunkCount, ChunkComparator());
        if (it != mFreeLargeChunks.end())
        {
            mpCurrentChunk = *it;
            mFreeLargeChunks.erase(it);
            return true;
        }
    }

    // No free chunks. Allocate
    if (mAllocatedChunks + chunkCount > mMaxChunkCount)
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
    if (pChunk->allocCount == 0 && (pChunk != mpCurrentChunk))
    {
        pChunk->reset();
        if (pChunk->chunkCount == 1)
            mFreeChunks.push_back(pChunk);
        else
            mFreeLargeChunks.insert(pChunk);
    }
}

D3D12DescriptorHeap::Allocation::SharedPtr D3D12DescriptorHeap::Allocation::create(
    ref<D3D12DescriptorHeap> pHeap,
    uint32_t baseIndex,
    uint32_t descCount,
    std::shared_ptr<Chunk> pChunk
)
{
    return SharedPtr(new Allocation(pHeap, baseIndex, descCount, pChunk));
}

D3D12DescriptorHeap::Allocation::Allocation(
    ref<D3D12DescriptorHeap> pHeap,
    uint32_t baseIndex,
    uint32_t descCount,
    std::shared_ptr<Chunk> pChunk
)
    : mpHeap(pHeap), mBaseIndex(baseIndex), mDescCount(descCount), mpChunk(pChunk)
{}

D3D12DescriptorHeap::Allocation::~Allocation()
{
    mpHeap->releaseChunk(mpChunk);
}
} // namespace Falcor
