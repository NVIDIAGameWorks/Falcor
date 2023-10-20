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
#include "BufferAllocator.h"
#include "Core/API/Device.h"
#include "Utils/Math/Common.h"

namespace Falcor
{
BufferAllocator::BufferAllocator(size_t alignment, size_t elementSize, size_t cacheLineSize, ResourceBindFlags bindFlags)
    : mAlignment(alignment), mElementSize(elementSize), mCacheLineSize(cacheLineSize), mBindFlags(bindFlags)
{
    // Check the different alignment requirements.
    // Some of these are stricter than they need be and can be relaxed in the future.
    FALCOR_CHECK(alignment == 0 || isPowerOf2(alignment), "Alignment must be a power of two.");
    FALCOR_CHECK(cacheLineSize == 0 || isPowerOf2(cacheLineSize), "Cache line size must be a power of two.");
    FALCOR_CHECK(cacheLineSize == 0 || alignment <= cacheLineSize, "Alignment must be smaller or equal to the cache line size.");

    // Check requirements on element size for structured buffers.
    // The alignment can be smaller or larger than the element size, but they need to be integer multiples.
    if (elementSize > 0 && alignment > 0)
    {
        size_t a = std::max(alignment, elementSize);
        size_t b = std::min(alignment, elementSize);
        FALCOR_CHECK(a % b == 0, "Alignment and element size needs to be integer multiples.");
    }
}

size_t BufferAllocator::allocate(size_t byteSize)
{
    computeAndAllocatePadding(byteSize);
    return allocInternal(byteSize);
}

void BufferAllocator::setBlob(const void* pData, size_t byteOffset, size_t byteSize)
{
    FALCOR_CHECK(pData != nullptr, "Invalid pointer.");
    FALCOR_CHECK(byteOffset + byteSize <= mBuffer.size(), "Memory region is out of range.");
    std::memcpy(mBuffer.data() + byteOffset, pData, byteSize);
    markAsDirty(byteOffset, byteSize);
}

void BufferAllocator::modified(size_t byteOffset, size_t byteSize)
{
    FALCOR_CHECK(byteOffset + byteSize <= mBuffer.size(), "Memory region is out of range.");
    markAsDirty(byteOffset, byteSize);
}

void BufferAllocator::clear()
{
    mBuffer.clear();
    mDirty = {};
}

ref<Buffer> BufferAllocator::getGPUBuffer(ref<Device> pDevice)
{
    if (mBuffer.empty())
    {
        // If there is no allocated data, we don't need a GPU buffer and return nullptr.
        return nullptr;
    }

    // Compute required size of the buffer on the GPU including padding and allocate
    // buffer of the right type (structured or raw buffer).
    size_t elemSize = mElementSize > 0 ? mElementSize : 4ull;
    size_t bufSize = align_to(elemSize, mBuffer.size());

    if (mpGpuBuffer == nullptr || mpGpuBuffer->getSize() < bufSize)
    {
        if (mElementSize > 0)
        {
            size_t elemCount = bufSize / mElementSize;
            FALCOR_ASSERT(elemCount * mElementSize == bufSize);
            mpGpuBuffer = pDevice->createStructuredBuffer(
                mElementSize, elemCount, mBindFlags, MemoryType::DeviceLocal, nullptr, false /* no UAV counter */
            );
        }
        else
        {
            mpGpuBuffer = pDevice->createBuffer(bufSize, mBindFlags, MemoryType::DeviceLocal, nullptr);
        }

        mDirty = Range(0, mBuffer.size()); // Mark entire buffer as dirty so the data gets uploaded.
    }

    // If any range is dirty, upload the data from the CPU to the GPU.
    if (mDirty.start < mDirty.end)
    {
        size_t byteSize = mDirty.end - mDirty.start;
        size_t byteOffset = mDirty.start;
        FALCOR_ASSERT(byteOffset + byteSize <= mBuffer.size());
        FALCOR_ASSERT(mBuffer.size() <= mpGpuBuffer->getSize());
        mpGpuBuffer->setBlob(mBuffer.data() + byteOffset, byteOffset, byteSize);

        mDirty = {mBuffer.size(), 0}; // Reset dirty range to inverted range. Any min/max operation on it will work.
    }

    return mpGpuBuffer;
}

// Private

void BufferAllocator::computeAndAllocatePadding(size_t byteSize)
{
    size_t currentOffset = mBuffer.size();

    if (mAlignment > 0 && currentOffset % mAlignment > 0)
    {
        // We're not at the minimum alignment; get aligned.
        currentOffset += mAlignment - (currentOffset % mAlignment);
    }

    if (mCacheLineSize > 0)
    {
        const size_t cacheLineOffset = currentOffset % mCacheLineSize;
        if (byteSize <= mCacheLineSize && cacheLineOffset + byteSize > mCacheLineSize)
        {
            // The allocation is smaller than or equal to a cache line but
            // would span two cache lines; move to the start of the next cache line.
            currentOffset += mCacheLineSize - cacheLineOffset;
        }
    }

    size_t pad = currentOffset - mBuffer.size();
    if (pad > 0)
    {
        allocInternal(pad);
    }
    FALCOR_ASSERT(mAlignment == 0 || mBuffer.size() % mAlignment == 0);
}

size_t BufferAllocator::allocInternal(size_t byteSize)
{
    size_t byteOffset = mBuffer.size();
    mBuffer.insert(mBuffer.end(), byteSize, {});
    return byteOffset;
}

void BufferAllocator::markAsDirty(const Range& range)
{
    FALCOR_ASSERT(range.start < range.end);
    mDirty.start = std::min(mDirty.start, range.start);
    mDirty.end = std::max(mDirty.end, range.end);
}
} // namespace Falcor
