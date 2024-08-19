/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"
#include "Core/Program/DefineList.h"
#include "Core/Program/ShaderVar.h"
#include "Core/Error.h"

#include <vector>
#include <fmt/format.h>

namespace Falcor
{

/**
 * @brief Represents a cpu/gpu buffer, that handles the GPU limit on 4GB buffers, up to 4 billion items.
 *
 * GPU's currently handle at most 4GB buffers. If we want to store more data, it has to be split into multiple buffers.
 * This class facilitates this for objects with the power-of-2 byte size, by using the upper bits of the 32b index
 * to select which 4GB buffer. The number of buffers is equal to the size of the stored object, i.e.,
 * for uint32_t, we can have up to 2^30 in a single 4GB buffer, leaving only 2 bits for buffer selection.
 * This naturally limites the size to the total of 2^32 items.
 *
 * @tparam T - Type of the object stored in the buffer.
 */
template<typename T, bool TByteBuffer>
class SplitBuffer
{
    /// We are encoding the buffer index in the top bits of the index,
    /// so we can only have as many buffers as it the sizeof(T) rounded down
    /// to the nearest power of two
    static_assert((sizeof(T) & (sizeof(T) - 1)) == 0, "Size has to be power of 2");

public:
    using ElementType = T;

public:
    SplitBuffer() : mCpuBuffers(1) {}

    /// Returns the maximum number of buffers supported for the given split buffer.
    size_t getMaxBufferCount() const { return kMaxBufferCount; }

    /// Forces the number of buffers to be at least bufferCount.
    /// Used for debugging, as SplitBuffer will balance content of all existing buffers
    /// before creating new ones, so setting this to MaxBufferCount will create balanced
    /// buffers even for small scenes.
    void setBufferCount(uint32_t bufferCount)
    {
        FALCOR_ASSERT(mGpuBuffers.empty(), "Cannot change buffer count after creating GPU buffers.");
        FALCOR_CHECK(bufferCount >= getBufferCount(), "Cannot reduce number of existing buffers ({}).", getBufferCount());
        FALCOR_CHECK(bufferCount <= kMaxBufferCount, "Cannot exceed the max number of buffers ({}).", kMaxBufferCount);
        mCpuBuffers.resize(bufferCount);
    }

    /// Sets name used for CPU reporting, and for GPU buffer names
    void setName(std::string_view name) { mBufferName = name; }

    /// Sets the name of the define that governs the number of buffers in the GPU code.
    /// This is less than ideal, but there is no way to automatically detect which
    /// type/define it is on the GPU.
    void setBufferCountDefinePrefix(std::string bufferCountDefinePrefix) { mBufferCountDefinePrefix = bufferCountDefinePrefix; }

    /// Insert range of items into the split buffer, returning an index at which it starts.
    /// All items in the range will be in the same GPU buffer.
    /// Not possible after the GPU buffers have been created.
    template<typename Iter>
    uint32_t insert(Iter first, Iter last)
    {
        FALCOR_ASSERT(mGpuBuffers.empty(), "Cannot insert after creating GPU buffers.");
        if (first == last)
            return 0;
        const size_t itemCount = std::distance(first, last);

        // Find the buffer with the fewest items.
        auto it = std::min_element(
            mCpuBuffers.begin(),
            mCpuBuffers.end(),
            [](const std::vector<T>& lhs, const std::vector<T>& rhs) { return lhs.size() < rhs.size(); }
        );
        uint32_t bufferIndex = std::distance(mCpuBuffers.begin(), it);

        // If new items wouldn't fit into the buffer with fewest items, create a new buffer,
        // throw if new buffer cannot be created.
        if ((mCpuBuffers[bufferIndex].size() + itemCount) * sizeof(T) > kBufferSizeLimit)
        {
            bufferIndex = mCpuBuffers.size();
            if (bufferIndex >= kMaxBufferCount)
                FALCOR_THROW("Buffers {} cannot accomodate all the date within the buffer limit.", mBufferName);
            mCpuBuffers.push_back({});
        }

        const uint32_t elementIndex = mCpuBuffers[bufferIndex].size();
        FALCOR_ASSERT((((1 << kBufferIndexOffset) - 1) & elementIndex) == elementIndex, "Element index overflows into buffer index");
        mCpuBuffers[bufferIndex].insert(mCpuBuffers[bufferIndex].end(), first, last);
        return ((bufferIndex << kBufferIndexOffset) | elementIndex);
    }

    /// Inserts an empty range, mostly for testing and debugging purposes
    uint32_t insertEmpty(size_t itemCount)
    {
        FALCOR_ASSERT(mGpuBuffers.empty(), "Cannot insert after creating GPU buffers.");
        if (itemCount == 0)
            return 0;

        // Find the buffer with the fewest items.
        auto it = std::min_element(
            mCpuBuffers.begin(),
            mCpuBuffers.end(),
            [](const std::vector<T>& lhs, const std::vector<T>& rhs) { return lhs.size() < rhs.size(); }
        );
        uint32_t bufferIndex = std::distance(mCpuBuffers.begin(), it);

        // If new items wouldn't fit into the buffer with fewest items, create a new buffer,
        // throw if new buffer cannot be created.
        if ((mCpuBuffers[bufferIndex].size() + itemCount) * sizeof(T) > kBufferSizeLimit)
        {
            bufferIndex = mCpuBuffers.size();
            if (bufferIndex >= kMaxBufferCount)
                FALCOR_THROW("Buffers {} cannot accomodate all the date within the buffer limit.", mBufferName);
            mCpuBuffers.push_back({});
        }

        const uint32_t elementIndex = mCpuBuffers[bufferIndex].size();
        FALCOR_ASSERT((((1 << kBufferIndexOffset) - 1) & elementIndex) == elementIndex, "Element index overflows into buffer index");
        mCpuBuffers[bufferIndex].resize(mCpuBuffers[bufferIndex].size() + itemCount);
        return ((bufferIndex << kBufferIndexOffset) | elementIndex);
    }

    /// Creates the GPU buffers, locking further inserts.
    /// Will clear any existing GPU buffers.
    void createGpuBuffers(const ref<Device>& mpDevice, ResourceBindFlags bindFlags)
    {
        mGpuBuffers.clear();
        mGpuBuffers.reserve(mCpuBuffers.size());
        for (size_t i = 0; i < mCpuBuffers.size(); ++i)
        {
            if (mCpuBuffers[i].empty())
            {
                mGpuBuffers.push_back({});
                continue;
            }

            ref<Buffer> buffer = mpDevice->createStructuredBuffer(
                sizeof(T), mCpuBuffers[i].size(), bindFlags, MemoryType::DeviceLocal, mCpuBuffers[i].data(), false
            );
            buffer->setName(fmt::format("SplitBuffer:{}:[{}]", mBufferName, i));
            mGpuBuffers.push_back(std::move(buffer));
        }
    };

    /// Return true when the SplitBuffer empty.
    bool empty() const
    {
        // We check if all CPU buffers are empty. If so, we also check GPU buffers, as the CPU buffers
        // maybe have been dropped.
        for (auto& it : mCpuBuffers)
            if (!it.empty())
                return false;
        for (auto& it : mGpuBuffers)
            if (it)
                return false;
        return true;
    }

    /// Returns the number of buffers.
    size_t getBufferCount() const
    {
        // We check both CPU and GPU buffers, to get correct answer even before `createGpuBuffers`
        // and after `dropCpuBuffers`
        return std::max(mCpuBuffers.size(), mGpuBuffers.size());
    }

    /// Total number of bytes used by the buffers (mostly for statistics)
    size_t getByteSize() const
    {
        size_t result = 0;
        if (!mCpuBuffers.empty())
        {
            for (auto& it : mCpuBuffers)
                result += it.size() * sizeof(T);
        }
        else
        {
            for (auto& it : mGpuBuffers)
                result += it->getSize();
        }
        return result;
    }

    uint32_t getBufferIndex(uint32_t index) const { return (index >> kBufferIndexOffset); }

    uint32_t getElementIndex(uint32_t index) const { return index & kElementIndexMask; }

    /// Access to the CPU data via index returned from `insert`
    const T& operator[](uint32_t index) const
    {
        FALCOR_ASSERT(!mCpuBuffers.empty());
        const uint32_t bufferIndex = getBufferIndex(index);
        const uint32_t elementIndex = getElementIndex(index);
        return mCpuBuffers[bufferIndex][elementIndex];
    }

    /// Access to the CPU data via index returned from `insert`
    T& operator[](uint32_t index)
    {
        FALCOR_ASSERT(!mCpuBuffers.empty());
        const uint32_t bufferIndex = getBufferIndex(index);
        const uint32_t elementIndex = getElementIndex(index);
        return mCpuBuffers[bufferIndex][elementIndex];
    }

    /// Removes all CPU data, to conserve memory.
    void dropCpuData() { mCpuBuffers.clear(); }

    /// True when there is any CPU buffer present.
    bool hasCpuData() const { return !mCpuBuffers.empty(); }

    /// Return a GPU buffer, indexed by buffer index.
    ref<Buffer> getGpuBuffer(uint32_t bufferIndex) const { return mGpuBuffers[bufferIndex]; }

    const std::vector<T>& getCpuBuffer(uint32_t bufferIndex) const { return mCpuBuffers[bufferIndex]; }

    /// Gets GPU address of the index returned from `insert`
    uint64_t getGpuAddress(uint32_t index) const
    {
        const uint32_t bufferIndex = getBufferIndex(index);
        const uint32_t elementIndex = getElementIndex(index);
        return getGpuBuffer(bufferIndex)->getGpuAddress() + size_t(elementIndex) * sizeof(T);
    }

    /// Sets the define that governs how many buffers will the corresponding GPU type have.
    void getShaderDefines(DefineList& defines) const
    {
        FALCOR_ASSERT(!mBufferCountDefinePrefix.empty());
        defines.add(mBufferCountDefinePrefix + "_BUFFER_COUNT", std::to_string(mGpuBuffers.size()));
        defines.add(mBufferCountDefinePrefix + "_BUFFER_INDEX_BITS", std::to_string(kBufferIndexBits));
    }

    /// Binds the SplitBuffer to the corresponding GPU type
    void bindShaderData(const ShaderVar& var) const
    {
        static const std::string kDataStr = "data";
        for (size_t i = 0; i < mGpuBuffers.size(); ++i)
            var[kDataStr][i] = mGpuBuffers[i];
    }

private:
    /// Min number of bits needed to store the number
    static constexpr uint32_t bitCount(uint32_t number) { return number < 2 ? number : (bitCount(number / 2) + 1); }

private:
    static constexpr size_t k4GBSizeLimit = (UINT64_C(1) << UINT64_C(32)) - UINT64_C(1024);
    static constexpr size_t k2GBSizeLimit = (UINT64_C(1) << UINT64_C(31));
    static constexpr size_t kBufferSizeLimit = (sizeof(T) < 16 || TByteBuffer) ? k2GBSizeLimit : k4GBSizeLimit;
    static constexpr size_t kMaxElementCount = (kBufferSizeLimit + 1 - sizeof(T)) / sizeof(T);
    static_assert(kMaxElementCount * sizeof(T) <= kBufferSizeLimit);
    static constexpr size_t kElementIndexBits = bitCount(kMaxElementCount - 1);
    static constexpr size_t kBufferIndexBits = 32 - kElementIndexBits;
    static constexpr size_t kMaxBufferCount = (1u << kBufferIndexBits);

    static constexpr uint32_t kBufferIndexOffset = 32 - kBufferIndexBits;
    static constexpr size_t kElementIndexMask = ((1 << kBufferIndexOffset) - 1);

    std::string mBufferName;
    std::string mBufferCountDefinePrefix;
    std::vector<std::vector<T>> mCpuBuffers;
    std::vector<ref<Buffer>> mGpuBuffers;

    friend class SceneCache;
};

} // namespace Falcor
