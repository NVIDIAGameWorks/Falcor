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
#pragma once
#include "Core/Assert.h"
#include "Utils/Math/Common.h"
#include <new>
#include <utility>
#include <vector>

namespace Falcor
{

    /** Utility class for aligned memory allocations on the GPU.

        AlignedAllocator can enforce various alignment requirements,
        including minimum byte alignment and (optionally) that
        allocated objects don't span two cache lines if they can fit
        into one.  Note that it's intended to be used to manage GPU
        allocations and so it assumes that the base pointer starts at a
        cache line.  As such, it doesn't provide any alignment
        guarantees on the CPU side (where it doesn't matter anyway).
    */
    class AlignedAllocator
    {
    public:
        /** Sets the minimum alignment for allocated objects. If a value of
            zero is provided, no additional alignment is performed.
        */
        void setMinimumAlignment(int minAlignment)
        {
            FALCOR_ASSERT(minAlignment == 0 || isPowerOf2(minAlignment));
            mMinAlignment = minAlignment;
        }

        /** Sets the cache line size so that allocations can be aligned so
            that they don't span multiple cache lines (if possible).  If a
            value of zero is provided, then the allocator doesn't prevent
            objects from spanning cache lines.
        */
        void setCacheLineSize(int cacheLineSize)
        {
            FALCOR_ASSERT(cacheLineSize == 0 || isPowerOf2(cacheLineSize));
            mCacheLineSize = cacheLineSize;
        }

        /** Allocates an object of given type and executes its constructor.
            \param[in] args Arguments to pass to the constructor.
            \return pointer to allocated object.
        */
        template <typename T, typename ...Args> T* allocate(Args&&... args)
        {
            const size_t size = sizeof(T);
            computeAndAllocatePadding(size);
            void* ptr = allocInternal(size);
            return new (ptr) T(std::forward<Args>(args)...);
        }

        /** Allocates an object of given type, potentially including additional memory at
            the end of it, and executes its constructor.
            \param[in] size Amount of memory to allocate. Must be >= sizeof(T).
            \param[in] args Arguments to pass to the constructor.
            \return pointer to allocated object.
        */
        template <typename T, typename ...Args> T* allocateSized(size_t size, Args&&... args)
        {
            FALCOR_ASSERT(size >= sizeof(T));
            computeAndAllocatePadding(size);
            void* ptr = allocInternal(size);
            return new (ptr) T(std::forward<Args>(args)...);
        }

        void reserve(size_t size) { mBuffer.reserve(size); }

        void resize(size_t size) { mBuffer.resize(size, 0); }

        /** Returns the pointer to the start of the allocated buffer.
        */
        void* getStartPointer() { return mBuffer.data(); }
        const void* getStartPointer() const { return mBuffer.data(); }

        /** Returns of the offset of the given pointer inside the allocation buffer.
        */
        size_t offsetOf(void* ptr) const
        {
            FALCOR_ASSERT(ptr >= mBuffer.data() && ptr < mBuffer.data() + mBuffer.size());
            return static_cast<uint8_t*>(ptr) - mBuffer.data();
        }

        void reset() { mBuffer.clear(); }

        size_t getSize() const { return mBuffer.size(); }
        size_t getCapacity() const { return mBuffer.capacity(); }

    private:
        void computeAndAllocatePadding(size_t size)
        {
            const size_t currentOffset = mBuffer.size();

            if (mCacheLineSize > 0)
            {
                const size_t cacheLineOffset = currentOffset % mCacheLineSize;
                if (size < mCacheLineSize && cacheLineOffset + size > mCacheLineSize)
                {
                    // The allocation is smaller than a cache line but
                    // would span two cache lines; move to the start of the
                    // next cache line.
                    const size_t pad = mCacheLineSize - cacheLineOffset;
                    (void)allocInternal(pad);
                    // There's need to worry about any further alignment
                    // issues now.
                    return;
                }
            }

            if (mMinAlignment > 0 && currentOffset % mMinAlignment)
            {
                // We're not at the minimum alignment; get aligned.
                const size_t pad = mMinAlignment - (currentOffset % mMinAlignment);
                (void)allocInternal(pad);
            }
        }

        void* allocInternal(size_t size)
        {
            auto iter = mBuffer.insert(mBuffer.end(), size, {});
            return &*iter;
        }

        size_t mMinAlignment = 16;
        size_t mCacheLineSize = 128;
        std::vector<uint8_t> mBuffer;
    };
}
