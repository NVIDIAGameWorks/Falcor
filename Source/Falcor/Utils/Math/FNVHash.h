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

#include "Core/Macros.h"
#include "Core/Assert.h"

#include <cstdint>

namespace Falcor
{

/** Accumulates Fowler-Noll-Vo hash for inserted data.
    To hash multiple items, create one Hash and insert all the items into it if at all possible.
    This is superior to hashing the items individually and combining the hashes.

    \tparam T - type of the storage for the hash, either 32 or 64 unsigned integer
 */
template<typename T>
class FNVHash
{
    template<typename TT>
    struct ConstantTraits
    {};

    template<>
    struct ConstantTraits<uint64_t>
    {
        static constexpr uint64_t kOffsetBasis = UINT64_C(4695981039346656037);
        static constexpr uint64_t kPrime = UINT64_C(1099511628211);
    };

    template<>
    struct ConstantTraits<uint32_t>
    {
        static constexpr uint32_t kOffsetBasis = UINT32_C(2166136261);
        static constexpr uint32_t kPrime = UINT32_C(16777619);
    };

public:
    static constexpr T kOffsetBasis = ConstantTraits<T>::kOffsetBasis;
    static constexpr T kPrime = ConstantTraits<T>::kPrime;

    /** Inserts all data between [begin,end) into the hash.
        \param[in] begin
        \param[in] end
     */
    void insert(const void* begin, const void* end)
    {
        FALCOR_ASSERT(begin <= end);
        const uint8_t* srcData8 = reinterpret_cast<const uint8_t*>(begin);

        for(; srcData8 != end; ++srcData8)
        {
            mHash *= kPrime;
            mHash ^= *srcData8;
        }
    }

    /** Inserts all data starting at data and going for size bytes into the hash
        \param[in] data
        \param[in] size
     */
    void insert(const void* data, size_t size)
    {
        const uint8_t* srcData8 = reinterpret_cast<const uint8_t*>(data);
        insert(srcData8, srcData8 + size);
    }

    T get() const { return mHash; }

private:
    T mHash = kOffsetBasis;
};

using FNVHash64 = FNVHash<uint64_t>;
using FNVHash32 = FNVHash<uint32_t>;

inline uint64_t fnvHashArray64(const void* data, size_t size)
{
    FNVHash64 hash;
    hash.insert(data, size);
    return hash.get();
}

inline uint32_t fnvHashArray32(const void* data, size_t size)
{
    FNVHash32 hash;
    hash.insert(data, size);
    return hash.get();
}


} // namespace Falcor
