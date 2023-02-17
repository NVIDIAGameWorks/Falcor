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
#include <vector>

namespace Falcor
{

/**
 * UnionFind for any integral type.
 * @tparam T - The integral type for which UnionFind works.
 */
template<typename T>
class UnionFind
{
    static_assert(std::is_unsigned_v<T>, "UnionFind only works on unsigned integer types");

public:
    UnionFind() = default;
    UnionFind(size_t size) { reset(size); }

    void reset(size_t size)
    {
        mParent.resize(size);
        for (size_t i = 0; i < size; ++i)
            mParent[i] = T(i);
        mSetSize.assign(size, 1);
        mSetCount = size;
    }

    T findSet(T v)
    {
        // If root of the set, return it
        if (v == mParent[v])
            return v;
        // Otherwise find the root from the parent, and relink to the root (so the search chain shortens)
        mParent[v] = findSet(mParent[v]);
        return mParent[v];
    }

    bool connectedSets(T v0, T v1) { return findSet(v0) == findSet(v1); }

    void unionSet(T v0, T v1)
    {
        // Find the roots
        v0 = findSet(v0);
        v1 = findSet(v1);
        // If already in the same set, bail out
        if (v0 == v1)
            return;
        // Make v0 root of the larger set
        if (mSetSize[v0] < mSetSize[v1])
            std::swap(v0, v1);
        // The smaller set is parented under the larger set (balances the depth)
        mParent[v1] = v0;
        mSetSize[v0] += mSetSize[v1];
        --mSetCount;
    }

    size_t getSetCount() const { return mSetCount; }

private:
    std::vector<T> mParent;
    std::vector<size_t> mSetSize;
    size_t mSetCount{0};
};

} // namespace Falcor
