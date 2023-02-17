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
#include "Testing/UnitTest.h"
#include "Utils/Algorithm/UnionFind.h"

#include <random>
#include <vector>
#include <set>

namespace Falcor
{

namespace
{

template<typename T>
class TrivialUnionFind
{
public:
    TrivialUnionFind() = default;
    TrivialUnionFind(size_t size) { reset(size); }

    void reset(size_t size)
    {
        mSets.clear();
        mSets.resize(size);
        for (size_t i = 0; i < size; ++i)
            mSets[i].insert(i);
    }

    T findSet(T v)
    {
        for (size_t i = 0; i < mSets.size(); ++i)
            if (mSets[i].count(v) > 0)
                return i;
        return 0;
    }

    bool connectedSets(T v0, T v1) { return findSet(v0) == findSet(v1); }

    void unionSet(T v0, T v1)
    {
        v0 = findSet(v0);
        v1 = findSet(v1);
        if (v0 == v1)
            return;
        mSets[v0].insert(mSets[v1].begin(), mSets[v1].end());
        mSets.erase(mSets.begin() + v1);
    }

    size_t getSetCount() const { return mSets.size(); }

private:
    std::vector<std::set<T>> mSets;
};

} // namespace

CPU_TEST(UnionFind_randomized)
{
    size_t count = 10;

    for (int run = 0; run < 20; ++run)
    {
        std::mt19937 r(1234 + run);
        UnionFind<size_t> uf;
        TrivialUnionFind<size_t> reference;
        uf.reset(count);
        reference.reset(count);

        size_t iter = 0;
        while (reference.getSetCount() > 1 && iter < 1000)
        {
            size_t v0 = r() % count;
            size_t v1 = r() % count;

            EXPECT_EQ(uf.connectedSets(v0, v1), reference.connectedSets(v0, v1)) << fmt::format("Iter: {}/{}", iter, run);
            uf.unionSet(v0, v1);
            reference.unionSet(v0, v1);
            ASSERT_EQ(uf.getSetCount(), reference.getSetCount()) << fmt::format("Iter: {}/{}", iter, run);

            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = i + 1; j < count; ++j)
                {
                    EXPECT_EQ(uf.connectedSets(i, j), reference.connectedSets(i, j))
                        << fmt::format("Iter: {}/{}; i = {}; j = {}", iter, run, i, j);
                }
            }

            ++iter;
        }
    }
}

} // namespace Falcor
