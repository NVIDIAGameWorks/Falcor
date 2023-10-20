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
#pragma once

#include <pxr/base/vt/array.h>
#include <unordered_map>

namespace Falcor
{

/**
 * @brief Class to convert vector of possibly-duplicate items to a vector of indices into a set of unique data items.
 *
 * @tparam T Underlying type
 * @tparam I Index value type
 * @tparam H Hash object on type T, used to determine data item equivalence
 */
template<typename T, typename I, typename H, typename E = std::equal_to<T>>
class IndexedVector
{
public:
    /**
     * @brief Append data item.
     * @param[in] v Data item to append
     */
    void append(const T& v)
    {
        uint32_t idx;
        append(v, idx);
    }

    /**
     * @brief Append data item.
     *
     * @param[in] v Data item to append
     * @param[out] idx Index of the unique item corresponding to v
     * @return True if @p v was newly inserted into the set of unique data item
     */
    bool append(const T& v, uint32_t& outIdx)
    {
        bool insertedNew = false;
        auto iter = mIndexMap.find(v);
        if (iter == mIndexMap.end())
        {
            iter = mIndexMap.insert(std::make_pair(v, I(mValues.size()))).first;
            outIdx = mValues.size();
            mValues.push_back(v);
            insertedNew = true;
        }
        else
        {
            outIdx = iter->second;
        }
        mIndices.push_back(iter->second);
        return insertedNew;
    }
    /**
     * @brief Get the set of unique data items.
     */
    const pxr::VtArray<T>& getValues() const { return mValues; }

    /**
     * @brief Get the ordered list of item indices.
     */
    const pxr::VtArray<I>& getIndices() const { return mIndices; }

private:
    std::unordered_map<T, I, H, E> mIndexMap;
    pxr::VtArray<T> mValues;
    pxr::VtArray<I> mIndices;
};
} // namespace Falcor
