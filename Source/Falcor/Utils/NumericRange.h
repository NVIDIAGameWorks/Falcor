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
#include "Core/Errors.h"
#include <utility>
#include <stdexcept>

namespace Falcor
{
    template<typename T, typename Enable = void>
    class NumericRange final {};

    /** Numeric range that can be iterated over.
        Should be replaced with C++20 std::views::iota when available.
    */
    template<typename T>
    class NumericRange<T, typename std::enable_if<std::is_integral<T>::value>::type> final
    {
    public:
        class Iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = T;
            using pointer = const T*;
            using reference = T;

            explicit Iterator(const T& value = T(0)) : mValue(value) {}
            const Iterator& operator++() { ++mValue; return *this; }
            bool operator!=(const Iterator& other) const { return other.mValue != mValue; }
            T operator*() const { return mValue; }
        private:
            T mValue;
        };

        explicit NumericRange(const T& begin, const T& end)
            : mBegin(begin)
            , mEnd(end)
        {
            if (begin > end) throw ArgumentError("Invalid range");
        }
        NumericRange() = delete;
        NumericRange(const NumericRange&) = delete;
        NumericRange(NumericRange&& other) = delete;

        Iterator begin() const { return Iterator(mBegin); }
        Iterator end() const { return Iterator(mEnd); }

    private:
        T mBegin, mEnd;
    };
};
