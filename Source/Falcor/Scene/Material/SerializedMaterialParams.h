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

#include "Core/Error.h"
#include "Utils/Math/Vector.h"
#include <array>

namespace Falcor
{

struct SerializedMaterialParams : public std::array<float, 20>
{
    static constexpr size_t kParamCount = 20;

    void write(float value, size_t offset)
    {
        FALCOR_ASSERT(offset <= size());
        (*this)[offset] = value;
    }

    template<int N>
    void write(math::vector<float, N> value, size_t offset)
    {
        FALCOR_ASSERT(offset + N <= size());
        for (size_t i = 0; i < N; ++i)
            (*this)[offset + i] = value[i];
    }

    void read(float& value, size_t offset) const
    {
        FALCOR_ASSERT(offset <= size());
        value = (*this)[offset];
    }

    template<int N>
    void read(math::vector<float, N>& value, size_t offset) const
    {
        FALCOR_ASSERT(offset + N <= size());
        for (size_t i = 0; i < N; ++i)
            value[i] = (*this)[offset + i];
    }
};

} // namespace Falcor
