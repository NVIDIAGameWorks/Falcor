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

#include <algorithm>
#include <type_traits>

namespace Falcor
{
    /** Clamps a value within a range.
        \param[in] val Value to clamp
        \param[in] minVal Low end to clamp to
        \param[in] maxVal High end to clamp to
        \return Result
    */
    template<typename T>
    inline T clamp(const T& val, const T& minVal, const T& maxVal)
    {
        return std::min(std::max(val, minVal), maxVal);
    }

    /** Linearly interpolate two values.
        \param[in] a The first value.
        \param[in] b The second value.
        \param[in] t Interpolation weight.
        \return (1-t) * a + t * b.
    */
    template<typename T, typename U>
    inline typename std::enable_if<std::is_floating_point_v<U>, T>::type lerp(const T& a, const T& b, const U& t)
    {
        return (U(1) - t) * a + t * b;
    }

    /** Returns whether an integer number is a power of two.
    */
    template<typename T>
    constexpr typename std::enable_if<std::is_integral<T>::value, bool>::type isPowerOf2(T a)
    {
        return (a & (a - (T)1)) == 0;
    }

    /** Divide an a by b and round up to the next integer.
    */
    template<typename T>
    constexpr T div_round_up(T a, T b)
    {
        return (a + b - T(1)) / b;
    }

    /** Helper to align an integer value to a given alignment.
    */
    template<typename T>
    constexpr typename std::enable_if<std::is_integral<T>::value, T>::type align_to(T alignment, T value)
    {
        return ((value + alignment - T(1)) / alignment) * alignment;
    }
}
