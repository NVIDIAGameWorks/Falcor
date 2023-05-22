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

#include "ScalarMath.h"
#include "Vector.h"

namespace Falcor
{

///////////////////////////////////////////////////////////////////////////////
//                              16-bit snorm
///////////////////////////////////////////////////////////////////////////////

/**
 * Convert float value to 16-bit snorm value.
 * Values outside [-1,1] are clamped and NaN is encoded as zero.
 * @return 16-bit snorm value in low bits, high bits are all zeros or ones depending on sign.
 */
inline int floatToSnorm16(float v)
{
    v = math::isnan(v) ? 0.f : math::min(math::max(v, -1.f), 1.f);
    return (int)math::trunc(v * 32767.f + (v >= 0.f ? 0.5f : -0.5f));
}

/**
 * Unpack a single 16-bit snorm from the lower bits of a dword.
 * @param[in] packed 16-bit snorm in low bits, high bits don't care.
 * @return Float value in [-1,1].
 */
inline float unpackSnorm16(uint packed)
{
    int bits = (int)(packed << 16) >> 16;
    float unpacked = math::max((float)bits / 32767.f, -1.0f);
    return unpacked;
}

/**
 * Pack single float into a 16-bit snorm in the lower bits of the returned dword.
 * @return 16-bit snorm in low bits, high bits all zero.
 */
inline uint packSnorm16(float v)
{
    return floatToSnorm16(v) & 0x0000ffff;
}

/**
 * Unpack two 16-bit snorm values from the lo/hi bits of a dword.
 * @param[in] packed Two 16-bit snorm in low/high bits.
 * @return Two float values in [-1,1].
 */
inline float2 unpackSnorm2x16(uint packed)
{
    int2 bits = int2(packed << 16, packed) >> 16;
    float2 unpacked = math::max((float2)bits / 32767.f, float2(-1.0f));
    return unpacked;
}

/**
 * Pack two floats into 16-bit snorm values in the lo/hi bits of a dword.
 * @return Two 16-bit snorm in low/high bits.
 */
inline uint packSnorm2x16(float2 v)
{
    return (floatToSnorm16(v.x) & 0x0000ffff) | (floatToSnorm16(v.y) << 16);
}

} // namespace Falcor
