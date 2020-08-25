/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/** Utility code for converting between various packed formats.

    The functions have been written to be compatible with the DXGI formats.
    Some use the 'precise' keyword to ensure bit exact results. We should add
    unit tests to make sure it is correctly implemented.

    It'd also be good to add optimized versions that don't care about NaN/inf
    propagation etc., as well as make the header shareable between the CPU/GPU.
*/


///////////////////////////////////////////////////////////////////////////////
//                              16-bit snorm
///////////////////////////////////////////////////////////////////////////////

/** Convert float value to 16-bit snorm value.
    Values outside [-1,1] are clamped and NaN is encoded as zero.
    \return 16-bit snorm value in low bits, high bits are all zeros or ones depending on sign.
*/
int floatToSnorm16(float v)
{
    v = isnan(v) ? 0.f : min(max(v, -1.f), 1.f);
    return (int)trunc(v * 32767.f + (v >= 0.f ? 0.5f : -0.5f));
}

/** Unpack a single 16-bit snorm from the lower bits of a dword.
    \param[in] packed 16-bit snorm in low bits, high bits don't care.
    \return Float value in [-1,1].
*/
float unpackSnorm16(uint packed)
{
    int bits = (int)(packed << 16) >> 16;
    precise float unpacked = max((float)bits / 32767.f, -1.0f);
    return unpacked;
}

/** Pack single float into a 16-bit snorm in the lower bits of the returned dword.
    \return 16-bit snorm in low bits, high bits all zero.
*/
uint packSnorm16(precise float v)
{
    return floatToSnorm16(v) & 0x0000ffff;
}

/** Unpack two 16-bit snorm values from the lo/hi bits of a dword.
*/
float2 unpackSnorm2x16(uint packed)
{
    int2 bits = int2(packed << 16, packed) >> 16;
    precise float2 unpacked = max((float2)bits / 32767.f, -1.0f);
    return unpacked;
}

/** Pack two floats into 16-bit snorm values in the lo/hi bits of a dword.
*/
uint packSnorm2x16(precise float2 v)
{
    return (floatToSnorm16(v.x) & 0x0000ffff) | (floatToSnorm16(v.y) << 16);
}


///////////////////////////////////////////////////////////////////////////////
//                              16-bit unorm
///////////////////////////////////////////////////////////////////////////////

/** Convert float value to 16-bit unorm (unsafe version).
    \param[in] v Value assumed to be in [0,1].
    \return 16-bit unorm in low bits, high bits all zeros.
*/
uint packUnorm16_unsafe(float v)
{
    return (uint)trunc(v * 65535.f + 0.5f);
}

/** Convert float value to 16-bit unorm.
    Values outside [0,1] are clamped and NaN is encoded as zero.
    \return 16-bit unorm in low bits, high bits all zeros.
*/
uint packUnorm16(float v)
{
    v = isnan(v) ? 0.f : saturate(v);
    return packUnorm16_unsafe(v);
}

/** Pack two floats into 16-bit unorm values in a dword.
*/
uint packUnorm2x16(float2 v)
{
    return (packUnorm16(v.y) << 16) | packUnorm16(v.x);
}

/** Pack two floats into 16-bit unorm values in a dword (unsafe version)
    \param[in] v Two values assumed to be in [0,1].
*/
uint packUnorm2x16_unsafe(float2 v)
{
    return (packUnorm16_unsafe(v.y) << 16) | packUnorm16_unsafe(v.x);
}

/** Convert 16-bit unorm to float value.
    \param[in] packed 16-bit unorm in low bits, high bits don't care.
    \return Float value in [0,1].
*/
float unpackUnorm16(uint packed)
{
    return float(packed & 0xffff) * (1.f / 65535);
}

/** Unpack two 16-bit unorm values from a dword.
*/
float2 unpackUnorm2x16(uint packed)
{
    return float2(packed & 0xffff, packed >> 16) * (1.f / 65535);
}

///////////////////////////////////////////////////////////////////////////////
// 32-bit HDR color format
///////////////////////////////////////////////////////////////////////////////

/** Pack three positive floats into a dword.
    https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli
*/
uint packR11G11B10(float3 v)
{
    // Clamp upper bound so that it doesn't accidentally round up to INF 
    v = min(v, asfloat(0x477C0000));  
    // Exponent=15, Mantissa=1.11111
    uint r = ((f32tof16(v.x) + 8) >> 4) & 0x000007ff;
    uint g = ((f32tof16(v.y) + 8) << 7) & 0x003ff800;
    uint b = ((f32tof16(v.z) + 16) << 17) & 0xffc00000;
    return r | g | b;
}

/** Unpack three positive floats from a dword.
    https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli
*/
float3 unpackR11G11B10(uint packed)
{
    float r = f16tof32((packed << 4 ) & 0x7FF0);
    float g = f16tof32((packed >> 7 ) & 0x7FF0);
    float b = f16tof32((packed >> 17) & 0x7FE0);
    return float3(r, g, b);
}
