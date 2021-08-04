/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "CompositeMode.slangh"

/** Simple composite pass that blends two buffers together.
*/

cbuffer CB
{
    uint2 frameDim;
    float scaleA;
    float scaleB;
}

// Inputs
Texture2D<float4> A;
Texture2D<float4> B;

// Output
#if !defined(OUTPUT_FORMAT)
#error OUTPUT_FORMAT is undefined
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_FLOAT
RWTexture2D<float4> output;
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_UINT
RWTexture2D<uint4> output;
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_SINT
RWTexture2D<int4> output;
#else
#error OUTPUT_FORMAT unknown
#endif

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim)) return;

    float4 result = float4(0.f);
#if !defined(COMPOSITE_MODE)
    #error COMPOSITE_MODE is undefined
#elif COMPOSITE_MODE == COMPOSITE_MODE_ADD
    result = (scaleA * A[pixel]) + (scaleB * B[pixel]);
#elif COMPOSITE_MODE == COMPOSITE_MODE_MULTIPLY
    result = (scaleA * A[pixel]) * (scaleB * B[pixel]);
#else
    #error COMPOSITE_MODE unknown
#endif

#if OUTPUT_FORMAT != OUTPUT_FORMAT_FLOAT
    result = round(result);
#endif
    output[pixel] = result;
}
