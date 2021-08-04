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

/** Temporal accumulation render pass.

    There are entry points for each of the three supported accumulation modes.
    Note that for the compensated summation mode the shader _must_ be compiled
    for precise floating-point operations (no reordering).

    In all modes, the shader writes the current accumulated average to the
    output texture. The intermediate buffers are internal to the pass.
*/

cbuffer PerFrameCB
{
    uint2   gResolution;
    uint    gAccumCount;
    bool    gAccumulate;
    bool    gMovingAverageMode;
}

// Input data to accumulate and accumulated output.
#define INPUT_FORMAT_FLOAT 0
#define INPUT_FORMAT_UINT  1
#define INPUT_FORMAT_SINT  2

#if _INPUT_FORMAT == INPUT_FORMAT_FLOAT
typedef float4 InputDataType;
#elif _INPUT_FORMAT == INPUT_FORMAT_UINT
typedef uint4 InputDataType;
#elif _INPUT_FORMAT == INPUT_FORMAT_SINT
typedef int4 InputDataType;
#else
#error "Invalid format"
#endif

Texture2D<InputDataType> gCurFrame;
RWTexture2D<float4> gOutputFrame;

// Last frame data, format depends on accumulation mode.
RWTexture2D<float4> gLastFrameSum;      // If mode is Single or SingleKahan
RWTexture2D<float4> gLastFrameCorr;     // If mode is SingleKahan
RWTexture2D<uint4>  gLastFrameSumLo;    // If mode is Double
RWTexture2D<uint4>  gLastFrameSumHi;    // If mode is Double


/** Single precision standard summation.
*/
[numthreads(16, 16, 1)]
void accumulateSingle(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution)) return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate)
    {
        float curWeight = 1.0 / (gAccumCount + 1);

        if (gMovingAverageMode)
        {
            // Exponential weighted moving average mode.
            // Blend the accumulated frame with the current frame.
            output = lerp(gLastFrameSum[pixelPos], curColor, curWeight);

            gLastFrameSum[pixelPos] = output;
        }
        else
        {
            // High precision mode with no frame limit.
            // Fetch previous sum and compute the new sum.
            float4 sum = gLastFrameSum[pixelPos] + curColor;
            output = sum * curWeight;

            gLastFrameSum[pixelPos] = sum;
        }
    }
    else
    {
        output = curColor;
    }

    gOutputFrame[pixelPos] = output;
}

/** Single precision compensated summation.
*/
[numthreads(16, 16, 1)]
void accumulateSingleCompensated(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution)) return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate)
    {
        // Fetch the previous sum and running compensation term.
        float4 sum = gLastFrameSum[pixelPos];
        float4 c = gLastFrameCorr[pixelPos];                // c measures how large (+) or small (-) the current sum is compared to what it should be.

        // Adjust current value to minimize the running error.
        // Compute the new sum by adding the adjusted current value.
        float4 y = curColor - c;
        float4 sumNext = sum + y;                           // The value we'll see in 'sum' on the next iteration.
        output = sumNext / (gAccumCount + 1);

        gLastFrameSum[pixelPos] = sumNext;
        gLastFrameCorr[pixelPos] = (sumNext - sum) - y;     // Store new correction term.
    }
    else
    {
        output = curColor;
    }

    gOutputFrame[pixelPos] = output;
}

/** Double precision standard summation.
*/
[numthreads(16, 16, 1)]
void accumulateDouble(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution)) return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate)
    {
        double curWeight = 1.0 / (gAccumCount + 1);

        // Fetch the previous sum in double precision.
        // There is no 'double' resource format, so the bits are stored in two uint4 textures.
        uint4 sumLo = gLastFrameSumLo[pixelPos];
        uint4 sumHi = gLastFrameSumHi[pixelPos];

        double sum[4];

        if (gMovingAverageMode)
        {
            // Exponential weighted moving average mode.
            for (int i = 0; i < 4; i++)
            {
                sum[i] = asdouble(sumLo[i], sumHi[i]);
                sum[i] = sum[i] * (1.0 - curWeight) + (double)curColor[i] * curWeight;
                asuint(sum[i], sumLo[i], sumHi[i]);
                output[i] = (float)sum[i];
            }
        }
        else
        {
            // High precision mode with no frame limit.
            for (int i = 0; i < 4; i++)
            {
                sum[i] = asdouble(sumLo[i], sumHi[i]);
                sum[i] += (double)curColor[i];
                asuint(sum[i], sumLo[i], sumHi[i]);
                output[i] = (float)(sum[i] * curWeight);
            }
        }

        gLastFrameSumLo[pixelPos] = sumLo;
        gLastFrameSumHi[pixelPos] = sumHi;
    }
    else
    {
        output = curColor;
    }

    gOutputFrame[pixelPos] = output;
}
