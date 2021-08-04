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
#include "Utils/Math/MathConstants.slangh"

struct TextureAnalyzer
{
    Texture2D<float4> input;
    RWByteAddressBuffer result;
    uint32_t resultOffset;
    uint2 inputDim;

    void clear(const uint index)
    {
        if (index >= inputDim.x) return;

        uint offset = resultOffset + index * 64;
        result.Store4(offset, uint4(0));                        // mask
        result.Store4(offset + 16, asuint(float4(0)));          // value
        result.Store4(offset + 32, asuint(float4(FLT_MAX)));    // minValue
        result.Store4(offset + 48, asuint(float4(0)));          // maxValue
    }

    void analyze(const uint2 texel)
    {
        uint mask = 0;
        float4 minVal = FLT_MAX;
        float4 maxVal = -FLT_MAX;
        bool valid = false;

        if (all(texel < inputDim))
        {
            // Read reference value from top-left texel.
            float4 ref = input[uint2(0, 0)];

            if (all(texel == uint2(0, 0)))
            {
                result.Store4(resultOffset + 16, asuint(ref));
            }

            // Compare current texel to the reference value.
            // Produce bit mask in bits 0-3 indicating which color channels differ.
            float4 val = input[texel];

            mask |= val.x != ref.x ? 1 : 0;
            mask |= val.y != ref.y ? 2 : 0;
            mask |= val.z != ref.z ? 4 : 0;
            mask |= val.w != ref.w ? 8 : 0;

            // Compute range flags for each color channel.
            // Produce flags in bits 4-19 for | 0...0 | A | B | G | R | 0000 | (each group is 4 bits).
            uint range = 0;
            for (int i = 3; i >= 0; i--)
            {
                if (val[i] > 0.f) range |= 1;
                if (val[i] < 0.f) range |= 2;
                if (isinf(val[i])) range |= 4;
                if (isnan(val[i])) range |= 8;
                range <<= 4;
            }

            // Store range and const flags together.
            mask |= range;

            minVal = val;
            maxVal = val;
            valid = true;
        }

        if (WaveActiveAllTrue(!valid)) return;

        // Reduce the result across the warp.
        // These instructions are purposely placed outside of control flow.
        mask = WaveActiveBitOr(mask);
        minVal = WaveActiveMin(minVal);
        maxVal = WaveActiveMax(maxVal);

        // Let the first lane write out result atomically.
        if (WaveIsFirstLane())
        {
            if (mask != 0) result.InterlockedOr(resultOffset, mask);

            // Clamp to zero since we're using integer atomics.
            minVal = max(minVal, float4(0));
            maxVal = max(maxVal, float4(0));

            result.InterlockedMin(resultOffset + 32, asuint(minVal.x));
            result.InterlockedMin(resultOffset + 36, asuint(minVal.y));
            result.InterlockedMin(resultOffset + 40, asuint(minVal.z));
            result.InterlockedMin(resultOffset + 44, asuint(minVal.w));

            result.InterlockedMax(resultOffset + 48, asuint(maxVal.x));
            result.InterlockedMax(resultOffset + 52, asuint(maxVal.y));
            result.InterlockedMax(resultOffset + 56, asuint(maxVal.z));
            result.InterlockedMax(resultOffset + 60, asuint(maxVal.w));
        }
    }
};

ConstantBuffer<TextureAnalyzer> gTextureAnalyzer;

[numthreads(256, 1, 1)]
void clear(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTextureAnalyzer.clear(dispatchThreadId.x);
}

[numthreads(16, 16, 1)]
void analyze(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTextureAnalyzer.analyze(dispatchThreadId.xy);
}
