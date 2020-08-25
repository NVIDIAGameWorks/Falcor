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

/** Compute shader for building a hierarchical importance map from an
    environment map. The result is used by EnvMapSampler.slang for sampling.
*/

import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;

cbuffer CB
{
    uint2 outputDim;            // Resolution of the importance map in texels.
    uint2 outputDimInSamples;   // Resolution of the importance map in samples.
    uint2 numSamples;           // Per-texel subsamples s.xy at finest mip.
    float invSamples;           // 1 / (s.x*s.y).
};

SamplerState gEnvSampler;
Texture2D<float4> gEnvMap;
RWTexture2D<float> gImportanceMap;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= outputDim)) return;

    float L = 0.f;
    for (uint y = 0; y < numSamples.y; y++)
    {
        for (uint x = 0; x < numSamples.x; x++)
        {
            // Compute sample pos p in [0,1)^2 in octahedral map.
            uint2 samplePos = pixel * numSamples + uint2(x, y);
            float2 p = ((float2)samplePos + 0.5f) / outputDimInSamples;

            // Convert p to (u,v) coordinate in latitude-longitude map.
            float3 dir = oct_to_ndir_equal_area_unorm(p);
            float2 uv = world_to_latlong_map(dir);

            // Accumulate the radiance from this sample.
            float3 radiance = gEnvMap.SampleLevel(gEnvSampler, uv, 0).rgb;
            L += luminance(radiance);
        }
    }

    // Store average radiance for this texel.
    gImportanceMap[pixel] = L * invSamples;
}
