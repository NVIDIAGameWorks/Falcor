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
import Utils.Math.MathHelpers;

/** Converts env map from equal-area octahedral mapping to lat-long mapping.
*/
struct EnvMapConverter
{
    Texture2D<float4> src;
    SamplerState srcSampler;

    RWTexture2D<float4> dst;

    uint2 dstDim;

    void execute(uint2 threadID)
    {
        const uint2 pixel = threadID;
        if (any(pixel >= dstDim)) return;

        const float2 latlong = (float2(pixel) + 0.5f) / dstDim;
        const float3 dir = latlong_map_to_world(latlong);
        const float2 oct = ndir_to_oct_equal_area_unorm(dir);

        const float3 L = src.SampleLevel(srcSampler, oct, 0).rgb;
        dst[pixel].rgb = L;
    }
};

ParameterBlock<EnvMapConverter> gEnvMapConverter;

[numthreads(16, 16, 1)]
void main(uint2 threadID : SV_DispatchThreadID)
{
    gEnvMapConverter.execute(threadID);
}
