/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Common.hlsl"

cbuffer PerFrameCB
{
    int2 causticsDim;
    int2 gBufferDim;
    float normalKernel;
    float depthKernel;
    float colorKernel;
    float screenKernel;
    int passID;
    float gSmallPhotonColorScale;
};

Texture2D depthTexThis;
Texture2D normalTexThis;

Texture2D<uint> smallPhotonTex;

//Texture2D   causticsTexLast;
RWTexture2D causticsTexThis;

SamplerState gSampler;

[numthreads(16, 16, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint2 causticsPixelPos = threadIdx.xy;
    if (any(causticsPixelPos.xy >= causticsDim))
    {
        return;
    }
    float2 uv = float2(causticsPixelPos.xy + 0.5) / causticsDim;
    uint2 gBufferPixelPos = uv * gBufferDim;

    float depth0 = depthTexThis[gBufferPixelPos].r;
    float2 depthGrad;
    depthGrad.x = depthTexThis[gBufferPixelPos + uint2(1, 0)].r;
    depthGrad.y = depthTexThis[gBufferPixelPos + uint2(0, 1)].r;
    float4 color0 = causticsTexThis[causticsPixelPos].rgba;
    float luminance0 = getLuminance(color0.rgb);

    float3 normal0 = normalTexThis[gBufferPixelPos].rgb;
    float totalWeight = 0;
    float4 totalColor = 0;
    int step = (1 << passID);
    float hArray[5] = { 1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16 };
    [unroll]
    for (int i = 0; i < 5; i++)
    {
        float hi = hArray[i];
        for (int j = 0; j < 5; j++)
        {
            float hj = hArray[j];
            float h = pow(hi * hj, screenKernel);

            uint2 offset = uint2(i - 2, j - 2);
            uint2 samplePos = gBufferPixelPos + offset * step;
            float depth = depthTexThis[samplePos].r;
            float3 normal = normalTexThis[samplePos].rgb;
            float4 color = causticsTexThis[causticsPixelPos + offset];
            uint smallPhotonData = smallPhotonTex[causticsPixelPos + offset];
            float4 smallPhotonClr = decompressColor(smallPhotonData, gSmallPhotonColorScale);
            color += smallPhotonClr;
            float luminance = getLuminance(color.rgb);
            if (depth == 1)
            {
                continue;
            }

            float depthWeight = exp(-abs((depth - depth0) / (depthKernel * dot(depthGrad, offset) + 0.00001)));
            float normalWeight = pow(saturate(dot(normal, normal0)), normalKernel);
            float colorWeight = exp(-abs(luminance - luminance0) / colorKernel);

            float weight = h * depthWeight * normalWeight * colorWeight;
            totalWeight += weight;
            totalColor += weight * color;
        }
    }
    if (totalWeight == 0)
    {
        totalColor = color0;
    }
    else
    {
        totalColor /= totalWeight;
    }
    causticsTexThis[causticsPixelPos] = totalColor; // float4(totalColor.rgb, 1);
}
