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
    float4x4 reprojMatrix;
    float4x4 invProjMatThis;
    float4x4 invProjMatLast;

    int2 causticsDim;
    int2 gBufferDim;

    float blendWeight;
    float normalKernel;
    float depthKernel;
    float colorKernel;
};

Texture2D depthTexThis;
Texture2D depthTexLast;

Texture2D normalTexThis;
Texture2D normalTexLast;

Texture2D causticsTexLast;
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

    float depth = depthTexThis[gBufferPixelPos].r;
    float3 normal = normalTexThis[gBufferPixelPos].rgb;
    float4 color = causticsTexThis[causticsPixelPos];

    float4 ndc = float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
    float4 ndcLast = mul(ndc, reprojMatrix);
    ndcLast.xyz /= ndcLast.w;
    float2 uvLast = (ndcLast.xy * float2(1, -1) + 1) * 0.5;
    uint2 pixelPosLast = uvLast * causticsDim;
    uint2 gBufferPixelPosLast = uvLast * gBufferDim;

    float depthLast = depthTexLast[gBufferPixelPosLast].r;
    float3 normalLast = normalTexLast[gBufferPixelPosLast].rgb;
    float4 colorLast = causticsTexLast.SampleLevel(gSampler, uvLast, 0);

    float viewDepthThis = toViewSpace(invProjMatThis, depth);
    float viewDepthLast = toViewSpace(invProjMatLast, depthLast);

    float weight = 0;
    if (depthLast < 1.0f)
    {
        float normalDiff = (1.0f - saturate(dot(normal, normalLast))) / normalKernel * 100;
        float depthDiff = abs(viewDepthThis - viewDepthLast) / depthKernel;
        float colorDiff = length(colorLast - color) / colorKernel;
        weight = blendWeight * exp(-1.0f * (normalDiff * normalDiff + depthDiff * depthDiff + colorDiff * colorDiff));
    }

    //causticsTexThis[causticsPixelPos] = float4(normalLast, 1);
    causticsTexThis[causticsPixelPos] = color * (1 - weight) +
        //causticsTexLast[pixelPosLast] * weight;
        colorLast * weight;
}
