/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
__import ShaderCommon;
#include "SSAOData.h"

cbuffer SSAOCB : register(b0)
{
    SSAOData gData;
}

SamplerState gNoiseSampler;
SamplerState gTextureSampler;

Texture2D gDepthTex;
Texture2D gNormalTex;
Texture2D gNoiseTex;

float4 getPosition(float2 uv)
{
    float4 pos;
    pos.x = uv.x * 2.0f - 1.0f;
    pos.y = (1.0f - uv.y) * 2.0f - 1.0f;
    pos.z = gDepthTex.SampleLevel(gTextureSampler, uv, 0).r;
    pos.w = 1.0f;

    float4 posW = mul(pos, gCam.invViewProj);
    posW /= posW.w;

    return posW;
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    if (gDepthTex.SampleLevel(gTextureSampler, texC, 0).r >= 1)
    {
        return 1.0f;
    }

    // Calculate world position of pixel
    float3 posW = getPosition(texC).xyz;

    float3 normal = normalize(gNormalTex.Sample(gTextureSampler, texC).xyz * 2.0f - 1.0f);

    float originDist = length(posW - gCam.position);

    float3 randDir = gNoiseTex.Sample(gNoiseSampler, texC * gData.noiseScale).xyz * 2.0f - 1.0f;

    float3 tangent = normalize(randDir - normal * dot(randDir, normal));
    float3 bitangent = cross(normal, tangent);
    float3x3 tbn = float3x3(tangent, bitangent, normal);

    float occlusion = 0.0f;
    for (uint i = 0; i < gData.kernelSize; i++)
    {
        // Orient sample
        float3 kernelPos = mul(gData.sampleKernel[i].xyz, tbn);

        // Calculate sample world space pos
        float3 samplePosW = posW + (kernelPos * gData.radius);
        float sampleDepth = length(samplePosW - gCam.position);

        // Get screen space pos of sample
        float4 samplePosProj = mul(float4(samplePosW, 1.0f), gCam.viewProjMat);
        samplePosProj /= samplePosProj.w;

        // Sample depth buffer at the same place as sample
        float2 sampleUV = clamp(float2(samplePosProj.x, -samplePosProj.y) * 0.5f + 0.5f, 0.0f, 1.0f);
        float sceneDepth = length(getPosition(sampleUV).xyz - gCam.position);

        float rangeCheck = step(abs(sampleDepth - sceneDepth), gData.radius);
        occlusion += step(sceneDepth, sampleDepth) * rangeCheck;
    }

    float factor = 1 - (occlusion / float(gData.kernelSize));
    return float4(factor.xxx, 1);
}