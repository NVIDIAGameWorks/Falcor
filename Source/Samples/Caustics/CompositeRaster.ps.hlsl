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

import Scene.Raster;
import Scene.Lights.LightData;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

#include "Common.hlsl"

SamplerState gPointSampler : register(s1);

Texture2D gDepthTex;
Texture2D gNormalTex;
Texture2D gDiffuseTex;
Texture2D gSpecularTex;
Texture2D gPhotonTex;
Texture2D gRaytracingTex;
Texture2D<float4> gRayTex;
Texture1D<uint> gStatisticsTex;
Texture2D<uint> gSmallPhotonTex;
StructuredBuffer<uint4> gRayCountQuadTree;
StructuredBuffer<PixelInfo> gPixelInfo;

// Debug modes
#define ShowDepth                  1
#define ShowNormal                 2
#define ShowDiffuse                3
#define ShowSpecular               4
#define ShowPhoton                 5
#define ShowWorld                  6
#define ShowRoughness              7
#define ShowRayTex                 8
#define ShowRayTracing             9
#define ShowAvgScreenArea         10
#define ShowAvgScreenAreaVariance 11
#define ShowCount                 12
#define ShowTotalPhoton           13
#define ShowRayCountMipTex        14
#define ShowPhotonDensity         15
#define ShowSmallPhotonTex        16
#define ShowSmallPhotonCount      17

cbuffer PerImageCB
{
    // Lighting params
    LightData gLightData[16]; //Scene.Lights.LightData, inside LightHelpers
    float4x4 gInvWvpMat;
    float4x4 gInvPMat;

    float3 gCameraPos;
    uint gNumLights;

    int2 screenDim;
    int2 dispatchSize;

    uint gDebugMode;
    float gMaxPixelArea;
    int gRayTexScale;
    uint gStatisticsOffset;

    float gMaxPhotonCount;
    int gRayCountMip;
    float gSmallPhotonColorScale;
};

//VSOut vsMain(VSIn vIn)
//{
//    return defaultVS(vIn);
//}

float4 psMain( /*VSOut vsOut*/float2 texC : TEXCOORD, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    // Or vsOut.texC
    float depth = gDepthTex.Sample(gPointSampler, texC).r;
    float4 screenPnt = float4(texC * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), depth, 1.0f); // NDC space i am not mistaken
    float4 worldPnt = mul(screenPnt, gInvWvpMat);
    worldPnt /= worldPnt.w;
    float4 normalVal = gNormalTex.Sample(gPointSampler, texC);
    float4 diffuseVal = gDiffuseTex.Sample(gPointSampler, texC);
    float4 specularVal = gSpecularTex.Sample(gPointSampler, texC);
    float4 rtColor = gRaytracingTex.Sample(gPointSampler, texC);

    float4 color = rtColor;
    if (gDebugMode == ShowDepth)
    {
        float4 viewPnt = mul(screenPnt, gInvPMat);
        viewPnt /= viewPnt.w;
        float viewDepth = (gInvPMat[2][2] * depth + gInvPMat[3][2]) / (gInvPMat[2][3] * depth + gInvPMat[3][3]);
        color = float4(abs(viewDepth.xxx) * 0.01f, 1.0f);
    }
    else if (gDebugMode == ShowNormal)
        color = normalVal;
    else if (gDebugMode == ShowDiffuse)
        //color = gDiffuseTex.Sample(gPointSampler, texC);
        color = diffuseVal;
    else if (gDebugMode == ShowSpecular)
        //color = gSpecularTex.Sample(gPointSampler, texC);
        color = specularVal;
    else if (gDebugMode == ShowWorld)
        color = frac(worldPnt * 0.01f + 0.01f);
    else if (gDebugMode == ShowRoughness)
        color = diffuseVal.a;
    else if (gDebugMode == ShowRayTex)
    {
        int2 screenPixel = texC * screenDim;
        //float2 uv = clamp(screenPixel / 2048, 0, 1);
        //color = gRayTex.Sample(gPointSampler, uv);
        float4 v = gRayTex.Load(int3(screenPixel.xy / gRayTexScale, 0));
        //color = float4(v >> 16, 0, 0, v & 0xff);
        //color /= color.w;
        //color.r /= gMaxPixelArea;
        if (v.r <= gMaxPixelArea)
            color = float4(v.rgb / gMaxPixelArea, 1.0f);
        else
            color = float4(1.0f, 0.0f, 1.0f, 1.0f);
    }
    else if (gDebugMode == ShowRayCountMipTex)
    {
        int2 screenPixel = texC * screenDim;
        int2 texelPos = screenPixel.xy / gRayTexScale;
        if (all(texelPos < getMipSize(gRayCountMip)))
        {
            int offset = getTextureOffset(texelPos.xy, gRayCountMip);
            uint4 v = gRayCountQuadTree[offset]; // .Load(int3(texelPos, gRayCountMip));
            uint sum = v.a;
            if (sum <= gMaxPixelArea)
                color = float4((sum / gMaxPixelArea).xxx, 1.0f);
            else
                color = float4(1.0f, 0.0f, 1.0f, 1.0f);
        }
    }
    else if (gDebugMode == ShowRayTracing)
    {
        color = rtColor;
    }
    else if (gDebugMode == ShowAvgScreenArea || gDebugMode == ShowAvgScreenAreaVariance || gDebugMode == ShowCount)
    {
        int2 screenPixel = texC * screenDim;
        int2 texelPos = screenPixel.xy / gRayTexScale;
        if (all(texelPos < dispatchSize))
        {
            PixelInfo info = gPixelInfo[texelPos.y * dispatchSize.x + texelPos.x];
            float value = 0.0f;
            float avgArea = float(info.screenArea) / float(info.count + 0.001f);
            float avgAreaSq = float(info.screenAreaSq) / float(info.count + 0.001f);
            if (gDebugMode == ShowAvgScreenArea)
            {
                value = avgArea;
            }
            else if (gDebugMode == ShowAvgScreenAreaVariance)
            {
                value = sqrt(avgAreaSq - avgArea * avgArea);
            }
            else if (gDebugMode == ShowCount)
            {
                value = info.count;
            }
            float maxVal = gMaxPixelArea;
            if (value <= maxVal)
                color = float4(value.xxx / maxVal, 1.0f);
            else
                color = float4(1.0f, 0.0f, 1.0f, 1.0f);
        }
    }
    else if (gDebugMode == ShowTotalPhoton)
    {
        texC.y = 1.0f - texC.y;
        int2 screenPixel = texC * screenDim;
        int texelPos = (gStatisticsOffset - screenPixel.x + screenDim.x) % screenDim.x;
        uint valueI = gStatisticsTex[texelPos].r;
        valueI = valueI * screenDim.y / gMaxPhotonCount;
        uint segmentWidth = screenDim.y / 4;
        float4 graphColor;
        float alpha = 0.3f;
        if (screenPixel.y % segmentWidth == 0)
            graphColor = float4(0.2f, 0.2f, 0.2f, alpha);
        else if (screenPixel.y <= valueI)
            graphColor = float4(1.0f, 0.0f, 0.0f, alpha);
        else
            graphColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
        color.rgb = lerp(rtColor.rgb, graphColor.rgb, graphColor.a);
        color.a = 1.0f;
    }
    else if (gDebugMode == ShowSmallPhotonTex)
    {
        texC.y = 1.0f - texC.y;
        int2 screenPixel = texC * screenDim;
        uint v = gSmallPhotonTex.Load(int3(screenPixel.xy, 0));
        float4 photonClr = decompressColor(v, gSmallPhotonColorScale);
        color = float4(photonClr.rgb, 1);
    }
    else if (gDebugMode == ShowSmallPhotonCount)
    {
        texC.y = 1.0f - texC.y;
        int2 screenPixel = texC * screenDim;
        uint v = gSmallPhotonTex.Load(int3(screenPixel.xy, 0));
        float4 photonClr = decompressColor(v, gSmallPhotonColorScale);
        if (photonClr.a <= gMaxPixelArea)
            color = float4(photonClr.aaa / gMaxPixelArea, 1);
        else
            color = float4(1.0f, 0.0f, 1.0f, 1.0f);
        color.rgb += rtColor.rgb; // *0.5;
    }
    else
    {
        let lod = ImplicitLodTextureSampler();
        ShadingData sd = { }; // initShadingData();
        sd.posW = worldPnt.xyz;
        sd.V = normalize(gCameraPos - sd.posW);
        //sd.N = normalVal.xyz;
        //sd.T = normalize(cross(sd.V, sd.N));
        //sd.B = normalize(cross(sd.T, sd.N));
        sd.uv = texC;
        //sd.faceN = sd.N;
        float4 tangentW = float4(1, 0, 0, 1);
        bool valid;
        sd.frame = ShadingFrame::createSafe(normalVal.xyz, tangentW, valid);
        sd.faceN = sd.frame.N;
        sd.frontFacing = dot(sd.V, sd.faceN) >= 0.0f;
        //sd.NdotV = saturate(dot(sd.V, sd.N));
        //sd.linearRoughness = diffuseVal.a;
        //sd.roughness = sd.linearRoughness * sd.linearRoughness;
        //sd.specular = specularVal.xyz;
        //sd.diffuse = diffuseVal.rgb;
        //sd.opacity = specularVal.a;
        color = float4(0.0f, 0.0f, 0.0f, 1.0f);
        for (uint l = 0; l < gScene.getLightCount(); l++)
        {
            const LightData light = gScene.getLight(l);
            float3 L = normalize(light.posW - sd.posW);
            //ShadingResult sr = evalMaterial(sd, gLightData[l], 1);
            color.rgb += diffuseVal.rgb * saturate(dot(L, sd.faceN)); // sr.color.rgb;
        }

        float4 photonClr = gPhotonTex.Sample(gPointSampler, texC);
        if (gDebugMode == ShowPhoton)
        {
            color.rgb = photonClr.rgb; // lerp(color.rgb, photonClr.rgb, photonClr.a);
        }
        else if (gDebugMode == ShowPhotonDensity)
        {
            if (photonClr.a <= gMaxPixelArea)
                color.rgb = photonClr.aaa / gMaxPixelArea;
            else
                color = float4(1, 0, 1, 1);
        }
        else
        {
            color.rgb += photonClr.rgb * diffuseVal.rgb; // sd.diffuse;
        }
    }

    return color;
}
