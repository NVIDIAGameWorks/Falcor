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
#include "Scene/VertexAttrib.slangh"
import Scene.Raster;
import RenderPasses.CSM.CascadedShadowMap;

layout(binding = 0) SamplerState alphaSampler : register(s0);
layout(binding = 1) texture2D alphaMap : register(t0);

layout(binding = 2) cbuffer AlphaMapCB : register(b1)
{
    float alphaThreshold;
};

layout(binding = 3) cbuffer PerLightCB : register(b0)
{
    CsmData gCsmData;
};

struct ShadowPassPSIn
{
    float2 texC : TexCoord;
    float4 pos : SV_POSITION;
    uint rtIndex : SV_RenderTargetArrayIndex;
};

struct ShadowPassVSOut
{
    float2 texC : TEXCOORD;
#ifdef _APPLY_PROJECTION
    float4 pos : SV_POSITION;
#else
    float4 pos : POSITION;
#endif
};

ShadowPassVSOut vsMain(VSIn vIn)
{
    ShadowPassVSOut vOut;
    float4x4 worldMat = gScene.getWorldMatrix(vIn.meshInstanceID);
    vOut.pos = mul(float4(vIn.pos, 1.f), worldMat);
#ifdef _APPLY_PROJECTION
    vOut.pos = mul(vOut.pos, gScene.camera.getViewProj());
#endif

    vOut.texC = vIn.texC;
    return vOut;
}

//This is just so it wont crash on link if _CASCADE_COUNT is not defined.
#ifndef _CASCADE_COUNT
#define _CASCADE_COUNT 1
#endif
[instance(_CASCADE_COUNT)]
[maxvertexcount(3)]
void gsMain(triangle ShadowPassVSOut input[3], uint InstanceID : SV_GSInstanceID, inout TriangleStream<ShadowPassPSIn> outStream)
{
    ShadowPassPSIn outputData;

    for(int i = 0 ; i < 3 ; i++)
    {
        outputData.pos = mul(input[i].pos, gCsmData.globalMat);
        outputData.pos.xyz /= input[i].pos.w;
        outputData.pos.xyz *= gCsmData.cascadeScale[InstanceID].xyz;
        outputData.pos.xyz += gCsmData.cascadeOffset[InstanceID].xyz;

        outputData.texC = input[i].texC;
        outputData.rtIndex = InstanceID;

        outStream.Append(outputData);
    }

    outStream.RestartStrip();
}

#if defined(_VSM) || defined(_EVSM2)
float2 psMain(ShadowPassPSIn pIn) : SV_TARGET0
#elif defined(_EVSM4)
float4 psMain(ShadowPassPSIn pIn) : SV_TARGET0
#else
void psMain(ShadowPassPSIn pIn)
#endif
{
#ifdef TEST_ALPHA
    float alpha = alphaMap.Sample(alphaSampler, pIn.texC)._ALPHA_CHANNEL;
    if(alpha < alphaThreshold)
    {
        discard;
    }
#endif

    float2 depth = saturate(pIn.pos.zz);

#if defined(_EVSM2) || defined(_EVSM4)
    depth = applyEvsmExponents(depth.x, gCsmData.evsmExponents);
#endif
    float4 outDepth = float4(depth, depth*depth);

#ifdef _EVSM4
    return outDepth.xzyw;
#elif defined(_VSM) || defined(_EVSM2)
    return outDepth.xz;
#else
#endif
}
