/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "VertexAttrib.h"
__import ShaderCommon;

struct VertexIn
{
    float4 pos         : POSITION;
#ifdef HAS_NORMAL
    float3 normal      : NORMAL;
#endif
#ifdef HAS_BITANGENT
    float3 bitangent   : BITANGENT;
#endif
#ifdef HAS_TEXCRD
    float2 texC        : TEXCOORD;
#endif
#ifdef HAS_LIGHTMAP_UV
    float2 lightmapC   : LIGHTMAP_UV;
#endif
#ifdef HAS_COLORS
    float3 color       : DIFFUSE_COLOR;
#endif
#ifdef _VERTEX_BLENDING
    float4 boneWeights : BONE_WEIGHTS;
    uint4  boneIds     : BONE_IDS;
#endif
#ifdef HAS_PREV_POSITION
    float4 prevPos     : PREV_POSITION;
#endif
    uint instanceID : SV_INSTANCEID;
};

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct VertexOut
{
    INTERPOLATION_MODE float3 normalW    : NORMAL;
    INTERPOLATION_MODE float3 bitangentW : BITANGENT;
    INTERPOLATION_MODE float2 texC       : TEXCRD;
    INTERPOLATION_MODE float3 posW       : POSW;
    INTERPOLATION_MODE float3 colorV     : COLOR;
    INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;
    INTERPOLATION_MODE float2 lightmapC  : LIGHTMAPUV;
    float4 posH : SV_POSITION;
#ifdef _SINGLE_PASS_STEREO
    INTERPOLATION_MODE float4 rightEyePosS : NV_X_RIGHT;
    uint4 viewportMask : NV_VIEWPORT_MASK;
    uint renderTargetIndex : SV_RenderTargetArrayIndex;
#endif
};

float4x4 getWorldMat(VertexIn vIn)
{
    float4x4 worldMat = gWorldMat[vIn.instanceID];

#ifdef _VERTEX_BLENDING
    worldMat = mul(getBlendedBoneMat(vIn.boneWeights, vIn.boneIds), worldMat);
#endif

    return worldMat;
}

float3x3 getWorldInvTransposeMat(VertexIn vIn)
{
    float3x3 worldInvTransposeMat = (float3x3)gWorldInvTransposeMat[vIn.instanceID];

#ifdef _VERTEX_BLENDING
    worldInvTransposeMat = mul(getBlendedInvTransposeBoneMat(vIn.boneWeights, vIn.boneIds), worldInvTransposeMat);
#endif

    return worldInvTransposeMat;
}

VertexOut defaultVS(VertexIn vIn)
{
    VertexOut vOut;
    float4x4 worldMat = getWorldMat(vIn);
    float4 posW = mul(vIn.pos, worldMat);
    vOut.posW = posW.xyz;
    vOut.posH = mul(posW, gCamera.viewProjMat);

#ifdef HAS_TEXCRD
    vOut.texC = vIn.texC;
#else
    vOut.texC = 0;
#endif

#ifdef HAS_COLORS
    vOut.colorV = vIn.color;
#else
    vOut.colorV = 0;
#endif

#ifdef HAS_NORMAL
    vOut.normalW = mul(vIn.normal, getWorldInvTransposeMat(vIn)).xyz;
#else
    vOut.normalW = 0;
#endif

#ifdef HAS_BITANGENT
    vOut.bitangentW = mul(vIn.bitangent, (float3x3)getWorldMat(vIn));
#else
    vOut.bitangentW = 0;
#endif

#ifdef HAS_LIGHTMAP_UV
    vOut.lightmapC = vIn.lightmapC;
#else
    vOut.lightmapC = 0;
#endif

#ifdef HAS_PREV_POSITION
    float4 prevPos = vIn.prevPos;
#else
    float4 prevPos = vIn.pos;
#endif
    float4 prevPosW = mul(prevPos, gPrevWorldMat[vIn.instanceID]);
    vOut.prevPosH = mul(prevPosW, gCamera.prevViewProjMat);

#ifdef _SINGLE_PASS_STEREO
    vOut.rightEyePosS = mul(posW, gCamera.rightEyeViewProjMat).x;
    vOut.viewportMask = 0x00000001;
    vOut.renderTargetIndex = 0;
#endif

  return vOut;
}
