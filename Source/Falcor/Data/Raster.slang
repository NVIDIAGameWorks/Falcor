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
#include "VertexAttrib.h"
import Shading;
import Scene;

struct VSIn
{
    float4 pos          : POSITION;
    float3 normal       : NORMAL;
    float3 bitangent    : BITANGENT;
    float2 texC         : TEXCOORD;
    uint meshInstanceID : DRAW_ID;
    float4 prevPos      : PREV_POSITION;
//#ifdef HAS_LIGHTMAP_UV
//    float2 lightmapC    : LIGHTMAP_UV;
//#endif
};

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct VSOut
{
    INTERPOLATION_MODE float3 normalW    : NORMAL;
    INTERPOLATION_MODE float3 bitangentW : BITANGENT;
    INTERPOLATION_MODE float2 texC       : TEXCRD;
    INTERPOLATION_MODE float3 posW       : POSW;
    INTERPOLATION_MODE float3 colorV     : COLOR;
    INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;

    // Per-triangle data
    nointerpolation uint meshInstanceID  : DRAW_ID;
    nointerpolation uint materialID      : MATERIAL_ID;

    float4 posH : SV_POSITION;

#ifdef _SINGLE_PASS_STEREO
    INTERPOLATION_MODE float4 rightEyePosS : NV_X_RIGHT;
    uint4 viewportMask : NV_VIEWPORT_MASK;
    uint renderTargetIndex : SV_RenderTargetArrayIndex;
#endif
};

VSOut defaultVS(VSIn vIn)
{
    VSOut vOut;
    float4x4 worldMat = gScene.getWorldMatrix(vIn.meshInstanceID);
    float4 posW = mul(vIn.pos, worldMat);
    vOut.posW = posW.xyz;
    vOut.posH = mul(posW, gScene.camera.viewProjMat);

    vOut.meshInstanceID = vIn.meshInstanceID;
    vOut.materialID = gScene.getMaterialID(vIn.meshInstanceID);

    vOut.texC = vIn.texC;
    vOut.normalW = mul(vIn.normal, (float3x3)gScene.getInverseTransposeWorldMatrix(vIn.meshInstanceID)).xyz;
    vOut.bitangentW = mul(vIn.bitangent,(float3x3) gScene.getWorldMatrix(vIn.meshInstanceID));

    float4 prevPosW = mul(vIn.prevPos, gScene.getPrevWorldMatrix(vIn.meshInstanceID));
    vOut.prevPosH = mul(prevPosW, gScene.camera.prevViewProjMatNoJitter);

#ifdef _SINGLE_PASS_STEREO
    vOut.rightEyePosS = mul(posW, gScene.camera.rightEyeViewProjMat).x;
    vOut.viewportMask = 0x00000001;
    vOut.renderTargetIndex = 0;
#endif

  return vOut;
}

VertexData prepareVertexData(VSOut vsOut, float3 faceNormalW)
{
    VertexData v;
    v.posW = vsOut.posW;
    v.normalW = vsOut.normalW;
    v.bitangentW = vsOut.bitangentW;
    v.texC = vsOut.texC;
    v.faceNormalW = faceNormalW;
    return v;
}

/** Helper function that prepares the ShadingData struct based on VSOut.
    This version uses implicit LOD and fetches material data directly from the scene.
*/
ShadingData prepareShadingData(VSOut vsOut, uint triangleIndex, float3 camPosW)
{
    float3 faceNormal = gScene.getFaceNormalW(vsOut.meshInstanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    return prepareShadingData(v, gScene.materials[vsOut.materialID], camPosW);
}
