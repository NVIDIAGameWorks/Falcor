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
#include "VertexAttrib.slangh"
__exported import Scene.Scene;
__exported import Scene.Shading;

struct VSIn
{
    // Packed vertex attributes, see PackedStaticVertexData
    float3 pos                      : POSITION;
    float3 packedNormalTangent      : PACKED_NORMAL_TANGENT;
    float2 texC                     : TEXCOORD;

    // Other vertex attributes
    uint meshInstanceID             : DRAW_ID;
    float3 prevPos                  : PREV_POSITION;

    StaticVertexData unpack()
    {
        PackedStaticVertexData v;
        v.position = pos;
        v.packedNormalTangent = packedNormalTangent;
        v.texCrd = texC;
        return v.unpack();
    }
};

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct VSOut
{
    INTERPOLATION_MODE float3 normalW    : NORMAL;
    INTERPOLATION_MODE float4 tangentW   : TANGENT;
    INTERPOLATION_MODE float2 texC       : TEXCRD;
    INTERPOLATION_MODE float3 posW       : POSW;
    INTERPOLATION_MODE float3 colorV     : COLOR;
    INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;

    // Per-triangle data
    nointerpolation uint meshInstanceID  : DRAW_ID;
    nointerpolation uint materialID      : MATERIAL_ID;

    float4 posH : SV_POSITION;
};

VSOut defaultVS(VSIn vIn)
{
    VSOut vOut;
    float4x4 worldMat = gScene.getWorldMatrix(vIn.meshInstanceID);
    float4 posW = mul(float4(vIn.pos, 1.f), worldMat);
    vOut.posW = posW.xyz;
    vOut.posH = mul(posW, gScene.camera.getViewProj());

    vOut.meshInstanceID = vIn.meshInstanceID;
    vOut.materialID = gScene.getMaterialID(vIn.meshInstanceID);

    vOut.texC = vIn.texC;
    vOut.normalW = mul(vIn.unpack().normal, gScene.getInverseTransposeWorldMatrix(vIn.meshInstanceID));
    float4 tangent = vIn.unpack().tangent;
    vOut.tangentW = float4(mul(tangent.xyz, (float3x3)gScene.getWorldMatrix(vIn.meshInstanceID)), tangent.w);

    float4 prevPosW = mul(float4(vIn.prevPos, 1.f), gScene.getPrevWorldMatrix(vIn.meshInstanceID));
    vOut.prevPosH = mul(prevPosW, gScene.camera.data.prevViewProjMatNoJitter);

  return vOut;
}

/** Setup vertex data based on interpolated vertex attributes.
    \param[in] vsOut Interpolated vertex attributes.
    \param[in] faceNormalW Face normal in world space (normalized).
    \return VertexData struct.
*/
VertexData prepareVertexData(VSOut vsOut, float3 faceNormalW)
{
    VertexData v;
    v.posW = vsOut.posW;
    v.texC = vsOut.texC;
    v.normalW = normalize(vsOut.normalW);
    v.faceNormalW = faceNormalW;
    // Handle invalid tangents gracefully (avoid NaN from normalization).
    v.tangentW.xyz = vsOut.tangentW.w != 0.f ? normalize(vsOut.tangentW.xyz) : float3(0, 0, 0);
    v.tangentW.w = sign(vsOut.tangentW.w); // Preserve zero to indicate invalid tangent.
    return v;
}

/** Helper function that prepares the ShadingData struct based on VSOut.
    This version uses implicit LOD and fetches material data directly from the scene.
*/
ShadingData prepareShadingData(VSOut vsOut, uint triangleIndex, float3 viewDir)
{
    float3 faceNormal = gScene.getFaceNormalW(vsOut.meshInstanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    return prepareShadingData(v, vsOut.materialID, gScene.materials[vsOut.materialID], gScene.materialResources[vsOut.materialID], viewDir);
}
