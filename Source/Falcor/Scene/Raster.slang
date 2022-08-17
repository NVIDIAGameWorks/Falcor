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
#include "VertexAttrib.slangh"

__exported import Scene.Shading;

struct VSIn
{
    // Packed vertex attributes, see PackedStaticVertexData
    float3 pos                              : POSITION;
    float3 packedNormalTangentCurveRadius   : PACKED_NORMAL_TANGENT_CURVE_RADIUS;
    float2 texC                             : TEXCOORD;

    // Other vertex attributes
    uint instanceID                         : DRAW_ID;

    // System values
    uint vertexID                           : SV_VertexID;

    StaticVertexData unpack()
    {
        PackedStaticVertexData v;
        v.position = pos;
        v.packedNormalTangentCurveRadius = packedNormalTangentCurveRadius;
        v.texCrd = texC;
        return v.unpack();
    }
};

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct VSOut
{
    INTERPOLATION_MODE float3 normalW    : NORMAL;      ///< Shading normal in world space (not normalized!).
    INTERPOLATION_MODE float4 tangentW   : TANGENT;     ///< Shading tangent in world space (not normalized!).
    INTERPOLATION_MODE float2 texC       : TEXCRD;      ///< Texture coordinate.
    INTERPOLATION_MODE float3 posW       : POSW;        ///< Position in world space.
    INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;    ///< Position in clip space for the previous frame.

    // Per-triangle data
    nointerpolation GeometryInstanceID instanceID   : INSTANCE_ID;  ///< Geometry instance ID.
    nointerpolation uint materialID                 : MATERIAL_ID;  ///< Material ID.

    float4 posH : SV_POSITION;
};

VSOut defaultVS(VSIn vIn)
{
    VSOut vOut;
    const GeometryInstanceID instanceID = { vIn.instanceID };

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(vIn.pos, 1.f)).xyz;
    vOut.posW = posW;
    vOut.posH = mul(gScene.camera.getViewProj(), float4(posW, 1.f));

    vOut.instanceID = instanceID;
    vOut.materialID = gScene.getMaterialID(instanceID);

    vOut.texC = vIn.texC;
    vOut.normalW = mul(gScene.getInverseTransposeWorldMatrix(instanceID), vIn.unpack().normal);
    float4 tangent = vIn.unpack().tangent;
    vOut.tangentW = float4(mul((float3x3)gScene.getWorldMatrix(instanceID), tangent.xyz), tangent.w);

    // Compute the vertex position in the previous frame.
    float3 prevPos = vIn.pos;
    GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    if (instance.isDynamic())
    {
        uint prevVertexIndex = gScene.meshes[instance.geometryIndex].prevVbOffset + vIn.vertexID;
        prevPos = gScene.prevVertices[prevVertexIndex].position;
    }
    float3 prevPosW = mul(gScene.getPrevWorldMatrix(instanceID), float4(prevPos, 1.f)).xyz;
    vOut.prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));

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
    v.tangentW.xyz = normalize(vsOut.tangentW.xyz);
    v.tangentW.w = sign(vsOut.tangentW.w); // Preserve zero to indicate invalid tangent.
    return v;
}

/** Helper function that prepares the ShadingData struct based on VSOut.
    \param[in] vsOut Interpolated vertex attributes.
    \param[in] triangleIndex Triangle index.
    \param[in] viewDir View direction. Points from the shading point towards the viewer.
    \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
    \return Shading data struct.
*/
ShadingData prepareShadingData<L:ITextureSampler>(VSOut vsOut, uint triangleIndex, float3 viewDir, L lod)
{
    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    return gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir, lod);
}

/** Helper function to evaluate alpha testing based on VSOut.
    \param[in] vsOut Interpolated vertex attributes.
    \param[in] triangleIndex Triangle index.
    \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
    \return True if hit should be ignored/discarded.
*/
bool alphaTest<L:ITextureSampler>(VSOut vsOut, uint triangleIndex, L lod)
{
    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    return gScene.materials.alphaTest(v, vsOut.materialID, lod);
}
