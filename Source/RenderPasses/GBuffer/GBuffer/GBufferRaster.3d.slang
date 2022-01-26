/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Math.MathHelpers;
import GBufferHelpers;
import Rendering.Materials.TexLODHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

// GBuffer channels
struct GBufferPSOut
{
    float4 posW         : SV_TARGET0;
    float4 normW        : SV_TARGET1;
    float4 tangentW     : SV_TARGET2;
    float4 faceNormalW  : SV_TARGET3;
    float2 texC         : SV_TARGET4;
    float4 texGrads     : SV_TARGET5;
    float2 mvec         : SV_TARGET6;
    uint4  mtlData      : SV_TARGET7;
};

// GBufferRaster channels
RasterizerOrderedTexture2D<PackedHitInfo> gVBuffer;
RasterizerOrderedTexture2D<float4> gDiffOpacity;
RasterizerOrderedTexture2D<float4> gSpecRough;
RasterizerOrderedTexture2D<float4> gEmissive;
RasterizerOrderedTexture2D<float4> gViewW;
RasterizerOrderedTexture2D<float2> gPosNormalFwidth;
RasterizerOrderedTexture2D<float2> gLinearZAndDeriv;

#define is_valid(name) (is_valid_##name != 0)

VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}

float2 computeMotionVector(const VSOut vsOut, const int2 ipos)
{
    float2 pixelPos = ipos + float2(0.5, 0.5); // Current sample in pixel coords.
    float4 prevPosH = vsOut.prevPosH; // Sample in previous frame in clip space coords, no jittering applied.
    return calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
}

[earlydepthstencil]
GBufferPSOut psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics)
{
    // Using vOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(vsOut.posH.xy);

    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    let lod = ImplicitLodTextureSampler();

#if USE_ALPHA_TEST
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod)) discard;
#endif
    const float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir, lod);

#if ADJUST_SHADING_NORMALS
    adjustShadingNormal(sd, v);
#endif

    // Create BSDF instance.
    let bsdf = gScene.materials.getBSDF(sd, lod);

    const GBufferData gbuf = prepareGBufferData(sd, bsdf);

    const float4 texGrads = float4(ddx(sd.uv), ddy(sd.uv));
    const float2 mvec = computeMotionVector(vsOut, ipos);

    GBufferPSOut psOut = {};

    // Store render target outputs.
    if (is_valid(gPosW))            psOut.posW          = gbuf.posW;
    if (is_valid(gNormW))           psOut.normW         = gbuf.normW;
    if (is_valid(gTangentW))        psOut.tangentW      = gbuf.tangentW;
    if (is_valid(gFaceNormalW))     psOut.faceNormalW   = gbuf.faceNormalW;
    if (is_valid(gTexC))            psOut.texC          = gbuf.texC;
    if (is_valid(gTexGrads))        psOut.texGrads      = texGrads;
    if (is_valid(gMotionVector))    psOut.mvec          = mvec;
    if (is_valid(gMaterialData))    psOut.mtlData       = gbuf.mtlData;

    // Store UAV outputs.
    if (is_valid(gViewW))           gViewW[ipos]        = float4(sd.V, 0.f);
    if (is_valid(gDiffOpacity))     gDiffOpacity[ipos]  = gbuf.diffuseOpacity;
    if (is_valid(gSpecRough))       gSpecRough[ipos]    = gbuf.specRough;
    if (is_valid(gEmissive))        gEmissive[ipos]     = gbuf.emissive;

    // Length of derivatives of position and normal
    if (is_valid(gPosNormalFwidth))
    {
        gPosNormalFwidth[ipos] = float2(length(fwidth(sd.posW)), length(fwidth(sd.N)));
    }

    // Linear z and its derivative
    if (is_valid(gLinearZAndDeriv))
    {
        const float linearZ = vsOut.posH.z * vsOut.posH.w;
        gLinearZAndDeriv[ipos] = float2(linearZ, max(abs(ddx(linearZ)), abs(ddy(linearZ))));
    }

    // Note on barycentrics:
    // The barycentric weights provided to pixel shader correspond to vertices A, B, C of the rasterized triangle.
    // For triangle strips, every odd primitive has the order for vertices B and C flipped. We don't handle triangle
    // strips as DXR does not support them and Falcor uses triangle lists exclusively in its scene representation.
    // DXR intersection attributes store barycentric weights in a float2 for triangle vertices B and C.
    // This is what we store in the hit info.

    // Store hit information.
    if (is_valid(gVBuffer))
    {
        TriangleHit triangleHit;
        triangleHit.instanceID = vsOut.instanceID;
        triangleHit.primitiveIndex = triangleIndex;
        triangleHit.barycentrics = barycentrics.yz;
        gVBuffer[ipos] = triangleHit.pack();
    }

    return psOut;
}
