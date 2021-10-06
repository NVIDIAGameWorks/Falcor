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
import Scene.HitInfo;
import Utils.Math.MathHelpers;
import GBufferHelpers;
import Experimental.Scene.Material.TexLODHelpers;

// UAV output channels
RasterizerOrderedTexture2D<PackedHitInfo> gVBuffer;
RasterizerOrderedTexture2D<float2> gMotionVectors;
RasterizerOrderedTexture2D<float4> gRoughness;
RasterizerOrderedTexture2D<float4> gMetallic;
RasterizerOrderedTexture2D<float4> gFaceNormalW;
RasterizerOrderedTexture2D<float4> gViewW;
RasterizerOrderedTexture2D<float2> gPosNormalFwidth;
RasterizerOrderedTexture2D<float2> gLinearZAndDeriv;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

#define is_valid(name) (is_valid_##name != 0)

struct GBufferPSOut
{
    float4 posW             : SV_TARGET0;
    float4 normW            : SV_TARGET1;
    float4 tangentW         : SV_TARGET2;
    float4 texC             : SV_TARGET3;
    float4 diffuseOpacity   : SV_TARGET4;
    float4 specRough        : SV_TARGET5;
    float4 emissive         : SV_TARGET6;
    float4 matlExtra        : SV_TARGET7;
};

VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}

[earlydepthstencil]
GBufferPSOut psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics)
{
    GBufferPSOut psOut;

    // Using vOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(vsOut.posH.xy);

    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);

#if USE_ALPHA_TEST
    if (alphaTest(v, gScene.materials[vsOut.materialID], gScene.materialResources[vsOut.materialID])) discard;
#endif
    float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = prepareShadingData(v, vsOut.materialID, gScene.materials[vsOut.materialID], gScene.materialResources[vsOut.materialID], viewDir);

#if ADJUST_SHADING_NORMALS
    adjustShadingNormal(sd, v);
#endif

    GBuffer gbuf = storeGBufferOutput(sd, v);

    // Store render target outputs.
    psOut.posW           = gbuf.posW;
    psOut.normW          = gbuf.normW;
    psOut.tangentW       = gbuf.tangentW;
    psOut.texC           = gbuf.texC;
    psOut.diffuseOpacity = gbuf.diffuseOpacity;
    psOut.specRough      = gbuf.specRough;
    psOut.emissive       = gbuf.emissive;
    psOut.matlExtra      = gbuf.matlExtra;

    // Store UAV outputs.
    if (is_valid(gFaceNormalW))
    {
        gFaceNormalW[ipos] = float4(sd.faceN, 0);
    }

    if (is_valid(gViewW))
    {
        gViewW[ipos] = float4(sd.V, 0);
    }

    // Compute motion vectors.
    if (is_valid(gMotionVectors))
    {
        const float2 pixelPos = ipos + float2(0.5, 0.5); // Current sample in pixel coords.
        const float4 prevPosH = vsOut.prevPosH; // Sample in previous frame in clip space coords, no jittering applied.
        const float2 mv = calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
        gMotionVectors[ipos] = mv;
    }

    if (is_valid(gRoughness))
    {
        gRoughness[ipos] = float4(float3(sd.linearRoughness), 1.0f);
    }

    if (is_valid(gMetallic))
    {
        gMetallic[ipos] = float4(float3(sd.metallic), 1.0f);
    }

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
        gVBuffer[ipos] = HitInfo(triangleHit).pack();
    }
    return psOut;
}
