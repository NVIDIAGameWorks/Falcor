/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import Scene;
import Shading;
import ShaderCommon;
import Raster;
import GBufferHelpers;

struct GBufferOut
{
    float4 posW             : SV_TARGET0;
    float4 normW            : SV_TARGET1;
    float4 bitangentW       : SV_TARGET2;
    float4 texC             : SV_TARGET3;
    float4 diffuseOpacity   : SV_TARGET4;
    float4 specRough        : SV_TARGET5;
    float4 emissive         : SV_TARGET6;
    float4 matlExtra        : SV_TARGET7;
};

// UAV output channels
IRWTexture2D gFaceNormalW;
IRWTexture2D gMotionVectors;
IRWTexture2D gPosNormalFwidth;
IRWTexture2D gLinearZAndDeriv;

/** Entry point for G-buffer rasterization pixel shader.
*/
[earlydepthstencil]
GBufferOut ps(VSOut vsOut, uint triangleIndex : SV_PrimitiveID /*, float4 pos : SV_Position*/)
{
    // BUG: When compiling with dxcompiler to shader model 6.0, the validation layers complains that SV_Position
    // has overlapping semantic index at 0. This is because GBufVertexOut contains VSOut from Raster.slang,
    // which in turn has a member posH with the the SV_Position semantic. So, we use that here instead.
    // However, when compiling graphics shaders for SM 6.0, we get no output. It's probably not related to this, but could be.
    const float4 pos = vsOut.posH;
    int2 ipos = int2(pos.xy);

    float3 faceNormal = gScene.getFaceNormalW(vsOut.meshInstanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);

    if (alphaTest(v, gScene.materials[vsOut.materialID])) discard;
    ShadingData sd = prepareShadingData(v, gScene.materials[vsOut.materialID], gScene.camera.posW);

    GBuffer gbuf = storeGBufferOutput(sd);

    // Store render target outputs.
    GBufferOut gout;
    gout.posW           = gbuf.posW;
    gout.normW          = gbuf.normW;
    gout.bitangentW     = gbuf.bitangentW;
    gout.texC           = gbuf.texC;
    gout.diffuseOpacity = gbuf.diffuseOpacity;
    gout.specRough      = gbuf.specRough;
    gout.emissive       = gbuf.emissive;
    gout.matlExtra      = gbuf.matlExtra;

    // Store UAV outputs.
    gFaceNormalW[ipos] = float4(sd.faceN, 0);

    // Compute motion vectors.
    const float2 pixelPos = ipos + float2(0.5, 0.5); // Current sample in pixel coords.
    const float4 prevPosH = vsOut.prevPosH; // Sample in previous frame in clip space coords, no jittering applied.
    const float2 mv = calcMotionVector(pixelPos, prevPosH, gParams.frameSize) + float2(gScene.camera.jitterX, -gScene.camera.jitterY); // Remove camera jitter from motion vector
    gMotionVectors[ipos] = float4(mv, 0, 0);

    // Length of derivatives of position and normal
    gPosNormalFwidth[ipos] = float4(length(fwidth(sd.posW)), length(fwidth(sd.N)), 0, 0);

    // Linear z and its derivative
    const float linearZ = vsOut.posH.z * vsOut.posH.w;
    gLinearZAndDeriv[ipos] = float4(linearZ, max(abs(ddx(linearZ)), abs(ddy(linearZ))), 0, 0);

    return gout;
}
