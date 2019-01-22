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
import Shading;
import DefaultVS;
__import DefaultVS;         // VertexOut declaration

#include "VertexAttrib.h"

__import ShaderCommon;
__import DefaultVS;

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct GBufVertexOut
{
    VertexOut base;
    INTERPOLATION_MODE float3 normalObj  : OBJECTNORMAL;
    uint instanceID : INSTANCEID;
}; 


// Note:  This is a modified version of DefaultVS.slang that passes a bit more data
//        down to the fragment shader to enable more complex g-buffer outputs

/** Entry point for G-buffer rasterization vertex shader.
*/
GBufVertexOut vs(VertexIn vIn)
{
    GBufVertexOut vOut;

    // Do the computations in the default vertex shader
    vOut.base = defaultVS(vIn);

    // Pass down the additional outputs needed by our fragment shader

    // We need our index ID
    vOut.instanceID = vIn.instanceID;

    // We need an object-space normal.
#ifdef HAS_NORMAL
    vOut.normalObj = vIn.normal;
#else
    vOut.normalObj = 0;
#endif

  return vOut;
}

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

/** Entry point for G-buffer rasterization pixel shader.
*/
GBufferOut ps(GBufVertexOut vsOut, uint primID : SV_PrimitiveID /*, float4 pos : SV_Position*/)
{
    // BUG: When compiling with dxcompiler to shader model 6.0, the validation layers complains that SV_Position
    // has overlapping semantic index at 0. This is because GBufVertexOut contains VertexOut from DefaultVS.slang,
    // which in turn has a member posH with the the SV_Position semantic. So, we use that here instead.
    // However, when compiling graphics shaders for SM 6.0, we get no output. It's probably not related to this, but could be.
    const float4 pos = vsOut.base.posH;

    if (alphaTest(vsOut.base, gMaterial)) discard;
    ShadingData sd = prepareShadingData(vsOut.base, gMaterial, gCamera.posW);

    GBufferOut gout;

    gout.posW           = float4(sd.posW, 1.f);
    gout.normW          = float4(sd.N, 0.f);
    gout.bitangentW     = float4(sd.B, 0.f);
    gout.texC           = float4(sd.uv, 0.f, 0.f);

    gout.diffuseOpacity = float4(sd.diffuse, sd.opacity);
    gout.specRough      = float4(sd.specular, sd.linearRoughness);
    gout.emissive       = float4(sd.emissive, 0.f);
    gout.matlExtra      = float4(sd.IoR, sd.doubleSidedMaterial ? 1.f : 0.f, 0.f, 0.f);

    return gout;
}
