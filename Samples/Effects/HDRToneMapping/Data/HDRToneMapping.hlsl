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
#define PI 3.141591

Texture2D gEnvMap;
SamplerState gSampler;

cbuffer PerFrameCB : register(b0)
{
    float4x4 gWvpMat;
    float4x4 gWorldMat;
    float3 gEyePosW;
    float gLightIntensity;
    float gSurfaceRoughness;
};

struct VsIn
{
    float4 pos : POSITION;
    float3 normal : NORMAL;
};

struct VsOut
{
    float4 pos : SV_POSITION;
    float3 posW : POSITION;
    float3 normalW : NORMAL;
};

VsOut vs(VsIn vIn)
{
    VsOut vOut;
    vOut.pos = (mul(vIn.pos, gWvpMat));
    vOut.posW = (mul(vIn.pos, gWorldMat)).xyz;
    vOut.normalW = (mul(float4(vIn.normal, 0), gWorldMat)).xyz;
    return vOut;
}

float4 ps(VsOut vOut) : SV_TARGET
{
    float3 p = normalize(vOut.normalW);
    float2 uv;
    uv.x = (1 + atan2(-p.z, p.x) / PI) * 0.5;
    uv.y = 1 - (-acos(p.y) / PI);
    float4 color = gEnvMap.Sample(gSampler, uv);
    color.rgb *= gLightIntensity;

    // compute halfway vector
    float3 eyeDir = normalize(gEyePosW - vOut.posW);
    float3 h = normalize(eyeDir + vOut.normalW);
    float edoth = dot(eyeDir, h);
    float intensity = pow(clamp(edoth, 0, 1), gSurfaceRoughness);

    color.rgb *= intensity;
    return color;
}
