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

cbuffer PerFrameCB
{
    float4x4 gWorldMat;
    float4x4 gWvpMat;
};

struct LightCB
{
    float3 vec3Val; // We're using 2 values. [0]: worldDir [1]: intensity
};

RWStructuredBuffer<LightCB> gRWBuffer; // Only UAV counter used
StructuredBuffer<LightCB> gLight[4];
RWByteAddressBuffer gInvocationBuffer;
Buffer<float3> gSurfaceColor[2];

struct VsOut
{
    float4 position : SV_POSITION;
    float3 normalW : NORMAL;
};

float4 ps(VsOut vsOut) : SV_TARGET
{
    float3 n = normalize(vsOut.normalW);
    float nDotL = dot(n, -gLight[3][0].vec3Val);
    nDotL = clamp(nDotL, 0, 1);
    float4 color = float4(nDotL * gLight[3][1].vec3Val * gSurfaceColor[1][0], 1);

    gInvocationBuffer.InterlockedAdd(0, 1);
    gRWBuffer.IncrementCounter();

    return color;
}

VsOut vs(in float4 posL : POSITION, in float3 normalL : NORMAL)
{
    VsOut vsOut;

    vsOut.position = mul(posL, gWvpMat);
    vsOut.normalW = (mul(float4(normalL, 0), gWorldMat)).xyz;

    return vsOut;
}
