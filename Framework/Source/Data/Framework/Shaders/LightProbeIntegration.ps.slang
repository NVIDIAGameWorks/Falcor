/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
__import Helpers;
#include "HostDeviceSharedMacros.h"

SamplerState gSampler : register(s0);
texture2D gInputTex : register(t1);

void importanceSampleCosDir(float2 u, float3 N, out float3 L, out float NdotL, out float pdf)
{
    float3 up = abs(N.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 right = normalize(cross(up, N));
    float3 forward = cross(N, right);

    float u1 = u.x;
    float u2 = u.y;

    float r = sqrt(u1);
    float phi = u2 * M_PI * 2.0f;

    L = float3(r * cos(phi),
               r * sin(phi), 
               sqrt(max(0.0f, 1.0f - u1)));

    L = normalize(right * L.y + forward * L.x + N * L.z);

    NdotL = dot(L, N);
    pdf = NdotL * M_INV_PI;
}

float4 integration(float3 N)
{
    float3 accumulation = 0;
    for (uint i = 0; i < _SAMPLE_COUNT; i++)
    {
        float2 eta = getHammersley(i, _SAMPLE_COUNT);
        float3 L;
        float NdotL;
        float pdf;

        importanceSampleCosDir(eta, N, L, NdotL, pdf);
        if (NdotL > 0)
        {
            float2 uv = dirToSphericalCrd(L);
            accumulation += gInputTex.Sample(gSampler, uv).rgb;
        }
    }

    return float4(accumulation / float(_SAMPLE_COUNT), 1.0f);
}

float4 main(float2 texC : TEXCOORD, float4 posS : SV_POSITION) : SV_TARGET0
{
    float3 dir = sphericalCrdToDir(texC);
    return integration(dir);
}
