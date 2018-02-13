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

float radicalInverse(uint i)
{
    i = (i & 0x55555555) << 1 | (i & 0xAAAAAAAA) >> 1;
    i = (i & 0x33333333) << 2 | (i & 0xCCCCCCCC) >> 2;
    i = (i & 0x0F0F0F0F) << 4 | (i & 0xF0F0F0F0) >> 4;
    i = (i & 0x00FF00FF) << 8 | (i & 0xFF00FF00) >> 8;
    i = (i << 16) | (i >> 16);
    return float(i) * 2.3283064365386963e-10f;
}

float2 getHammersley(uint i, uint N)
{
    return float2(float(i) / float(N), radicalInverse(i));
}

void importanceSampleCosDir(float2 u, float3 N,
    out float3 L, out float NdotL, out float pdf)
{
    // Local referencial
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

float4 integrate(float3 N)
{
    float3 accBrdf = 0;
    for (uint i = 0; i < _SAMPLE_COUNT; ++i)
    {
        float2 eta = getHammersley(i, _SAMPLE_COUNT);
        float3 L;
        float NdotL;
        float pdf;

        // see reference code in appendix
        importanceSampleCosDir(eta, N, L, NdotL, pdf);
        if (NdotL > 0)
        {
            float2 uv = dirToSphericalCrd(L);
            accBrdf += gInputTex.Sample(gSampler, uv).rgb;
        }
    }

    return float4(accBrdf / float(_SAMPLE_COUNT), 1.0f);
}

float4 main(float2 texC : TEXCOORD, float4 posS : SV_POSITION) : SV_TARGET0
{
    float3 dir = sphericalCrdToDir(texC);
    return integrate(dir);
}
