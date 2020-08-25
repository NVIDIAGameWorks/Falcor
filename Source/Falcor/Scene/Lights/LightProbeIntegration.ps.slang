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
#include "Utils/Math/MathConstants.slangh"
import Utils.Helpers;
import Scene.Material.BRDF;

SamplerState gSampler : register(s0);
Texture2D gInputTex : register(t1);

cbuffer DataCB : register(b0)
{
    float gRoughness;
    uint32_t gSampleCount;
}

void generateBasis(float3 N, out float3 up, out float3 right, out float3 forward)
{
    up = abs(N.z) < 0.999999f ? float3(0, 0, 1) : float3(1, 0, 0);
    right = normalize(cross(up, N));
    forward = cross(N, right);
}

float3 importanceSampleCosDir(float2 u, float3 N)
{
    float3 up, right, forward;
    generateBasis(N, up, right, forward);

    float u1 = u.x;
    float u2 = u.y;

    float r = sqrt(u1);
    float phi = u2 * M_2PI;

    float3 L = float3(r * cos(phi),
                      r * sin(phi),
                      sqrt(max(0.0f, 1.0f - u1)));

    return normalize(right * L.y + forward * L.x + N * L.z);
}

float3 importanceSampleGGX(float2 u, float3 N, float roughness)
{
    float a = roughness * roughness;

    float phi = M_2PI * u.x;
    float cosTheta = sqrt((1 - u.y) / (1 + (a * a - 1) * u.y));
    float sinTheta = sqrt(1 - cosTheta * cosTheta);

    // Tangent space H
    float3 tH;
    tH.x = sinTheta * cos(phi);
    tH.y = sinTheta * sin(phi);
    tH.z = cosTheta;

    float3 up, right, forward;
    generateBasis(N, up, right, forward);

    // World space H
    return normalize(right * tH.x + forward * tH.y + N * tH.z);
}

float smithGGX(float NdotL, float NdotV, float roughness)
{
    float k = ((roughness + 1) * (roughness + 1)) / 8;
    float g1 = NdotL / (NdotL * (1 - k) + k);
    float g2 = NdotV / (NdotV * (1 - k) + k);
    return g1 * g2;
}

float4 integrateDiffuseLD(float3 N)
{
    float3 accumulation = 0;
    float accBrdf = 0;
    float3 V = N;
    for (uint i = 0; i < gSampleCount; i++)
    {
        float2 u = getHammersley(i, gSampleCount);
        float3 L = importanceSampleCosDir(u, N);
        float NdotL = saturate(dot(N, L));
        if (NdotL > 0)
        {
            float2 uv = dirToSphericalCrd(L);
            float LdotH = saturate(dot(L, normalize(V + L)));
            float NdotV = saturate(dot(N, V));
            float brdf = disneyDiffuseFresnel(NdotV, NdotL, LdotH, 0);
            accumulation += gInputTex.Sample(gSampler, uv).rgb * brdf;
            accBrdf += brdf;
        }
    }

    return float4(accumulation / accBrdf, 1.0f);
}

float4 integrateSpecularLD(float3 V, float3 N, float roughness)
{
    // Resource Dimensions
    float width, height, mipCount;
    gInputTex.GetDimensions(0, width, height, mipCount);

    // As if the texture was wrapped around a cube like a cube map
    // The original algorithm uses cube map dimensions to calculate solid angle, but we don't use cube maps
    float cubeWidth = width / 4;

    float3 accBrdf = 0;
    float accBrdfWeight = 0;
    float NdotV = saturate(dot(N, V));
    for (uint i = 0; i < gSampleCount; i++)
    {
        float2 u = getHammersley(i, gSampleCount);
        float3 H = importanceSampleGGX(u, N, roughness);
        float3 L = reflect(-N, H);
        float NdotL = dot(N, L);

        if(NdotL > 0)
        {
            float NdotH = saturate(dot(N, H));
            float LdotH = saturate(dot(L, H));

            // D term GGX
            float pdf = (evalGGX(roughness, NdotH) * M_1_PI) * NdotH / (4 * LdotH);

            float omegaS = 1 / (gSampleCount * pdf);
            float omegaP = 4.0 * M_PI / (6 * cubeWidth * cubeWidth);
            float mipLevel = clamp(0.5 * log2(omegaS / omegaP), 0, mipCount);

            float2 uv = dirToSphericalCrd(L);
            float3 Li = gInputTex.SampleLevel(gSampler, uv, mipLevel).rgb;
            float weight = NdotL;
            weight *= evalSmithGGX(NdotL, NdotV, roughness);
            weight *= LdotH / NdotH; // this is likely 1
            weight *= saturate(1 - saturate(pow(1 - LdotH, 5))); // neglecting bias

            accBrdf += Li * weight;
            accBrdfWeight += weight;
        }
    }

    return float4(accBrdf / accBrdfWeight, 1.0f);
}

float4 integrateDFG(float3 N, float3 V, float roughness)
{
    float NdotV = dot(N, V);
    float3 accumulation = 0;

    for(uint i = 0; i < gSampleCount; i++)
    {
        float2 u = getHammersley(i, gSampleCount);

        // Specular GGX DFG integration (stored in RG)
        float3 H = importanceSampleGGX(u, N, roughness);
        float3 L = reflect(-N, H);
        float NdotH = saturate(dot(N, H));
        float LdotH = saturate(dot(L, H));
        float NdotL = saturate(dot(N, L));

        // #TODO Using our evalSmithGGX (modified to undo optimized terms) looks bad in certain cases.
        float G = smithGGX(NdotL, NdotV, roughness);
        if(NdotL > 0 && G > 0)
        {
            float GVis = (G * LdotH) / (NdotV * NdotH);
            float Fc = fresnelSchlick(0, 1, LdotH).r;
            accumulation.r += (1 - Fc) * GVis;
            accumulation.g += Fc * GVis;
        }

        // Disney Diffuse integration (stored in B)
        u = frac(u + 0.5);
        L = importanceSampleCosDir(u, N);
        NdotL = saturate(dot(N, L));
        if(NdotL > 0)
        {
            LdotH = saturate(dot(L, normalize(V + L)));
            NdotV = saturate(dot(N, V));
            accumulation.b += disneyDiffuseFresnel(NdotV, NdotL, LdotH, sqrt(roughness));
        }
    }

    return float4(accumulation / float(gSampleCount), 1.0f);
}

// DFG texture will be sampled using
// texC.x = NdotV
// texC.y = roughness

// Output:
// RG - Specular DFG
// B  - Disney Diffuse DFG

float4 main(float2 texC : TEXCOORD, float4 posS : SV_POSITION) : SV_TARGET0
{
#ifdef _INTEGRATE_DFG
    const float3 N = float3(0, 0, 1);

    // texC.x is NdotV, calculate a valid V assuming constant N
    float theta = acos(texC.x);
    const float3 V = float3(sin(theta), 0, cos(theta));

    return integrateDFG(N, V, texC.y);

#else// _INTEGRATE_DFG else

    float3 dir = sphericalCrdToDir(texC);

#ifdef _INTEGRATE_DIFFUSE_LD
    return integrateDiffuseLD(dir);
#endif

#ifdef _INTEGRATE_SPECULAR_LD
    return integrateSpecularLD(dir, dir, max(0.01, gRoughness));
#endif

#endif // _INTEGRATE_DFG else
}
