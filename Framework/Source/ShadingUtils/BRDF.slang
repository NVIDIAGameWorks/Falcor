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
#ifndef _FALCOR_BRDF_SLANG_
#define _FALCOR_BRDF_SLANG_

#include "HostDeviceSharedMacros.h"
__import ShaderCommon;
__import Lights;

#define DiffuseBrdfLambert      0
#define DiffuseBrdfDisney       1
#define DiffuseBrdfFrostbite    2

#define DiffuseBrdf DiffuseBrdfFrostbite

float3 fresnelSchlick(float3 f0, float3 f90, float u)
{
    return f0 + (f90 - f0) * pow(1 - u, 5);
}

/** Lambertian diffuse
*/
float3 evalDiffuseLambert(HitPoint hitPt, LightSample ls)
{
    return ls.NdotL * ls.diffuse;
}

/** Disney's diffuse term. Based on https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
*/
float3 evalDiffuseDisney(HitPoint hitPt, LightSample ls)
{
    float fd90 = 0.5 + 2 * ls.LdotH * ls.LdotH * hitPt.linearRoughness;
    float fd0 = 1;
    float lightScatter = fresnelSchlick(fd0, fd90, ls.NdotL).r;
    float viewScatter = fresnelSchlick(fd0, fd90, hitPt.NdotV).r;
    return ls.diffuse * (viewScatter * lightScatter);
}

/** Frostbites's diffuse term. Based on https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
*/
float3 evalDiffuseFrostbite(HitPoint hitPt, LightSample ls)
{
    float energyBias = lerp(0, 0.5, hitPt.linearRoughness);
    float energyFactor = lerp(1, 1.0 / 1.51, hitPt.linearRoughness);
    float fd90 = energyBias + 2 * ls.LdotH * ls.LdotH * hitPt.linearRoughness;
    float fd0 = 1;
    float lightScatter = fresnelSchlick(fd0, fd90, ls.NdotL).r;
    float viewScatter = fresnelSchlick(fd0, fd90, hitPt.NdotV).r;
    return ls.diffuse * (viewScatter * lightScatter * energyFactor);
}

float3 evalDiffuse(HitPoint hitPt, LightSample ls)
{
#if DiffuseBrdf == DiffuseBrdfLambert
    return evalDiffuseLambert(hitPt, ls);
#elif DiffuseBrdf == DiffuseBrdfDisney
    return evalDiffuseDisney(hitPt, ls);
#elif DiffuseBrdf == DiffuseBrdfFrostbite
    return evalDiffuseFrostbite(hitPt, ls);
#endif
}

float evalGGX(float roughness, float NdotH)
{
    float a2 = roughness * roughness;
    float d = ((NdotH * a2 - NdotH) * NdotH + 1);
    return a2 / (d * d);
}

float evalSmithGGX(float NdotL, float NdotV, float roughness)
{
    // Optimized version of SMITH, already taking into account the division by (4 * NdotV)
    float a2 = roughness * roughness;
    // `NdotV *` and `NdotL *` are inversed. It's not a mistake.
    float ggxv = NdotV * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
    float ggxl = NdotL * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);
    return 0.5f / (ggxv + ggxl);

}

float3 evalSpecular(HitPoint hitPt, LightSample ls)
{
    if (ls.NdotL * hitPt.NdotV <= 0.f)
    {
        return 0;
    }
    float roughness = hitPt.roughness;

    float D = evalGGX(roughness, ls.NdotH);
    float G = evalSmithGGX(ls.NdotL, hitPt.NdotV, roughness);
    float3 F = fresnelSchlick(hitPt.specular, 1, max(0, ls.LdotH));
    return D * G * F * ls.specular / M_PI;
}
#endif	// _FALCOR_BRDF_SLANG_
