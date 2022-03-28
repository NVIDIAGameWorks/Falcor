/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#define NRD_COMPILER_DXC
#include "nrd/Shaders/Include/NRD.hlsli"

import Utils.Color.ColorHelpers;

#ifndef NRD_METHOD
// NRD_METHOD must be defined in order to use this module.
#error "NRD_METHOD not defined!"
#endif

#define NRD_METHOD_RELAX_DIFFUSE_SPECULAR 0
#define NRD_METHOD_REBLUR_DIFFUSE_SPECULAR 1

cbuffer PerImageCB
{
    float4  gHitDistParams;
    float   gMaxIntensity;

    Texture2D<float>    gViewZ;
    Texture2D<float4>   gNormalRoughness;

    RWTexture2D<float4> gDiffuseRadianceHitDist;
    RWTexture2D<float4> gSpecularRadianceHitDist;
};

void clampRadiance(inout float3 diffuseRadiance, inout float3 specularRadiance)
{
    static const float kEpsilon = 1e-6f;

    float lDiff = luminance(diffuseRadiance);
    if (lDiff > kEpsilon)
    {
        diffuseRadiance *= min(gMaxIntensity / lDiff, 1.f);
    }

    float lSpec = luminance(specularRadiance);
    if (lSpec > kEpsilon)
    {
        specularRadiance *= min(gMaxIntensity / lSpec, 1.f);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    int2 ipos = dispatchThreadId.xy;

    float4 diffuseRadianceHitDist = gDiffuseRadianceHitDist[ipos];
    float4 specularRadianceHitDist = gSpecularRadianceHitDist[ipos];

    clampRadiance(diffuseRadianceHitDist.rgb, specularRadianceHitDist.rgb);

#if NRD_METHOD == NRD_METHOD_RELAX_DIFFUSE_SPECULAR

    diffuseRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);
    specularRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);

#else // NRD_METHOD == NRD_METHOD_REBLUR_DIFFUSE_SPECULAR

    float viewZ = gViewZ[ipos];
    float linearRoughness = gNormalRoughness[ipos].z;

    diffuseRadianceHitDist.a = REBLUR_FrontEnd_GetNormHitDist(diffuseRadianceHitDist.a, viewZ, gHitDistParams, linearRoughness);
    REBLUR_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);

    specularRadianceHitDist.a = REBLUR_FrontEnd_GetNormHitDist(specularRadianceHitDist.a, viewZ, gHitDistParams, linearRoughness);
    REBLUR_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);

#endif

    gDiffuseRadianceHitDist[ipos] = diffuseRadianceHitDist;
    gSpecularRadianceHitDist[ipos] = specularRadianceHitDist;
}
