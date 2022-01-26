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
#define COMPILER_DXC
#include "nrd/Shaders/NRD.hlsli"

import Utils.Color.ColorHelpers;

cbuffer PerImageCB
{
    float gMaxIntensity;

    Texture2D<float4>   gDiffuseRadianceHitDist;
    Texture2D<float4>   gSpecularRadianceHitDist;

    RWTexture2D<float4> gOutDiffuseRadianceHitDist;
    RWTexture2D<float4> gOutSpecularRadianceHitDist;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    static const float kEpsilon = 1e-6f;

    int2 ipos = dispatchThreadId.xy;

    float4 diffuseRadianceHitDist = gDiffuseRadianceHitDist[ipos];
    float4 specularRadianceHitDist = gSpecularRadianceHitDist[ipos];

    // Clamp radiance.
    float lDiff = luminance(diffuseRadianceHitDist.rgb);
    if (lDiff > kEpsilon)
    {
        diffuseRadianceHitDist.rgb *= min(gMaxIntensity / lDiff, 1.f);
    }

    float lSpec = luminance(specularRadianceHitDist.rgb);
    if (lSpec > kEpsilon)
    {
        specularRadianceHitDist.rgb *= min(gMaxIntensity / lSpec, 1.f);
    }

    diffuseRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);
    specularRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);

    gOutDiffuseRadianceHitDist[ipos] = diffuseRadianceHitDist;
    gOutSpecularRadianceHitDist[ipos] = specularRadianceHitDist;
}
