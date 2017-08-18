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
#include "RenderForLearningCommon.hlsli"

__import BSDFs;
__import Shading;
__import Helpers;

layout(set = 1, binding = 0) Texture2D gEnvMap;
layout(set = 1, binding = 1) SamplerState gSampler;

struct PsOut
{
    float4 color : SV_TARGET0;
    float4 normal : SV_TARGET1;
#ifdef _OUTPUT_MOTION_VECTORS
    float2 motion : SV_TARGET2;
#endif
};

PsOut main(MainVsOut vOut, float4 pixelCrd : SV_POSITION)
{
    PsOut psOut;

    ShadingAttribs shAttr;
    prepareShadingAttribs(gMaterial, vOut.vsData.posW, gCam.position, vOut.vsData.normalW, vOut.vsData.bitangentW, vOut.vsData.texC, shAttr);

    ShadingOutput result;
    result.finalValue = 0;
    float4 finalColor = 0;
    float envMapFactor = 1;
    float opacity = 1;
#ifdef _ENABLE_TRANSPARENCY
    opacity = getDiffuseColor(shAttr).a;
#endif

    [unroll]
    for (uint l = 0; l < _LIGHT_COUNT; l++)
    {
        float shadowFactor = 1;
#ifdef _ENABLE_SHADOWS
        if (l == 0)
        {
            shadowFactor = calcShadowFactor(gCsmData, vOut.shadowsDepthC, shAttr.P, vOut.vsData.posH.xy / vOut.vsData.posH.w);
            shadowFactor *= opacity;
            envMapFactor -= 1 - shadowFactor;
        }
#endif
        evalMaterial(shAttr, gLights[l], shadowFactor, result, l == 0);
    }

    finalColor = float4(result.finalValue, 1.f);
#ifdef _ENABLE_TRANSPARENCY
    finalColor.a = opacity * gOpacityScale;
#endif

#ifdef _ENABLE_REFLECTIONS
    // Calculate the view vector
    float3 view = reflect(-shAttr.E, shAttr.N);
    float2 texC = dirToSphericalCrd(view);
    float rough = shAttr.preparedMat.values.layers[1].albedo.a;
    uint h,w, mipCount;
    gEnvMap.GetDimensions(0, h, w, mipCount);
    float lod = rough * mipCount;
    float3 envMapVal = gEnvMap.SampleLevel(gSampler, texC, lod).rgb;

    envMapFactor = envMapFactor * evalGGXDistribution(float3(0, 0, 1), rough.rr);
    envMapFactor = saturate(envMapFactor) * gEnvMapFactorScale;
    finalColor.rgb += (result.specularAlbedo) * envMapVal * envMapFactor;
#endif

    // add ambient
    finalColor.rgb += gAmbientLighting * getDiffuseColor(shAttr).rgb;

    psOut.color = finalColor;
    psOut.normal = float4(vOut.vsData.normalW * 0.5f + 0.5f, 1.0f);

#ifdef _OUTPUT_MOTION_VECTORS
    psOut.motion = calcMotionVector(pixelCrd.xy, vOut.vsData.prevPosH, gRenderTargetDim);
#endif

#if defined(_VISUALIZE_CASCADES) && defined(_ENABLE_SHADOWS)
    psOut.color.rgb *= getBlendedCascadeColor(gCsmData, vOut.shadowsDepthC);
#endif
    return psOut;
}
