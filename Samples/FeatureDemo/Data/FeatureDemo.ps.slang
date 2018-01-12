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
#include "FeatureDemoCommon.hlsli"

__import Shading;
__import Helpers;
__import BRDF;

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

    HitPoint hitPt = prepareHitPoint(vOut.vsData, gMaterial, gCam.position);

    float4 finalColor = float4(0, 0, 0, 1);
    
    [unroll]
    for (uint l = 0; l < _LIGHT_COUNT; l++)
    {
        float shadowFactor = 1;
#ifdef _ENABLE_SHADOWS
        if (l == 0)
        {
            shadowFactor = calcShadowFactor(gCsmData, vOut.shadowsDepthC, hitPt.posW, vOut.vsData.posH.xy / vOut.vsData.posH.w);
            shadowFactor *= hitPt.opacity;
        }
#endif
        finalColor.rgb += evalMaterial(hitPt, gLights[l], shadowFactor).color.rgb;
    }

    // Add the emissive component
    finalColor.rgb += hitPt.emissive;

#ifdef _ENABLE_TRANSPARENCY
    finalColor.a = hitPt.opacity * gOpacityScale;
#endif

#ifdef _ENABLE_REFLECTIONS
    finalColor.rgb += evalMaterial(hitPt, gLightProbe).color.rgb;
#endif

    // add ambient
    finalColor.rgb += gAmbientLighting * hitPt.diffuse;

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
