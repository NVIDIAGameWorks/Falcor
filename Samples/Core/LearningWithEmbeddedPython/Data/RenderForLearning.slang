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
__import Shading;
__import Effects.CascadedShadowMap;
__import DefaultVS;

layout(binding = 0) cbuffer PerFrameCB : register(b0)
{
	float3 gAmbient;
    float gEnvMapFactorScale;
    CsmData gCsmData;
    float4x4 camVpAtLastCsmUpdate;
    float2 gRenderTargetDim;
    float gOpacityScale;
};

struct MainVsOut
{
    VertexOut vsData;
    float shadowsDepthC : DEPTH;
};

Texture2D gEnvMap;
SamplerState gSampler;

struct PsOut
{
    float4 color : SV_TARGET0;
    float4 normal : SV_TARGET1;
#ifdef _OUTPUT_MOTION_VECTORS
    float2 motion : SV_TARGET2;
#endif
};

MainVsOut vs(VertexIn vIn)
{
    MainVsOut vsOut;
    vsOut.vsData = defaultVS(vIn);

#ifdef _OUTPUT_MOTION_VECTORS
    vsOut.vsData.prevPosH.xy += vsOut.vsData.prevPosH.w * 2 * float2(gCam.jitterX, gCam.jitterY);
#endif

    vsOut.shadowsDepthC = mul(float4(vsOut.vsData.posW, 1), camVpAtLastCsmUpdate).z;
    return vsOut;
}

PsOut ps(MainVsOut vOut, float4 pixelCrd : SV_POSITION)
{
    PsOut psOut;

    ShadingData shData = prepareShadingData(vOut.vsData, gMaterial, gCamera.posW);

    ShadingResult result;
    float4 finalColor = float4(0, 0, 0, 1);
    float envMapFactor = 1;
    float opacity = 1;

    [unroll]
    for (uint l = 0; l < _LIGHT_COUNT; l++)
    {
        float shadowFactor = 1;
#ifdef _ENABLE_SHADOWS
        if (l == 0)
        {
            shadowFactor = calcShadowFactor(gCsmData, vOut.shadowsDepthC, shData.posW, vOut.vsData.posH.xy / vOut.vsData.posH.w);
            shadowFactor *= shData.opacity;
            envMapFactor -= 1 - shadowFactor;
        }
#endif
        finalColor.rgb += evalMaterial(shData, gLights[l], shadowFactor).color.rgb;
    }

#ifdef _ENABLE_TRANSPARENCY
    finalColor.a = shData.opacity * gOpacityScale;
#endif

#ifdef _ENABLE_REFLECTIONS
    finalColor.rgb += evalMaterial(shData, gLightProbe).color.rgb;
#endif

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
