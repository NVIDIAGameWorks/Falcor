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
__import ShaderCommon;
__import DefaultVS;
__import Effects.CascadedShadowMap;
__import Shading;
__import Helpers;
__import BRDF;

layout(binding = 0) cbuffer PerFrameCB : register(b0)
{
    CsmData gCsmData;
    float4x4 camVpAtLastCsmUpdate;
    float2 gRenderTargetDim;
    float gOpacityScale;
};

layout(set = 1, binding = 1) SamplerState gSampler;
Texture2D gVisibilityBuffer;

struct MainVsOut
{
    VertexOut vsData;
    float shadowsDepthC : DEPTH;
};

MainVsOut vs(VertexIn vIn)
{
    MainVsOut vsOut;
    vsOut.vsData = defaultVS(vIn);

#ifdef _OUTPUT_MOTION_VECTORS
    vsOut.vsData.prevPosH.xy += vsOut.vsData.prevPosH.w * 2 * float2(gCamera.jitterX, gCamera.jitterY);
#endif

    vsOut.shadowsDepthC = mul(float4(vsOut.vsData.posW, 1), camVpAtLastCsmUpdate).z;
    return vsOut;
}

struct PsOut
{
    float4 color : SV_TARGET0;
    float4 normal : SV_TARGET1;
#ifdef _OUTPUT_MOTION_VECTORS
    float2 motion : SV_TARGET2;
#endif
};

PsOut ps(MainVsOut vOut, float4 pixelCrd : SV_POSITION)
{
    PsOut psOut;

    ShadingData sd = prepareShadingData<TMaterial>(vOut.vsData, gMaterial, gCamera.posW);

    float4 finalColor = float4(0, 0, 0, 1);

    finalColor.rgb += evalMaterial<TMaterial.BxDF, TLightCollection>(sd, gLightEnv).color.rgb;
    // Add light-map and/or the emissive component
    finalColor.rgb += sd.bxdf.getIntrinsicLighting();

    psOut.color = finalColor;
    psOut.normal = float4(vOut.vsData.normalW * 0.5f + 0.5f, 1.0f);

#ifdef _OUTPUT_MOTION_VECTORS
    psOut.motion = calcMotionVector(pixelCrd.xy, vOut.vsData.prevPosH, gRenderTargetDim);
#endif

#if defined(_VISUALIZE_CASCADES) && defined(_ENABLE_SHADOWS)
    float3 cascadeColor = gVisibilityBuffer.Load(int3(vOut.vsData.posH.xy, 0)).gba;
    psOut.color.rgb *= cascadeColor;
#endif
    return psOut;
}
