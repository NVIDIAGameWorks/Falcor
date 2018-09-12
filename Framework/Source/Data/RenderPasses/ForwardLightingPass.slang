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
__import Shading;
__import Helpers;
__import BRDF;

cbuffer PerFrameCB
{
    float2 gRenderTargetDim;
};

SamplerState gSampler;
Texture2D visibilityBuffer;
VertexOut vsData;

VertexOut vs(VertexIn vIn)
{
    VertexOut vsOut;
    vsOut = defaultVS(vIn);
#ifdef _OUTPUT_MOTION_VECTORS
    vsOut.prevPosH.xy += vsOut.prevPosH.w * 2 * float2(gCamera.jitterX, gCamera.jitterY);
#endif
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

PsOut ps(VertexOut vOut, float4 pixelCrd : SV_POSITION)
{
    PsOut psOut;

    ShadingData sd = prepareShadingData(vOut, gMaterial, gCamera.posW);

    float4 finalColor = float4(0, 0, 0, 1);

    for (uint l = 0; l < gLightsCount; l++)
    {
        float shadowFactor = 1;
       if (l == 0)
        {
            shadowFactor = visibilityBuffer.Load(int3(vOut.posH.xy, 0)).r;
            shadowFactor *= sd.opacity;
        }
        finalColor.rgb += evalMaterial(sd, gLights[l], shadowFactor).color.rgb;
    }

    // Add the emissive component
    finalColor.rgb += sd.emissive;
    finalColor.a = sd.opacity;
    finalColor.rgb += evalMaterial(sd, gLightProbe).color.rgb;


    // Add light-map
    finalColor.rgb += sd.diffuse * sd.lightMap.rgb;

    psOut.color = finalColor;
    psOut.normal = float4(vOut.normalW * 0.5f + 0.5f, 1.0f);

#ifdef _OUTPUT_MOTION_VECTORS
    psOut.motion = calcMotionVector(pixelCrd.xy, vOut.prevPosH, gRenderTargetDim);
#endif

#if defined(_VISUALIZE_CASCADES) && defined(_ENABLE_SHADOWS)
    float3 cascadeColor = visibilityBuffer.Load(int3(vOut.posH.xy, 0)).gba;
    psOut.color.rgb *= cascadeColor;
#endif
    return psOut;
}
