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
__import Shading;

cbuffer PerImageCB
{
    // G-Buffer
    // Lighting params
    LightData gDirLight;
    LightData gPointLight;
    float3 gAmbient;
    // Debug mode
    uint gDebugMode;
};

// Debug modes
#define ShowPos         1
#define ShowNormals     2
#define ShowAlbedo      3
#define ShowLighting    4

float3 shade(float3 posW, float3 normalW, float linearRoughness, float4 albedo)
{
    // Discard empty pixels
    if (albedo.a <= 0)
    {
        discard;
    }

    /* Reconstruct the hit-point */
    ShadingData sd = initShadingData();
    sd.posW = posW;
    sd.V = normalize(gCamera.posW - posW);
    sd.N = normalW;
    sd.NdotV = abs(dot(sd.V, sd.N));
    sd.linearRoughness = linearRoughness;

    /* Reconstruct layers (one diffuse layer) */
    sd.diffuse = albedo.rgb;
    sd.opacity = 0;

    /* Do lighting */
    ShadingResult dirResult = evalMaterial(sd, gDirLight, 1);
    ShadingResult pointResult = evalMaterial(sd, gPointLight, 1);

    float3 result;
    // Debug vis
    if (gDebugMode == ShowPos)
        result = posW;
    else if (gDebugMode == ShowNormals)
        result = 0.5 * normalW + 0.5f;
    else if (gDebugMode == ShowAlbedo)
        result = albedo.rgb;
    else if (gDebugMode == ShowLighting)
        result = (dirResult.diffuseBrdf + pointResult.diffuseBrdf) / sd.diffuse.rgb;
    else
        result = dirResult.diffuse + pointResult.diffuse;

    return result;
}

Texture2D gGBuf0;
Texture2D gGBuf1;
Texture2D gGBuf2;

float4 main(float2 texC : TEXCOORD, float4 pos : SV_POSITION) : SV_TARGET
{
    // Fetch a G-Buffer
    float3 posW    = gGBuf0.Load(int3(pos.xy, 0)).rgb;
    float4 buf1Val = gGBuf1.Load(int3(pos.xy, 0));
    float3 normalW = buf1Val.rgb;
    float linearRoughness = buf1Val.a;
    float4 albedo  = gGBuf2.Load(int3(pos.xy, 0));

    float3 color = shade(posW, normalW, linearRoughness, albedo);

    return float4(color, 1);
}
