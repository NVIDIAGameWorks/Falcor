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
__import DefaultVS;
__import Shading;
__import ShaderCommon;
__import Effects.CascadedShadowMap;

cbuffer PerFrameCB : register(b0)
{
	float3 gAmbient;
    CsmData gCsmData[_LIGHT_COUNT];
    bool visualizeCascades;
    float4x4 camVpAtLastCsmUpdate;
};

struct ShadowsVSOut
{
    VS_OUT vsData;
    float shadowsDepthC : DEPTH;
};

float4 main(ShadowsVSOut pIn) : SV_TARGET0
{
    ShadingAttribs shAttr;
    prepareShadingAttribs(gMaterial, pIn.vsData.posW, gCam.position, pIn.vsData.normalW, pIn.vsData.bitangentW, pIn.vsData.texC, 0, shAttr);
    ShadingOutput result;
    float4 fragColor = float4(0,0,0,1);
    
    [unroll]
    for(uint l = 0 ; l < _LIGHT_COUNT ; l++)
    {
        float shadowFactor = calcShadowFactor(gCsmData[l], pIn.shadowsDepthC, shAttr.P, pIn.vsData.posH.xy/pIn.vsData.posH.w);
        evalMaterial(shAttr, gLights[l], result, l == 0);
        fragColor.rgb += result.diffuseAlbedo * result.diffuseIllumination * shadowFactor;
        fragColor.rgb += result.specularAlbedo * result.specularIllumination * (0.01f + shadowFactor * 0.99f);
    }

    fragColor.rgb += gAmbient * result.diffuseAlbedo * 0.1;
    if(visualizeCascades)
    {
        //Ideally this would be light index so you can visualize the cascades of the 
        //currently selected light. However, because csmData contains Textures, it doesn't
        //like getting them with a non literal index.
        fragColor.rgb *= getBlendedCascadeColor(gCsmData[_LIGHT_INDEX], pIn.shadowsDepthC);
    }

    return fragColor;
}
