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

cbuffer VsPerFrame : register(b0)
{
    float4x4 camVpAtLastCsmUpdate;
};

cbuffer PerFrameCB : register(b1)
{
    CsmData gCsmData;
    bool visualizeCascades;
};

struct ShadowsVSOut
{
    float4 posH : SV_POSITION;
    float3 posW : POSITION;
    float shadowsDepthC : DEPTH;
};

ShadowsVSOut vsMain(VertexIn vIn)
{
    ShadowsVSOut vOut;
    float4 posW = mul(vIn.pos, getWorldMat(vIn));
    vOut.posW = posW.xyz;
    vOut.posH = mul(posW, gCamera.viewProjMat);
    vOut.shadowsDepthC = mul(float4(vOut.posW, 1), camVpAtLastCsmUpdate).z;
    return vOut;
}

float4 psMain(ShadowsVSOut pIn) : SV_TARGET0
{
    float4 color = float4(0,0,0,1);
    color.r = calcShadowFactor(gCsmData, pIn.shadowsDepthC, pIn.posW, pIn.posH.xy / pIn.posH.w);

    if(visualizeCascades)
    {
        color.gba = getBlendedCascadeColor(gCsmData, pIn.shadowsDepthC);
    }

    return color;
}

