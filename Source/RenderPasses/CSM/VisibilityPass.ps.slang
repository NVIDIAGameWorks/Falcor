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
import RenderPasses.CSM.CascadedShadowMap;

struct VisibilityPassData
{
    bool visualizeCascades;
    float4x4 invViewProj;
    uint2 screenDimension;
    uint mapBitsPerChannel;
};

cbuffer PerFrameCB : register(b0)
{
    CsmData gCsmData;
    VisibilityPassData gPass;
};

Texture2D gDepth;

float3 loadPosition(float2 UV, float depth)
{
    float2 invSize = 1.0f / gPass.screenDimension.xy;
    // Recompute position by unprojecting the depth stored in the z-buffer
    float2 ndc = 2 * UV + invSize - 1;
#ifndef FALCOR_FLIP_Y
    // NDC Y is bottom-to-top
    ndc.y = -ndc.y;
#endif
    float4 wsPos = mul(gPass.invViewProj, float4(ndc.x, ndc.y, depth, 1.f));
    return wsPos.xyz / wsPos.w;
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    //[0, 1] -> [0, 2] -> [-1, 1]
    float2 ndcXy = (texC * 2) - 1;

    float depth = gDepth[texC * gPass.screenDimension].x;
    float3 posW = loadPosition(texC, depth);

    float4 color = float4(0,0,0,0);
    color.r = calcShadowFactor(gCsmData, depth, posW, ndcXy);

    if(gPass.visualizeCascades)
    {
        color.gba = getBlendedCascadeColor(gCsmData, depth);
    }

    return color;
}

