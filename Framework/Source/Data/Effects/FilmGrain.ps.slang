/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
__import Helpers;

SamplerState gSampler;

Texture2D srcTex;
Texture2D noiseTex;

cbuffer filmGrain
{
    float intensity;
    float grainSize;
    float3 grainTint;
    float2 randOffset;
};

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    float4 fragColor = srcTex.SampleLevel(gSampler, texC, 0); 
    float noiseVal = intensity * noiseTex.Sample(gSampler, (texC + randOffset) * grainSize).r;

    // apply grain only to specified luminance range
    float luma = luminance(fragColor.xyz);
    float3 grainColor = noiseVal * (grainTint * 2.0f);
#ifdef _USE_LUMINANCE_RANGE
    grainColor *= (luma < _LUMINANCE_RANGE_MIN || luma > _LUMINANCE_RANGE_MAX) ? 0.0f : 1.0f;
#endif
    fragColor.xyz += grainColor * luma;

    return fragColor;
}