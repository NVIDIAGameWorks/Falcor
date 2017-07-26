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

#include "LightingPassCommon.h"

Texture2D gGBuf0;
Texture2D gGBuf1;
Texture2D gGBuf2;

float4 main(float2 texC : TEXCOORD, float4 pos : SV_POSITION) : SV_TARGET
{
    // Fetch a G-Buffer
    const float3 posW    = gGBuf0.Load(int3(pos.xy, 0)).rgb;
    const float3 normalW = gGBuf1.Load(int3(pos.xy, 0)).rgb;
    const float4 albedo  = gGBuf2.Load(int3(pos.xy, 0));

    float3 color = shade(posW, normalW, albedo);

	return float4(color, 1);
}
