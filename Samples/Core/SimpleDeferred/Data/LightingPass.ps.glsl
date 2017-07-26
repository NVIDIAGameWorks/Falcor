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
#version 450

__import ShaderCommon;
__import Shading;

layout(set = 0, binding = 0) uniform PerImageCB
{
    // G-Buffer
    // Lighting params
	LightData gDirLight;
	LightData gPointLight;
	vec3 gAmbient;
    // Debug mode
	uint gDebugMode;
};

#include "LightingPassCommon.h"

layout(set = 1, binding = 0) uniform texture2D gGBuf0;
layout(set = 1, binding = 1) uniform texture2D gGBuf1;
layout(set = 1, binding = 2) uniform texture2D gGBuf2;
layout(set = 1, binding = 3) uniform sampler gSampler;

layout(location = 0) in  vec2 texC;
layout(location = 0) out vec4 fragColor;
 
void main()
{
    ivec2 crd = ivec2(gl_FragCoord.xy);
    // Fetch a G-Buffer
    vec3 posW    = texelFetch(sampler2D(gGBuf0, gSampler), crd, 0).rgb;
    vec3 normalW = texelFetch(sampler2D(gGBuf1, gSampler), crd, 0).rgb;
    vec4 albedo  = texelFetch(sampler2D(gGBuf2, gSampler), crd, 0);

    fragColor.rgb = shade(posW, normalW, albedo);
    fragColor.a = 1;
}
