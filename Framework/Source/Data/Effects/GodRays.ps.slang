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

__import ShaderCommon;
__import Lights;
__import Helpers;

SamplerState gSampler;

Texture2D gColor;
Texture2D gDepth;

struct Media
{
    float density;
    float decay;
    float weight; // intensity at each sample
};

cbuffer GodRaySettings
{
   Media gMedia;
   float exposer;
   uint lightIndex;
};

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
	float3 finalColor = float3(0, 0, 0);
	float2 toLightSource;
	float2 texPosition = texC;
	
	float4 screenSpaceLightPosition = 
		mul(float4(gLights[int(lightIndex)].posW, 1.0f), gCamera.viewProjMat);
	toLightSource = texPosition - screenSpaceLightPosition.xy;
	
	toLightSource *= 1.0f / (float(_NUM_SAMPLES) * gMedia.density);
	toLightSource.y *= -1.0f;
	
	float accumulatedDecay = 1.0f;
	float mask = (abs(gDepth.SampleLevel(gSampler, texC, 0).x) < 1.0f ? 0.0f : 1.0f);
	float3 currentColor = gColor.Sample( gSampler, texPosition ).xyz * mask;
	float3 lightIntensity = gLights[int(lightIndex)].intensity;

	for(uint i = 0; i < _NUM_SAMPLES; i++)
	{
		texPosition -= toLightSource;
		
		float3 nextColor = gColor.Sample( gSampler, texPosition ).xyz * (abs(gDepth.SampleLevel(gSampler, texPosition, 0).x) < 1.0f ? 0.0f : 1.0f);
		nextColor *= accumulatedDecay * gMedia.weight;
	
		currentColor += nextColor * lightIntensity;
	
		accumulatedDecay *= gMedia.decay;
	}
	
	finalColor += currentColor;
    return float4(finalColor * exposer, 1.0f);
}
