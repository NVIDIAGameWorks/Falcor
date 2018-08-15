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

Texture2D gSrcTex;
Texture2D gSrcDepthTex;

// depth map ???

SamplerState gSampler;

cbuffer SubsurfaceParams
{
	float2 dir;
	float2 pixelSize;
	float scatteringWidth; // 
};

Buffer<float> gaussianWeights;

#define MAX_DEPTH_GRADIENT 0.001f

float4 subsurface(float2 texC)
{
	float3 diffuseColor = gSrcTex.SampleLevel(gSampler, texC, 0).xyz;
	float depth = gSrcDepthTex.SampleLevel(gSampler, texC, 0).x;
	
	if (depth >= 0.99999f)
	{
	  return float4(diffuseColor, 1.0f);
	}
	
	// divide by depth for closer objects to have higher blur
	float2 offsetStep = (pixelSize * scatteringWidth * dir) / (3.0f );
	
	for(uint i = 1; i < _KERNEL_WIDTH; ++i)
    {
	    float2 texOffset = gaussianWeights[i * 4 + 3] * offsetStep;
		float3 sample = gSrcTex.SampleLevel(gSampler, texC + texOffset, 0).xyz;
		float sampleDepth = gSrcDepthTex.SampleLevel(gSampler, texC + texOffset, 0).x;
		
		// test change of depth from first pixel. go back to original color if too big
		float lerpVal = min(1.0f, 0.0125f * scatteringWidth * abs(depth - sampleDepth));
		sample = lerp(sample, diffuseColor, lerpVal);
		
		diffuseColor += float3(gaussianWeights[i * 4], gaussianWeights[i * 4 + 1], gaussianWeights[i * 4 + 2]) * sample;
		
	}
	
	return float4(diffuseColor, 1.0f);
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    return subsurface(texC);
}