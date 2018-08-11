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
	float gaussianWidth; // 
	float level;         // 31.5f
	float scattering;    // 800
	float maxDepth;      // 0.001
};

Buffer<float> gaussianWeights;

float3 subsurface(float2 texC)
{
	float3 diffuseColor = gSrcTex.SampleLevel(gSampler, texC, 0).xyz;
	float depth = gSrcDepthTex.SampleLevel(gSampler, texC, 0).x;
	float widthXLevel = gaussianWidth * level;
	
	// early breakout if width is too small to blur in neighboring pixels
	
	// divide by depth for closer objects to have higher blur
	float2 offsetStep = (pixelSize * dir * widthXLevel) / depth;
	float2 texOffset = -offsetStep;
	
	// calculate stretch factor based on the depths for the gaussianWeights
	float depthGradient = abs( min( dir.x > 0 ? ddx(depth) : ddy(depth), maxDepth ) );
	float sfDenom = depth + scattering * depthGradient;
	float stretchFactor = level / sfDenom;
	
	
	$for(i in Range(_KERNEL_WIDTH))
    {
		float3 sample = gSrcTex.SampleLevel(gSampler, texC + texOffset, 0).xyz;
		float sampleDepth = gSrcDepthTex.SampleLevel(gSampler, texC + texOffset, 0).x;
		
		// test change of depth from first pixel. go back to original color if too big
		float lerpVal = min(1.0f, 0.0125f * scattering * abs(depth - sampleDepth));
		sample = lerp(sample, diffuseColor, lerpVal);
		
		diffuseColor += float3(gaussianWeights[i * 4], gaussianWeights[i * 4 + 1], gaussianWeights[i * 4 + 2]) * sample;
		texOffset += gaussianWeights[i * 4 + 3] * offsetStep / _KERNEL_WIDTH;
	}
	
	return diffuseColor;
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    return float4(subsurface(texC), 1.0f);
}