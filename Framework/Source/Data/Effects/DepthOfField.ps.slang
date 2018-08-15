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

Texture2D gSrcTex[6];
Texture2D gSrcDepthTex;

SamplerState gSampler;

cbuffer DepthOfField
{
    float planeOfFocus;
    float aperture;
    float focalLength;
    float nearZ;
    float farZ;
};

float4 depthOfField(float2 texC)
{
   // read depth from scene render and calculate CoC per pixel
   float depth = gSrcDepthTex.SampleLevel(gSampler, texC, 0).r;

   float objectDistance = -farZ * nearZ / (depth * (farZ - nearZ) - farZ);

   float circleOfConfusionRadius = aperture * (focalLength * (objectDistance - planeOfFocus));
   float denom = max(0.00001, abs(objectDistance * (planeOfFocus - focalLength)));
   circleOfConfusionRadius /= denom;
   circleOfConfusionRadius = min(abs(circleOfConfusionRadius), 5.0f);

   float low = floor(circleOfConfusionRadius);
   float high = ceil(circleOfConfusionRadius);
	
	float4 lowSample = gSrcTex[int(low)].SampleLevel(gSampler, texC, 0);
	float4 highSample = gSrcTex[int(high)].SampleLevel(gSampler, texC, 0);
	
   return  lerp(lowSample, highSample, circleOfConfusionRadius - low);
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    float4 fragColor = float4(1.f, 1.f, 1.f, 1.f);
    fragColor = depthOfField(texC);
    return fragColor;
}