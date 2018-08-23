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

__import Lights;

SamplerState gSampler;

Texture2D srcDepth;

cbuffer GodRaySettings
{
   float exposer;
   LightData light;
   float cameraRatio;
   float viewportHeight;
   float3 screenSpaceLightPosition;
};

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
	float3 finalColor = float3(0, 0, 0);
	float3 lightIntensity = light.intensity;
	float pixelDepth = srcDepth.SampleLevel(gSampler, texC, 0.0f).x;
	float2 toLightSource = (texC - screenSpaceLightPosition.xy);
	toLightSource.x *= cameraRatio;
	
	float mask  = ((pixelDepth < screenSpaceLightPosition.z) ? 0.0f : 1.0f);
	float dstToLightSqrd = max(0.001f, dot(toLightSource, toLightSource));
	
	finalColor += mask * lightIntensity * exposer / (dstToLightSqrd * viewportHeight);
    return float4(finalColor, 1.0f);
}
