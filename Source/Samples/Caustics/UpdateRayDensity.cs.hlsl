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
#include "Common.hlsl"

cbuffer PerFrameCB
{
    int2 coarseDim;
    float minPhotonPixelSize;
    float smoothWeight;
    int maxTaskPerPixel;
    float updateSpeed;
    float varianceGain;
    float derivativeGain;
};

StructuredBuffer<PixelInfo> gPixelInfo;
RWStructuredBuffer<RayArgument> gRayArgument;
RWTexture2D<float4> gRayDensityTex;

void getAreaValue(uint2 pos, out float avgArea, out float avgArea2)
{
    int idx = coarseDim.x * pos.y + pos.x;
    avgArea = (gPixelInfo[idx].screenArea / float(gPixelInfo[idx].count + 0.01f));
    avgArea2 = (gPixelInfo[idx].screenAreaSq / float(gPixelInfo[idx].count + 0.01f));
}

[numthreads(16, 16, 1)]
void updateRayDensityTex(uint3 threadIdx : SV_DispatchThreadID)
{
    if (all(threadIdx == uint3(0, 0, 0)))
    {
        gRayArgument[0].rayTaskCount = 0;
    }

    int idx = coarseDim.x * threadIdx.y + threadIdx.x;
    
    float area, areaSq;
    getAreaValue(threadIdx.xy, area, areaSq);
    float stdVariance = (abs(areaSq - area * area));

    float targetArea = minPhotonPixelSize * minPhotonPixelSize;

    float4 value = gRayDensityTex[threadIdx.xy];
    float oldDensity = value.r;
    float weight = smoothWeight * updateSpeed;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(1, 0)].r * weight;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(0, 1)].r * weight;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(-1, 0)].r * weight;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(0, -1)].r * weight;
    oldDensity /= (1 + weight * 4);

    //float density = 1 / (area + 0.001);
    //float targetDensity = 1 / (targetArea * 0.01 + 0.001);
    //float newDensity = oldDensity + updateSpeed * 100 * (targetDensity - density) + stdVariance * varianceGain * 0.01 - value.g * derivativeGain;

    float newDensity = oldDensity * area / targetArea + stdVariance * varianceGain * 0.01 - value.g * derivativeGain;
    newDensity = newDensity * updateSpeed + oldDensity * (1 - updateSpeed);
    newDensity = clamp(newDensity, 0.1, maxTaskPerPixel);
    gRayDensityTex[threadIdx.xy] = float4(newDensity, newDensity - value.r, 0, 0);

    //gPixelInfo[idx].screenArea = 0;
    //gPixelInfo[idx].screenAreaSq = 0;
    //gPixelInfo[idx].count = 0;
}
