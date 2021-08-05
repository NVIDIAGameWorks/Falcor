/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Math/MathConstants.slangh"
#include "flip.hlsli"

#define GroupX 32
#define GroupY 32

Texture2D gInputA;
Texture2D gInputB;
RWTexture2D<float4> gOutput;

cbuffer PerFrameCB
{
    uint2 gDimensions;
    bool gUseMagma;

    //  viewing conditions, for PPD calculation
    uint gMonitorWidthPixels;
    float gMonitorWidthMeters;
    float gMonitorDistance;
    uint2 pad;
};

static const float gqc = 0.7;
static const float gpc = 0.4;
static const float gpt = 0.95;
static const float gw = 0.082;
static const float gqf = 0.5;

static float MaxDistance = pow(HyAB(Hunt(linearRGB2CIELab(float3(0.0, 1.0, 0.0))), Hunt(linearRGB2CIELab(float3(0.0, 0.0, 1.0)))), gqc);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//  Pixels per degree (PPD)
float ppd(void)
{
    return gMonitorDistance * (gMonitorWidthPixels / gMonitorWidthMeters) * (M_PI / 180.0);
}

float calculateWeight(float dist2, float4 ab)
{
    float b1Inv = 1.0 / ab.z;
    float b2Inv = 1.0 / ab.w;
    return ab.x * sqrt(Pi * b1Inv) * exp(b1Inv * dist2) + ab.y * sqrt(Pi * b2Inv) * exp(b2Inv * dist2);
}

float3 gaussianFilter(uint3 loc3d, int radius, float4 abValuesA, float4 abValuesRG, float4 abValuesBY, bool useChannelA)
{
    float3 kernelSum = float3(0.0, 0.0, 0.0);
    float3 colorSum = float3(0.0, 0.0, 0.0);

    const float dx = 1.0 / ppd();
    //[unroll]
    for (int y = -radius; y <= radius; y++)
    {
        //[unroll]
        for (int x = -radius; x <= radius; x++)
        {
            int2 loc2d = loc3d.xy + int2(x, y);
            loc2d = min(max(int2(0, 0), loc2d), gDimensions-1);

            float2 p = float2(x, y) * dx;
            float dist2 = -(p.x * p.x + p.y * p.y) * PiSquared;
            float weightA = calculateWeight(dist2, abValuesA);
            float weightRG = calculateWeight(dist2, abValuesRG);
            float weightBY = calculateWeight(dist2, abValuesBY);
            float3 weight = float3(weightA, weightRG, weightBY);

            float3 color = (useChannelA ? gInputA[loc2d].rgb : gInputB[loc2d].rgb);

            kernelSum += weight;
            colorSum += weight * linearRGB2YCxCz(color); //< Inputs are in linear space
        }
    }

    float3 value = colorSum / kernelSum;
    value = clamp(YCxCz2LinearRGB(value), 0.0, 1.0);

    return linearRGB2CIELab(value);
}


float3 filterCSF(uint3 loc3d, bool bUseChannelA)
{
    const float4 abValuesA = { 1.0, 0.0, 0.0047, 1e-5 };  //  a1, a2, b1, b2 for A
    const float4 abValuesRG = { 1.0, 0.0, 0.0053, 1e-5 };  //  a1, a2, b1, b2 for RG
    const float4 abValuesBY = { 34.1, 13.5, 0.04, 0.025 };  //  a1, a2, b1, b2 for BY

    const float b1Max = max(max(abValuesA.z, abValuesRG.z), abValuesBY.z);
    const float b2Max = max(max(abValuesA.w, abValuesRG.w), abValuesBY.w);
    const float bMax = max(b1Max, b2Max);

    // 3 sigmas --> 99.7% of the signal
    const float sigmaFactor = 3.0f;

    int radius = int(ceil(sqrt(bMax / (2.0 * PiSquared)) * sigmaFactor * ppd()));

    return gaussianFilter(loc3d, radius, abValuesA, abValuesRG, abValuesBY, bUseChannelA);
}


float2 gradient(uint3 loc3d, bool bPointInsteadOfEdge, bool bUseInputA)
{
    float sigma = 0.5 * gw * ppd();
    int radius = int(ceil(3.0 * sigma));

    float sigmaSquared = sigma * sigma;

    //  Gaussian derivative
    float positiveKernelSum = 0.0;
    float negativeKernelSum = 0.0;

    float kernelSum = 0.0;

    float2 weight = { 0.0, 0.0 };
    float2 normalization = { 0.0, 0.0 };

    //[unroll]
    for (int y = -radius; y <= radius; y++)
    {
        //[unroll]
        for (int x = -radius; x <= radius; x++)
        {
            int2 loc2d = loc3d.xy + int2(x, y);
            loc2d = min(max(int2(0, 0), loc2d), gDimensions - 1);

            float g = exp(-(x * x + y * y) / (2.0 * sigmaSquared));

            if (bPointInsteadOfEdge)
            {
                weight = (float2(x * x, y * y) / sigmaSquared - 1) * g;

                positiveKernelSum += (weight.x >= 0.0 ? weight.x : 0.0);
                negativeKernelSum += (weight.x < 0.0 ? -weight.x : 0.0);
            }
            else
            {
                weight = -float2(x, y) * g;

                kernelSum += (weight.x >= 0.0 ? weight.x : 0.0);
            }
        }
    }

    float2 gradient = float2(0.0, 0.0);

    //[unroll]
    for (int y = -radius; y <= radius; y++)
    {
        //[unroll]
        for (int x = -radius; x <= radius; x++)
        {
            int2 loc2d = loc3d.xy + int2(x, y);
            loc2d = min(max(int2(0, 0), loc2d), gDimensions - 1);

            float g = exp(-(x * x + y * y) / (2.0 * sigmaSquared));

            if (bPointInsteadOfEdge)
            {
                weight = (float2(x * x, y * y) / sigmaSquared - 1) * g;

                normalization = float2(1.0, 1.0) / float2(weight.x >= 0.0 ? positiveKernelSum : negativeKernelSum, weight.y >= 0.0 ? positiveKernelSum : negativeKernelSum);
            }
            else
            {
                weight = -float2(x, y) * g;

                normalization = float2(1.0, 1.0) / float2(kernelSum, kernelSum);
            }
            
            float luminance = (linearRGB2YCxCz(bUseInputA ? gInputA[loc2d].rgb : gInputB[loc2d].rgb).x + 16.0) / 116.0; //< Inputs are in linear space
            gradient += luminance * weight * normalization;
        }
    }

    return gradient;
}


float featureDetection(uint3 loc3d, bool pointInsteadOfEdge)
{
    float2 gA = gradient(loc3d, pointInsteadOfEdge, true);
    float2 gB = gradient(loc3d, pointInsteadOfEdge, false);

    return abs(length(gA) - length(gB));
}


float featureDifference(uint3 loc3d)
{
    float pointDifference = featureDetection(loc3d, true);
    float edgeDifference = featureDetection(loc3d, false);

    return pow(max(pointDifference, edgeDifference) * InvSqrt2, gqf);
}


float HyAB(float3 a, float3 b)
{
    float3 diff = a - b;

    return abs(diff.x) + length(diff.yz);
}


float3 Hunt(float3 color)
{
    float huntValue = 0.01 * color.x;
    return float3(color.x, huntValue * color.y, huntValue * color.z);
}


float colorDifference(uint3 loc3d)
{
    float3 A = Hunt(filterCSF(loc3d, true));
    float3 B = Hunt(filterCSF(loc3d, false));

    return HyAB(A, B);
}


float errorFunction(float colorDifference, float featureDifference)
{
    float error = pow(colorDifference, gqc);

    //  Normalization
    float perceptualCutoff = gpc * MaxDistance;

    if (error < perceptualCutoff)
    {
        error *= (gpt / perceptualCutoff);
    }
    else
    {
        error = gpt + ((error - perceptualCutoff) / (MaxDistance - perceptualCutoff)) * (1.0 - gpt);
    }

    error = pow(error, (1.0 - featureDifference));

    return error;
}


float FLIP(uint3 loc3d)
{
    return errorFunction(colorDifference(loc3d), featureDifference(loc3d));
}


[numthreads(GroupX, GroupY, 1)]
void main(uint3 globalId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadId)
{
    uint2 loc2d = globalId.xy;
    uint3 loc3d = uint3(loc2d, 0);

    float value = FLIP(loc3d);

    if (isnan(value) || isinf(value) || value < 0.0 || value > 1.0)
        gOutput[loc2d] = float4(1.0, 0.0, 0.0, 1.0);
    else
        gOutput[loc2d] = float4((gUseMagma ? MagmaMap[int(value * 255.0 + 0.5)] : float3(value)), value);
}
