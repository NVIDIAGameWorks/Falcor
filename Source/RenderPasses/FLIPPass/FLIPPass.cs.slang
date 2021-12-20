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

import ToneMappers;

Texture2D gTestImage;
Texture2D gReferenceImage;
RWTexture2D<float4> gFLIPErrorMap;
RWTexture2D<float4> gFLIPErrorMapDisplay;
RWTexture2D<float4> gExposureMapDisplay;

cbuffer PerFrameCB
{
    uint2 gResolution;
    bool gIsHDR;
    bool gUseMagma;
    bool gClampInput;

    // HDR-FLIP constants.
    float gStartExposure;
    float gExposureDelta;
    uint gNumExposures;

    //  Viewing conditions for PPD calculation.
    uint gMonitorWidthPixels;
    float gMonitorWidthMeters;
    float gMonitorDistance;
    uint2 pad;
};

static const FLIPToneMapperType kHDRFLIPToneMapper = FLIPToneMapperType(TONE_MAPPER);

static const float gqc = 0.7f;
static const float gpc = 0.4f;
static const float gpt = 0.95f;
static const float gw = 0.082f;
static const float gqf = 0.5f;

static float MaxDistance = pow(HyAB(Hunt(linearRGB2CIELab(float3(0.0f, 1.0f, 0.0f))), Hunt(linearRGB2CIELab(float3(0.0f, 0.0f, 1.0f)))), gqc);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//  Pixels per degree (PPD)
float ppd(void)
{
    return gMonitorDistance * (gMonitorWidthPixels / gMonitorWidthMeters) * (M_PI / 180.0f);
}

float3 getPixel(Texture2D image, int2 pixel, float exposure)
{
    float3 pixelCol = image[pixel].rgb;
    if (gIsHDR)
    {
        pixelCol = gClampInput ? max(pixelCol, 0.0f) : pixelCol;
        pixelCol = pow(2.0f, exposure) * pixelCol; // Exposure compensation.
        pixelCol = toneMap(pixelCol, kHDRFLIPToneMapper);
    }
    else
    {
        pixelCol = gClampInput ? clamp(pixelCol, 0.0f, 1.0f) : pixelCol;
    }

    return linearRGB2YCxCz(pixelCol);
}

float calculateWeight(float dist2, float4 ab)
{
    float b1Inv = 1.0f / ab.z;
    float b2Inv = 1.0f / ab.w;
    return ab.x * sqrt(Pi * b1Inv) * exp(b1Inv * dist2) + ab.y * sqrt(Pi * b2Inv) * exp(b2Inv * dist2);
}

float HyAB(float3 a, float3 b)
{
    float3 diff = a - b;

    return abs(diff.x) + length(diff.yz);
}

float3 Hunt(float3 color)
{
    float huntValue = 0.01f * color.x;
    return float3(color.x, huntValue * color.y, huntValue * color.z);
}

float redistributeErrors(float colorDifference, float featureDifference)
{
    float error = pow(colorDifference, gqc);

    //  Normalization.
    float perceptualCutoff = gpc * MaxDistance;

    if (error < perceptualCutoff)
    {
        error *= (gpt / perceptualCutoff);
    }
    else
    {
        error = gpt + ((error - perceptualCutoff) / (MaxDistance - perceptualCutoff)) * (1.0f - gpt);
    }

    error = pow(error, (1.0f - featureDifference));

    return error;
}

float LDRFLIP(uint2 pixel, float exposure = 0.0f)
{
    // ************* SETUP ************** //
    // Compute pixels per degree.
    const float pixelsPerDegree = ppd();
    const float dx = 1.0f / ppd();

    // Variables for CSF filtering.
    const float4 abValuesA = { 1.0f, 0.0f, 0.0047f, 1.0e-5f };  //  a1, a2, b1, b2 for A.
    const float4 abValuesRG = { 1.0f, 0.0f, 0.0053f, 1.0e-5f };  //  a1, a2, b1, b2 for RG.
    const float4 abValuesBY = { 34.1f, 13.5f, 0.04f, 0.025f };  //  a1, a2, b1, b2 for BY.
    float3 colorWeight = { 0.0f, 0.0f };
    float3 csfKernelSum = float3(0.0f, 0.0f, 0.0f);
    float3 referenceColorSum = float3(0.0f, 0.0f, 0.0f);
    float3 testColorSum = float3(0.0f, 0.0f, 0.0f);
    float3 spatialFilteredReference = float3(0.0f);
    float3 spatialFilteredTest = float3(0.0f);

    // Variables for feature detection.
    float sigmaFeatures = 0.5f * gw * pixelsPerDegree;
    float sigmaFeaturesSquared = sigmaFeatures * sigmaFeatures;
    float positiveKernelSum = 0.0f;
    float negativeKernelSum = 0.0f;
    float edgeKernelSum = 0.0f;
    float2 pointWeight = { 0.0f, 0.0f };
    float2 edgeWeight = { 0.0f, 0.0f };
    float2 pointNormalization = { 0.0f, 0.0f };
    float2 edgeNormalization = { 0.0f, 0.0f };
    float2 referenceEdgeGradient = float2(0.0f, 0.0f);
    float2 referencePointGradient = float2(0.0f, 0.0f);
    float2 testEdgeGradient = float2(0.0f, 0.0f);
    float2 testPointGradient = float2(0.0f, 0.0f);

    // Use radius of the spatial filter kernel, as it is always greater than or equal to the radius of the feature detection kernel.
    int radius = int(ceil(3.0f * sqrt(0.04f / (2.0f * PiSquared)) * pixelsPerDegree)); // See FLIP paper for explanation of the 0.04 and 3.0 factors.

    // Prepare point and edge kernel sums for feature detection.
    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            int2 pixelNeighbor = pixel + int2(x, y);
            pixelNeighbor = min(max(int2(0, 0), pixelNeighbor), gResolution - 1);

            float g = exp(-(x * x + y * y) / (2.0f * sigmaFeaturesSquared));

            pointWeight = (float2(x * x, y * y) / sigmaFeaturesSquared - 1) * g;
            positiveKernelSum += (pointWeight.x >= 0.0f ? pointWeight.x : 0.0f);
            negativeKernelSum += (pointWeight.x < 0.0f ? -pointWeight.x : 0.0f);

            edgeWeight = -float2(x, y) * g;
            edgeKernelSum += (edgeWeight.x >= 0.0f ? edgeWeight.x : 0.0f);
        }
    }

    //[unroll]
    for (int y = -radius; y <= radius; y++)
    {
        //[unroll]
        for (int x = -radius; x <= radius; x++)
        {
            // Load pixel colors.
            int2 pixelNeighbor = pixel + int2(x, y);
            pixelNeighbor = min(max(int2(0, 0), pixelNeighbor), gResolution - 1);
            float3 referenceColor = getPixel(gReferenceImage, pixelNeighbor, exposure);
            float3 testColor = getPixel(gTestImage, pixelNeighbor, exposure);


            // **************** COLOR PIPELINE ******************** //
            float2 p = float2(x, y) * dx;
            float dist2 = -(p.x * p.x + p.y * p.y) * PiSquared;
            float weightA = calculateWeight(dist2, abValuesA);
            float weightRG = calculateWeight(dist2, abValuesRG);
            float weightBY = calculateWeight(dist2, abValuesBY);
            colorWeight = float3(weightA, weightRG, weightBY);

            csfKernelSum += colorWeight;
            referenceColorSum += colorWeight * referenceColor;
            testColorSum += colorWeight * testColor;


            // **************** FEATURE PIPELINE ******************** //
            float g = exp(-(x * x + y * y) / (2.0f * sigmaFeaturesSquared));

            pointWeight = (float2(x * x, y * y) / sigmaFeaturesSquared - 1) * g;
            pointNormalization = float2(1.0f, 1.0f) / float2(pointWeight.x >= 0.0f ? positiveKernelSum : negativeKernelSum, pointWeight.y >= 0.0f ? positiveKernelSum : negativeKernelSum);
            edgeWeight = -float2(x, y) * g;
            edgeNormalization = float2(1.0f, 1.0f) / float2(edgeKernelSum, edgeKernelSum);

            float referenceLuminance = (referenceColor.x + 16.0f) / 116.0f; // Normalized Y from YCxCz.
            referencePointGradient += referenceLuminance * pointWeight * pointNormalization;
            referenceEdgeGradient += referenceLuminance * edgeWeight * edgeNormalization;

            float testLuminance = (testColor.x + 16.0f) / 116.0f; // Normalized Y from YCxCz.
            testPointGradient += testLuminance * pointWeight * pointNormalization;
            testEdgeGradient += testLuminance * edgeWeight * edgeNormalization;
        }
    }

    // **************** COLOR PIPELINE ******************** //
    spatialFilteredReference = referenceColorSum / csfKernelSum;
    spatialFilteredTest = testColorSum / csfKernelSum;
    spatialFilteredReference = clamp(YCxCz2LinearRGB(spatialFilteredReference), 0.0f, 1.0f);
    spatialFilteredTest = clamp(YCxCz2LinearRGB(spatialFilteredTest), 0.0f, 1.0f);
    float colorDiff = HyAB(Hunt(linearRGB2CIELab(spatialFilteredReference)), Hunt(linearRGB2CIELab(spatialFilteredTest)));

    // **************** FEATURE PIPELINE ******************** //
    float edgeDifference = abs(length(referenceEdgeGradient) - length(testEdgeGradient));
    float pointDifference = abs(length(referencePointGradient) - length(testPointGradient));
    float featureDiff = pow(max(pointDifference, edgeDifference) * InvSqrt2, gqf);

    return redistributeErrors(colorDiff, featureDiff);
}

float HDRFLIP(uint2 pixel)
{
    float exposure;
    float ldrflip;
    float hdrflip = 0.0f;
    uint exposureMapIndex = 0u;

    // HDR-FLIP is maximum LDR-FLIP over a range of exposures.
    for (uint i = 0; i < gNumExposures; i++)
    {
        exposure = gStartExposure + i * gExposureDelta;
        ldrflip = LDRFLIP(pixel, exposure);
        if (ldrflip > hdrflip)
        {
            hdrflip = ldrflip;
            exposureMapIndex = i;
        }
    }

    // Store exposure map.
    float exposureMapFloatIndex = float(exposureMapIndex) / (gNumExposures - 1.0f);
    gExposureMapDisplay[pixel] = float4(sRGB2LinearRGB(ViridisMap[int(exposureMapFloatIndex * 255.0f + 0.5f)]), 1.0f); // See reasoning for sRGB2LinearRGB in main().

    return hdrflip;
}

float FLIP(uint2 pixel)
{
    if (gIsHDR) return HDRFLIP(pixel);
    else return LDRFLIP(pixel);
}


[numthreads(32, 32, 1)]
void main(uint3 globalId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadId)
{
    uint2 pixel = globalId.xy;

    float value = FLIP(pixel);

    if (isnan(value) || isinf(value) || value < 0.0f || value > 1.0f)
    {
        gFLIPErrorMap[pixel] = float4(1.0f, 0.0f, 0.0f, 1.0f);
        gFLIPErrorMapDisplay[pixel] = float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        // Store FLIP error after possible magma conversion.
        float3 col = gUseMagma ? MagmaMap[int(value * 255.0f + 0.5f)] : float3(value);
        gFLIPErrorMap[pixel] = float4(col, value);

        /*
        Before storing for display, apply an sRGB2LinearRGB transform to make the final output (when displayed)
        the same as what other FLIP implementations generate. This is necessary because, after this shader,
        which stores linear RGB in the FLIP error map, is run, the buffer will be blit:ed to an sRGB buffer.
        The blit includes a LinearRGB2sRGB transform. To cancel the effect of that transform, the sRGB2LinearRGB
        transform is added here. This also applies to the exposure map.
        */
        gFLIPErrorMapDisplay[pixel] = float4(sRGB2LinearRGB(col), 1.0f);
    }
}
