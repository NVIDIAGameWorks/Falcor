/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.HairChiang16;

StructuredBuffer<float> gBetaM;
StructuredBuffer<float> gBetaN;
StructuredBuffer<float> gAlpha;
StructuredBuffer<float> gIoR;
StructuredBuffer<float3> gSigmaA;
StructuredBuffer<float> gH;
StructuredBuffer<float3> gWi;
StructuredBuffer<float3> gWo;
RWStructuredBuffer<float3> gResultOurs;

StructuredBuffer<float2> roughness;
RWStructuredBuffer<float> result;

cbuffer TestCB
{
    uint resultSize;
    uint sampleCount;
};

[numthreads(1024, 1, 1)]
void testPbrtReference(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= resultSize) return;

    // Setup BCSDF.
    HairChiang16 bcsdf;
    bcsdf.betaM = gBetaM[idx];
    bcsdf.betaN = gBetaN[idx];
    bcsdf.alpha = gAlpha[idx];
    bcsdf.IoR = gIoR[idx];
    bcsdf.eta = 1.f / bcsdf.IoR;
    bcsdf.sigmaA = gSigmaA[idx];
    bcsdf.h = gH[idx];
    bcsdf.precompute();

    SampleGenerator sg = SampleGenerator(threadId.xy, 0);

    gResultOurs[idx] = bcsdf.eval(gWi[idx], gWo[idx], sg);
}

[numthreads(16, 1, 1)]
void testWhiteFurnaceUniform(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= resultSize) return;

    // Setup BCSDF.
    float betaM = roughness[threadId.x].x;
    float betaN = roughness[threadId.x].y;

    HairChiang16 bcsdf;
    bcsdf.betaM = betaM;
    bcsdf.betaN = betaN;
    bcsdf.alpha = 0.f;
    bcsdf.IoR = 1.55f;
    bcsdf.eta = 1.f / bcsdf.IoR;
    bcsdf.sigmaA = float3(0.f);

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(threadId.xy, threadId.z);

    float3 wi = sample_sphere(sampleNext2D(sg));

    float3 sum = float3(0.f);
    for (uint i = 0; i < sampleCount; i++)
    {
        bcsdf.h = -1.f + 2.f * sampleNext1D(sg);
        bcsdf.precompute();

        float3 wo = sample_sphere(sampleNext2D(sg));
        sum += bcsdf.eval(wi, wo, sg);
    }

    result[threadId.x] = luminance(sum) / sampleCount * M_4PI;
}

[numthreads(16, 1, 1)]
void testWhiteFurnaceImportanceSampling(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= resultSize) return;

    // Setup BCSDF.
    float betaM = roughness[threadId.x].x;
    float betaN = roughness[threadId.x].y;

    HairChiang16 bcsdf;
    bcsdf.betaM = betaM;
    bcsdf.betaN = betaN;
    bcsdf.alpha = 0.f;
    bcsdf.IoR = 1.55f;
    bcsdf.eta = 1.f / bcsdf.IoR;
    bcsdf.sigmaA = float3(0.f);

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(threadId.xy, threadId.z);

    float3 wi = sample_sphere(sampleNext2D(sg));

    float3 sum = float3(0.f);
    for (uint i = 0; i < sampleCount; i++)
    {
        bcsdf.h = -1.f + 2.f * sampleNext1D(sg);
        bcsdf.precompute();

        float3 wo, weight;
        float pdf;
        uint lobe;

        if (bcsdf.sample(wi, wo, pdf, weight, lobe, sg))
        {
            sum += weight;
        }
    }

    result[threadId.x] = luminance(sum) / sampleCount;
}

[numthreads(16, 16, 1)]
void testImportanceSamplingWeights(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= resultSize || threadId.y >= sampleCount) return;

    // Setup BCSDF.
    float betaM = roughness[threadId.x].x;
    float betaN = roughness[threadId.x].y;

    HairChiang16 bcsdf;
    bcsdf.betaM = betaM;
    bcsdf.betaN = betaN;
    bcsdf.alpha = 0.f;
    bcsdf.IoR = 1.55f;
    bcsdf.eta = 1.f / bcsdf.IoR;
    bcsdf.sigmaA = float3(0.f);

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(threadId.xy, threadId.z);

    bcsdf.h = -1.f + 2.f * sampleNext1D(sg);
    bcsdf.precompute();

    float3 wi = sample_sphere(sampleNext2D(sg));
    float3 wo, weight;
    float pdf;
    uint lobe;
    if (bcsdf.sample(wi, wo, pdf, weight, lobe, sg))
    {
        result[threadId.x * sampleCount + threadId.y] = luminance(weight);
    }
    else
    {
        result[threadId.x * sampleCount + threadId.y] = -1.f;
    }
}

[numthreads(16, 1, 1)]
void testSamplingConsistency(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= resultSize) return;

    // Setup BCSDF.
    float betaM = roughness[threadId.x].x;
    float betaN = roughness[threadId.x].y;

    HairChiang16 bcsdf;
    bcsdf.betaM = betaM;
    bcsdf.betaN = betaN;
    bcsdf.IoR = 1.55f;
    bcsdf.eta = 1.f / bcsdf.IoR;

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(threadId.xy, threadId.z);

    float3 color = sampleNext3D(sg);
    bcsdf.sigmaA = sigmaAFromColor(color, betaN);
    bcsdf.alpha = 5.f * sampleNext1D(sg);

    float3 wi = sample_sphere(sampleNext2D(sg));

    float3 fIS = float3(0.f);
    float3 fUniform = float3(0.f);
    for (uint i = 0; i < sampleCount; i++)
    {
        bcsdf.h = -1.f + 2.f * sampleNext1D(sg);
        bcsdf.precompute();

        float3 wo, weight;
        float pdf;
        uint lobe;

        // Assume Li = wo.z * wo.z.

        // Importance sampling.
        if (bcsdf.sample(wi, wo, pdf, weight, lobe, sg))
        {
            fIS += weight * (wo.z * wo.z) / sampleCount;
        }

        // Uniform sampling.
        wo = sample_sphere(sampleNext2D(sg));
        fUniform += bcsdf.eval(wi, wo, sg) * M_4PI * (wo.z * wo.z) / sampleCount;
    }

    result[threadId.x] = abs(luminance(fIS) - luminance(fUniform)) / luminance(fUniform);
}
