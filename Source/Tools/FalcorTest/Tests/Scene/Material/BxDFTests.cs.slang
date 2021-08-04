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

import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.UniformSampleGenerator;
import Experimental.Scene.Material.BxDF;

struct SamplingTest
{
    // Avoid sampling the poles of the spherical domain due to numerical precision.
    static double kCosThetaMargin = 1e-3;

    uint testCount;
    uint phiBinCount;
    uint cosThetaBinCount;
    uint sampleCount;
    uint threadSampleCount;
    uint binSampleCount;

    StructuredBuffer<float3> testWo;
    StructuredBuffer<float4> testParams;

    RWStructuredBuffer<uint> histogramSampling;     ///< Written during tabulateHistogram().
    RWStructuredBuffer<double> histogramPdf;        ///< Written during tabulatePdf().
    RWStructuredBuffer<uint> histogramWeightError;  ///< Written during tabulateWeightAndPdfError().
    RWStructuredBuffer<uint> histogramPdfError;     ///< Written during tabulateWeightAndPdfError().

    void tabulateHistogram(uint3 threadId)
    {
        uint threadCount = sampleCount / threadSampleCount;
        uint testIndex = threadId.z;
        if (threadId.x >= threadCount || testIndex >= testCount) return;

        const float3 wo = normalize(testWo[testIndex]);
        const float4 params = testParams[testIndex];

        TEST_BxDF bxdf;
        TEST_BxDF_INIT;

        // Create sample generator.
        UniformSampleGenerator sg = UniformSampleGenerator.create(threadId.xy, 0);

        for (uint i = 0; i < threadSampleCount; ++i)
        {
            float3 wi;
            float pdf;
            float3 weight;
            uint lobe;
            if (bxdf.sample(wo, wi, pdf, weight, lobe, sg))
            {
                if (wi.z < -1.0 + kCosThetaMargin || wi.z > 1.0 - kCosThetaMargin) continue;
                uint cosThetaBin = clamp((uint)floor(((double)wi.z * 0.5 + 0.5) * cosThetaBinCount), 0, cosThetaBinCount - 1);
                double scaledPhi = atan2((double)wi.y, (double)wi.x) * M_1_2PI;
                if (scaledPhi < 0.0) scaledPhi += 1.0;
                uint phiBin = clamp((uint)floor(scaledPhi * phiBinCount), 0, phiBinCount - 1);

                uint binIndex = testIndex * (phiBinCount * cosThetaBinCount) + cosThetaBin * phiBinCount + phiBin;
                InterlockedAdd(histogramSampling[binIndex], 1);
            }
        }
    }

    void tabulatePdf(uint3 threadId)
    {
        uint phiBin = threadId.x;
        uint cosThetaBin = threadId.y;
        uint testIndex = threadId.z;
        if (phiBin >= phiBinCount || cosThetaBin >= cosThetaBinCount || testIndex >= testCount) return;

        const float3 wo = normalize(testWo[testIndex]);
        const float4 params = testParams[testIndex];

        TEST_BxDF bxdf;
        TEST_BxDF_INIT;

        double phiStart =  phiBin      * M_2PI / phiBinCount;
        double phiEnd   = (phiBin + 1) * M_2PI / phiBinCount;

        double cosThetaStart = -1.0 +  cosThetaBin      * 2.0 / cosThetaBinCount;
        double cosThetaEnd   = -1.0 + (cosThetaBin + 1) * 2.0 / cosThetaBinCount;

        cosThetaStart = max(cosThetaStart, -1.0 + kCosThetaMargin);
        cosThetaEnd = min(cosThetaEnd, 1.0 - kCosThetaMargin);

        double pdf = 0.0;

        for (uint ix = 0; ix < binSampleCount; ++ix)
        {
            for (uint iy = 0; iy < binSampleCount; ++iy)
            {
                double phi = lerp(phiStart, phiEnd, ((double)ix + 0.5) / (double)binSampleCount);
                double cosTheta = lerp(cosThetaStart, cosThetaEnd, ((double)iy + 0.5) / (double)binSampleCount);

                float sinTheta = sqrt(1.f - cosTheta * cosTheta);
                float sinPhi = sin(phi);
                float cosPhi = cos(phi);
                float3 wi = normalize(float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

                pdf += bxdf.evalPdf(wo, wi);
            }
        }

        pdf *= (cosThetaEnd - cosThetaStart) * (phiEnd - phiStart) / (double)(binSampleCount * binSampleCount);

        uint binIndex = testIndex * (phiBinCount * cosThetaBinCount) + cosThetaBin * phiBinCount + phiBin;
        histogramPdf[binIndex] = pdf * sampleCount;
    }

    void tabulateWeightAndPdfError(uint3 threadId)
    {
        uint threadCount = sampleCount / threadSampleCount;
        uint threadIndex = threadId.x;
        uint testIndex = threadId.z;
        if (threadIndex >= threadCount || testIndex >= testCount) return;

        const float3 wo = normalize(testWo[testIndex]);
        const float4 params = testParams[testIndex];

        TEST_BxDF bxdf;
        TEST_BxDF_INIT;

        // Create sample generator.
        UniformSampleGenerator sg = UniformSampleGenerator.create(threadId.xy, 0);

        for (uint i = 0; i < threadSampleCount; ++i)
        {
            float3 wi;
            float pdf;
            float3 weight;
            uint lobe;
            if (bxdf.sample(wo, wi, pdf, weight, lobe, sg))
            {
                float3 value = bxdf.eval(wo, wi);
                float pdfRef = bxdf.evalPdf(wo, wi);
                float3 weightRef = pdfRef > 0.f ? (value / pdfRef) : float3(0.f);

                float3 weightErrors = abs(weight - weightRef) / (weightRef > 0.f ? weightRef : 1.f);
                float weightError = max(max(weightErrors.x, weightErrors.y), weightErrors.z);

                float pdfError = abs(pdf - pdfRef) / (pdfRef > 0.f ? pdfRef : 1.f);

                uint cosThetaBin = clamp((uint)floor(((double)wi.z * 0.5 + 0.5) * cosThetaBinCount), 0, cosThetaBinCount - 1);
                double scaledPhi = atan2((double)wi.y, (double)wi.x) * M_1_2PI;
                if (scaledPhi < 0.0) scaledPhi += 1.0;
                uint phiBin = clamp((uint)floor(scaledPhi * phiBinCount), 0, phiBinCount - 1);

                uint binIndex = testIndex * (phiBinCount * cosThetaBinCount) + cosThetaBin * phiBinCount + phiBin;
                InterlockedMax(histogramWeightError[binIndex], asuint(weightError));
                InterlockedMax(histogramPdfError[binIndex], asuint(pdfError));
            }
        }
    }
};

cbuffer SamplingTestCB
{
    SamplingTest gSamplingTest;
};

[numthreads(256, 1, 1)]
void tabulateHistogram(uint3 threadId : SV_DispatchThreadID)
{
    gSamplingTest.tabulateHistogram(threadId);
}

[numthreads(16, 16, 1)]
void tabulatePdf(uint3 threadId : SV_DispatchThreadID)
{
    gSamplingTest.tabulatePdf(threadId);
}

[numthreads(256, 1, 1)]
void tabulateWeightAndPdfError(uint3 threadId : SV_DispatchThreadID)
{
    gSamplingTest.tabulateWeightAndPdfError(threadId);
}
