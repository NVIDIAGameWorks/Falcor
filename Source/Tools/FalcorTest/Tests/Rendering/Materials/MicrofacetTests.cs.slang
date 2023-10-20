/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Sampling.UniformSampleGenerator;
import Rendering.Materials.Microfacet;

struct MicrofacetSamplingTest
{
    // Avoid sampling the poles of the spherical domain due to numerical precision.
    static float kCosThetaMargin = 1e-3f;

    uint testCount;
    uint phiBinCount;
    uint cosThetaBinCount;
    uint sampleCount;
    uint threadSampleCount;
    uint binSampleCount;

    bool visibleNormals;
    float2 alpha;
    float rotation;

    StructuredBuffer<float3> testWi;

    RWStructuredBuffer<uint> histogramSampling; ///< Written during tabulateHistogram().
    RWStructuredBuffer<double> histogramPdf;    ///< Written during tabulatePdf().

    void tabulateHistogram(uint3 threadId)
    {
        uint threadCount = sampleCount / threadSampleCount;
        uint testIndex = threadId.z;
        if (threadId.x >= threadCount || testIndex >= testCount || threadId.y > 0)
            return;

        const float3 wi = normalize(testWi[testIndex]);

        Microfacet mf = Microfacet(TEST_NDF_TYPE(), alpha, rotation);

        // Create sample generator.
        UniformSampleGenerator sg = UniformSampleGenerator(threadId.xy, 0);

        for (uint i = 0; i < threadSampleCount; ++i)
        {
            float pdf;
            float3 wm;
            if (visibleNormals)
            {
                wm = mf.sampleVisible(wi, sg, pdf);
            }
            else
            {
                wm = mf.sampleFull(sg, pdf);
            }

            if (pdf > 0.0f)
            {
                if (wm.z < -1.f + kCosThetaMargin || wm.z > 1.f - kCosThetaMargin)
                    continue;
                uint cosThetaBin = clamp((uint)floor((wm.z * 0.5f + 0.5f) * cosThetaBinCount), 0, cosThetaBinCount - 1);
                float scaledPhi = atan2(wm.y, wm.x) * M_1_2PI;
                if (scaledPhi < 0.f)
                    scaledPhi += 1.f;
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
        if (phiBin >= phiBinCount || cosThetaBin >= cosThetaBinCount || testIndex >= testCount)
            return;

        const float3 wi = normalize(testWi[testIndex]);

        Microfacet mf = Microfacet(TEST_NDF_TYPE(), alpha, rotation);

        float phiStart = phiBin * M_2PI / phiBinCount;
        float phiEnd = (phiBin + 1) * M_2PI / phiBinCount;

        float cosThetaStart = -1.f + cosThetaBin * 2.f / cosThetaBinCount;
        float cosThetaEnd = -1.f + (cosThetaBin + 1) * 2.f / cosThetaBinCount;

        cosThetaStart = max(cosThetaStart, -1.f + kCosThetaMargin);
        cosThetaEnd = min(cosThetaEnd, 1.f - kCosThetaMargin);

        float pdf = 0.f;

        for (uint ix = 0; ix < binSampleCount; ++ix)
        {
            for (uint iy = 0; iy < binSampleCount; ++iy)
            {
                float phi = lerp(phiStart, phiEnd, (ix + 0.5f) / binSampleCount);
                float cosTheta = lerp(cosThetaStart, cosThetaEnd, (iy + 0.5f) / binSampleCount);

                float sinTheta = sqrt(1.f - cosTheta * cosTheta);
                float sinPhi = sin(phi);
                float cosPhi = cos(phi);
                float3 wm = normalize(float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

                if (visibleNormals)
                {
                    pdf += mf.evalPdfVisible(wi, wm);
                }
                else
                {
                    pdf += mf.evalPdfFull(wm);
                }
            }
        }

        pdf *= (cosThetaEnd - cosThetaStart) * (phiEnd - phiStart) / (binSampleCount * binSampleCount);

        uint binIndex = testIndex * (phiBinCount * cosThetaBinCount) + cosThetaBin * phiBinCount + phiBin;
        histogramPdf[binIndex] = pdf * sampleCount;
    }
};

ParameterBlock<MicrofacetSamplingTest> gMicrofacetSamplingTest;

[numthreads(256, 1, 1)]
void tabulateHistogram(uint3 threadId: SV_DispatchThreadID)
{
    gMicrofacetSamplingTest.tabulateHistogram(threadId);
}

[numthreads(16, 16, 1)]
void tabulatePdf(uint3 threadId: SV_DispatchThreadID)
{
    gMicrofacetSamplingTest.tabulatePdf(threadId);
}

StructuredBuffer<float3> testWi;
RWStructuredBuffer<float> result1;
RWStructuredBuffer<float> result2;

[numthreads(32, 1, 1)]
void sigmaIntegration(uint3 threadId: SV_DispatchThreadID)
{
    Microfacet mf = Microfacet(TEST_NDF_TYPE(), 1.f);
    float3 wi = testWi[threadId.x];
    UniformSampleGenerator sg = UniformSampleGenerator(threadId.xy, 0);
    double sum = 0.0;
    uint32_t N = 1000000;
    for (uint32_t i = 0; i < N; ++i)
    {
        float pdf;
        float3 wm = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        double integrand = max(0.f, dot(-wi, wm)) * mf.evalD(wm);
        sum += integrand / (double)pdf;
    }
    sum /= N;
    result1[threadId.x] = (float)sum;

    // Evaluate the analytical version of the function.
    result2[threadId.x] = mf.evalSigma(wi);
}

[numthreads(32, 1, 1)]
void sigmaLambdaConsistency(uint3 threadId: SV_DispatchThreadID)
{
    Microfacet mf = Microfacet(TEST_NDF_TYPE(), 1.f);
    float3 wi = testWi[threadId.x];
    result1[threadId.x] = mf.evalSigma(wi);
    result2[threadId.x] = mf.evalLambda(sign(wi.z) * wi); // Always evaluate for upfacing directions.
}

[numthreads(32, 1, 1)]
void lambdaNonsymmetry(uint3 threadId: SV_DispatchThreadID)
{
    Microfacet mf = Microfacet(TEST_NDF_TYPE(), 1.f);
    float3 wi = testWi[threadId.x];
    result1[threadId.x] = mf.evalLambda(wi);
    result2[threadId.x] = mf.evalLambda(-wi);
}

[numthreads(32, 1, 1)]
void g1Symmetry(uint3 threadId: SV_DispatchThreadID)
{
    Microfacet mf = Microfacet(TEST_NDF_TYPE(), 1.f);
    float3 wi = testWi[threadId.x];
    result1[threadId.x] = mf.evalG1(wi);
    result2[threadId.x] = mf.evalG1(-wi);
}
