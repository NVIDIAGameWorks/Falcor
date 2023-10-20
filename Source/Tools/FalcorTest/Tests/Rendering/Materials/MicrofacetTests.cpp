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
#include "Testing/UnitTest.h"
#include <hypothesis/hypothesis.h>
#include <iostream>

namespace Falcor
{
const char kShaderFile[] = "Tests/Rendering/Materials/MicrofacetTests.cs.slang";

// Print results as test is run (useful for development).
const bool kPrintResults = false;

// Dump the histograms generated for the chi^2 test and sampling weight/pdf error.
const bool kDumpHistograms = false;

// Dump the histograms only if the respective test fails.
const bool kDumpHistogramsOnFailedTest = false;

namespace
{
std::vector<std::string> kNdfs = {
    "TrowbridgeReitzNDF",
    "BeckmannSpizzichinoNDF",
};
}

void dumpHistogram(const std::filesystem::path& path, const double* pData, double factor, uint32_t width, uint32_t height)
{
    std::vector<float> img(width * height * 3, 0);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            float value = (float)(pData[y * width + x] * factor);
            size_t index = ((height - y - 1) * width + x) * 3;
            img[index + 0] = value;
            img[index + 1] = value;
            img[index + 2] = value;
        }
    }
    Bitmap::saveImage(
        path, width, height, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, ResourceFormat::RGB32Float, true, img.data()
    );
}

struct NDFConfig
{
    std::string name;
    float2 alpha;
    float rotation;
};

struct SamplingTestSpec
{
    bool visibleNormals;
    std::string ndf;
    NDFConfig ndfConfig;
    std::vector<float> incidentAngles;

    uint32_t phiBinCount = 128 + 2;
    uint32_t cosThetaBinCount = 64 + 1;
    uint32_t sampleCount = 64 * 1024 * 1024;
    uint32_t threadSampleCount = 128 * 1024;
    uint32_t binSampleCount = 128;
};

void setupSamplingTest(GPUUnitTestContext& ctx, const SamplingTestSpec& spec, const std::string& csEntry)
{
    ref<Device> pDevice = ctx.getDevice();

    uint32_t testCount = spec.visibleNormals ? (uint32_t)spec.incidentAngles.size() : 1;

    DefineList defines;
    defines.add("TEST_NDF_TYPE", spec.ndf);
    ctx.createProgram(kShaderFile, csEntry, defines);

    auto var = ctx.vars().getRootVar()["gMicrofacetSamplingTest"];
    var["testCount"] = testCount;
    var["phiBinCount"] = spec.phiBinCount;
    var["cosThetaBinCount"] = spec.cosThetaBinCount;
    var["sampleCount"] = spec.sampleCount;
    var["threadSampleCount"] = spec.threadSampleCount;
    var["binSampleCount"] = spec.binSampleCount;

    var["visibleNormals"] = spec.visibleNormals;
    var["alpha"] = spec.ndfConfig.alpha;
    var["rotation"] = spec.ndfConfig.rotation;

    std::vector<float3> testWi;
    for (uint32_t i = 0; i < testCount; ++i)
    {
        float theta = M_PI * spec.incidentAngles[i] / 180.f;
        testWi.push_back({std::sin(theta), 0.f, std::cos(theta)});
    }
    auto pTestWiBuffer = pDevice->createStructuredBuffer(
        var["testWi"], testCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, testWi.data()
    );
    var["testWi"] = pTestWiBuffer;
}

std::vector<std::vector<double>> tabulateHistogram(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
{
    ref<Device> pDevice = ctx.getDevice();

    uint32_t testCount = spec.visibleNormals ? (uint32_t)spec.incidentAngles.size() : 1;
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    setupSamplingTest(ctx, spec, "tabulateHistogram");

    auto var = ctx.vars().getRootVar()["gMicrofacetSamplingTest"];
    auto pHistogramBuffer = pDevice->createStructuredBuffer(var["histogramSampling"], testCount * binCount);
    var["histogramSampling"] = pHistogramBuffer;
    ctx.getRenderContext()->clearUAV(pHistogramBuffer->getUAV().get(), uint4(0));

    ctx.runProgram(spec.sampleCount / spec.threadSampleCount, 1, testCount);

    std::vector<uint32_t> histogramData = pHistogramBuffer->getElements<uint32_t>();
    std::vector<std::vector<double>> histograms;

    for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
    {
        std::vector<double> histogram(binCount);
        for (uint32_t i = 0; i < binCount; ++i)
            histogram[i] = (double)histogramData[testIndex * binCount + i];
        histograms.push_back(histogram);
    }

    return histograms;
}

std::vector<std::vector<double>> tabulatePdf(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
{
    ref<Device> pDevice = ctx.getDevice();

    uint32_t testCount = spec.visibleNormals ? (uint32_t)spec.incidentAngles.size() : 1;
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    setupSamplingTest(ctx, spec, "tabulatePdf");

    auto var = ctx.vars().getRootVar()["gMicrofacetSamplingTest"];
    auto pHistogramBuffer = pDevice->createStructuredBuffer(var["histogramPdf"], testCount * binCount);
    var["histogramPdf"] = pHistogramBuffer;

    ctx.runProgram(spec.phiBinCount, spec.cosThetaBinCount, testCount);

    std::vector<double> histogramData = pHistogramBuffer->getElements<double>();
    std::vector<std::vector<double>> histograms;

    for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
    {
        size_t offset = testIndex * binCount;
        histograms.push_back(std::vector<double>(histogramData.data() + offset, histogramData.data() + offset + binCount));
    }

    return histograms;
}

void testSampling(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
{
    uint32_t testCount = spec.visibleNormals ? (uint32_t)spec.incidentAngles.size() : 1;
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    // Tabulate sampling and pdf histograms for chi^2 test.
    auto obsFrequencies = tabulateHistogram(ctx, spec);
    auto expFrequencies = tabulatePdf(ctx, spec);

    for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
    {
        std::string sampleType = spec.visibleNormals ? "visible" : "all";
        std::string testName =
            spec.ndf + "__" + sampleType + "__" + spec.ndfConfig.name + "__" + std::to_string(spec.incidentAngles[testIndex]);

        if (kPrintResults)
            std::cout << testName << std::endl;

        // Execute chi^2 test to verify sampling.
        const auto [success, report] = hypothesis::chi2_test(
            (int)binCount, obsFrequencies[testIndex].data(), expFrequencies[testIndex].data(), (int)spec.sampleCount, 5.0, 0.01, testCount
        );
        if (kPrintResults && !success)
            std::cout << report << std::endl;
        EXPECT(success);

        // Dump various histograms to EXR files.
        if (kDumpHistograms || (!success && kDumpHistogramsOnFailedTest))
        {
            double factor = (double)binCount / (double)spec.sampleCount;
            dumpHistogram(testName + "_obs.exr", obsFrequencies[testIndex].data(), factor, spec.phiBinCount, spec.cosThetaBinCount);
            dumpHistogram(testName + "_exp.exr", expFrequencies[testIndex].data(), factor, spec.phiBinCount, spec.cosThetaBinCount);
        }
    }
}

GPU_TEST(MicrofacetSampling)
{
    std::vector<NDFConfig> ndfConfigs = {
        {
            "isotropic_high_roughness",
            {1.0f, 1.0f},
            0.f,
        },
        {
            "isotropic_medium_roughness",
            {0.6f, 0.6f},
            0.f,
        },
        {
            "anisotropic_axisaligned",
            {0.6f, 1.f},
            0.f,
        },
        {
            "anisotropic_rotated",
            {0.6f, 1.f},
            0.6f,
        },
    };

    SamplingTestSpec testSpec;
    testSpec.incidentAngles = {
        0.f,
        30.f,
        80.f,
        130.f,
    };

    if (kPrintResults)
        std::cout << std::endl;

    for (uint32_t i = 0; i < 2; ++i)
    {
        for (uint32_t j = 0; j < kNdfs.size(); ++j)
        {
            for (uint32_t k = 0; k < ndfConfigs.size(); ++k)
            {
                testSpec.visibleNormals = i;
                testSpec.ndf = kNdfs[j];
                testSpec.ndfConfig = ndfConfigs[k];
                testSampling(ctx, testSpec);
            }
        }
    }
}

GPU_TEST(MicrofacetSigmaIntegration)
{
    // Estimate the ground truth projected area (sigma) integral using Monte Carlo, see
    // "Additional Progress Towards the Unification of Microfacet and Microflake Theories"
    // by Dupuy et al. 2016, Eq. 18.
    for (uint32_t t = 0; t < kNdfs.size(); ++t)
    {
        DefineList defines;
        defines.add("TEST_NDF_TYPE", kNdfs[t]);
        ctx.createProgram(kShaderFile, "sigmaIntegration", defines);

        uint32_t N = 32;
        std::vector<float3> testWi;
        for (uint32_t i = 0; i < N; ++i)
        {
            float mu = -1 + 2 * float(i) / (N - 1);
            testWi.push_back({std::sqrt(1.f - mu * mu), 0.f, mu});
        }

        ctx.allocateStructuredBuffer("testWi", (uint32_t)testWi.size(), testWi.data());
        ctx.allocateStructuredBuffer("result1", N);
        ctx.allocateStructuredBuffer("result2", N);

        ctx.runProgram(N, 1, 1);
        std::vector<float> result1 = ctx.readBuffer<float>("result1");
        std::vector<float> result2 = ctx.readBuffer<float>("result2");
        for (uint32_t i = 0; i < N; ++i)
        {
            float sigmaRef = result1[i];
            float sigmaEval = result2[i];
            float diff = std::abs(sigmaRef - sigmaEval);
            EXPECT_LT(diff, 5e-3f);
        }
    }
}

GPU_TEST(MicrofacetSigmaLambdaConsistency)
{
    // Test the consistency of the projected area (sigma) and the Smith
    // Lambda function, see
    // "Additional Progress Towards the Unification of Microfacet and Microflake Theories"
    // by Dupuy et al. 2016, Eq. 18.
    for (uint32_t t = 0; t < kNdfs.size(); ++t)
    {
        DefineList defines;
        defines.add("TEST_NDF_TYPE", kNdfs[t]);
        ctx.createProgram(kShaderFile, "sigmaLambdaConsistency", defines);

        uint32_t N = 32;
        std::vector<float3> testWi;
        for (uint32_t i = 0; i < N; ++i)
        {
            float mu = -1 + 2 * float(i) / (N - 1);
            testWi.push_back({std::sqrt(1.f - mu * mu), 0.f, mu});
        }

        ctx.allocateStructuredBuffer("testWi", (uint32_t)testWi.size(), testWi.data());
        ctx.allocateStructuredBuffer("result1", N);
        ctx.allocateStructuredBuffer("result2", N);

        ctx.runProgram(N, 1, 1);
        std::vector<float> result1 = ctx.readBuffer<float>("result1");
        std::vector<float> result2 = ctx.readBuffer<float>("result2");
        for (uint32_t i = 0; i < N; ++i)
        {
            float mu = -1 + 2 * float(i) / (N - 1);

            float sigma = result1[i];
            float Lambda = result2[i];
            float rhs;
            if (mu > 0.0f)
            {
                rhs = Lambda * mu;
            }
            else
            {
                rhs = (1.0f + Lambda) * -mu;
            }
            float diff = std::abs(sigma - rhs);
            EXPECT_LT(diff, 1e-3f);
        }
    }
}

GPU_TEST(MicrofacetLambdaNonsymmetry)
{
    // Thest the non-symmetry of the Smith Lambda function, see
    // "Additional Progress Towards the Unification of Microfacet and Microflake Theories"
    // by Dupuy et al. 2016, Eq. 18 and 19.
    for (uint32_t t = 0; t < kNdfs.size(); ++t)
    {
        DefineList defines;
        defines.add("TEST_NDF_TYPE", kNdfs[t]);
        ctx.createProgram(kShaderFile, "lambdaNonsymmetry", defines);

        uint32_t N = 32;
        std::vector<float3> testWi;
        for (uint32_t i = 0; i < N; ++i)
        {
            float mu = std::max(1e-4f, float(i) / (N - 1));
            testWi.push_back({std::sqrt(1.f - mu * mu), 0.f, mu});
        }

        ctx.allocateStructuredBuffer("testWi", (uint32_t)testWi.size(), testWi.data());
        ctx.allocateStructuredBuffer("result1", N);
        ctx.allocateStructuredBuffer("result2", N);

        ctx.runProgram(N, 1, 1);
        std::vector<float> result1 = ctx.readBuffer<float>("result1");
        std::vector<float> result2 = ctx.readBuffer<float>("result2");
        for (uint32_t i = 0; i < N; ++i)
        {
            float LambdaPos = result1[i];
            float LambdaNeg = result2[i];
            float lhs = LambdaNeg;
            float rhs = 1 + LambdaPos;
            float diff = std::abs(lhs - rhs);
            EXPECT_LT(diff, 1e-3f);
        }
    }
}

GPU_TEST(MicrofacetG1Symmetry)
{
    // Test the symmetry of the Smith bistatic shadowing function G1.
    for (uint32_t t = 0; t < kNdfs.size(); ++t)
    {
        DefineList defines;
        defines.add("TEST_NDF_TYPE", kNdfs[t]);
        ctx.createProgram(kShaderFile, "g1Symmetry", defines);

        uint32_t N = 32;
        std::vector<float3> testWi;
        for (uint32_t i = 0; i < N; ++i)
        {
            float mu = -1 + 2 * float(i) / (N - 1);
            testWi.push_back({std::sqrt(1.f - mu * mu), 0.f, mu});
        }

        ctx.allocateStructuredBuffer("testWi", (uint32_t)testWi.size(), testWi.data());
        ctx.allocateStructuredBuffer("result1", N);
        ctx.allocateStructuredBuffer("result2", N);

        ctx.runProgram(N, 1, 1);
        std::vector<float> result1 = ctx.readBuffer<float>("result1");
        std::vector<float> result2 = ctx.readBuffer<float>("result2");
        for (uint32_t i = 0; i < N; ++i)
        {
            float g1Pos = result1[i];
            float g1Neg = result2[i];
            float diff = std::abs(g1Pos - g1Neg);
            EXPECT_LT(diff, 1e-3f);
        }
    }
}

} // namespace Falcor
