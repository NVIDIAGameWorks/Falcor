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
#include <random>
#include <fstream>
#include <iostream>

namespace Falcor
{
namespace
{
const char kShaderFile[] = "Tests/Scene/Material/BSDFTests.cs.slang";

// Print results as test is run (useful for development).
const bool kPrintResults = false;

// Dump the histograms generated for the chi^2 test and sampling weight/pdf error.
const bool kDumpHistograms = false;

const double kMaxWeightError = 1e-3;
const double kMaxPdfError = 1e-3;

struct BsdfConfig
{
    std::string name;
    float3 wi;
    float4 params;

    static std::vector<float3> getWi(const std::vector<BsdfConfig>& configs)
    {
        std::vector<float3> wi;
        std::transform(configs.begin(), configs.end(), std::back_inserter(wi), [](auto const& config) { return config.wi; });
        return wi;
    }

    static std::vector<float4> getParams(const std::vector<BsdfConfig>& configs)
    {
        std::vector<float4> params;
        std::transform(configs.begin(), configs.end(), std::back_inserter(params), [](auto const& config) { return config.params; });
        return params;
    }
};

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

struct SamplingTestSpec
{
    std::string bsdfImport;
    std::string bsdf;
    std::string bsdfInit;
    std::vector<BsdfConfig> bsdfConfigs;
    uint32_t phiBinCount = 128 + 2;
    uint32_t cosThetaBinCount = 64 + 1;
    uint32_t sampleCount = 64 * 1024 * 1024;
    uint32_t threadSampleCount = 128 * 1024;
    uint32_t binSampleCount = 128;
};

void setupSamplingTest(GPUUnitTestContext& ctx, const SamplingTestSpec& spec, const std::string& csEntry)
{
    FALCOR_ASSERT(!spec.bsdfImport.empty());
    FALCOR_ASSERT(!spec.bsdf.empty());
    FALCOR_ASSERT(!spec.bsdfConfigs.empty());
    FALCOR_ASSERT(spec.sampleCount > spec.threadSampleCount && spec.sampleCount % spec.threadSampleCount == 0);

    ref<Device> pDevice = ctx.getDevice();

    uint32_t testCount = (uint32_t)spec.bsdfConfigs.size();
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    DefineList defines;
    defines.add("TEST_BSDF_IMPORT", spec.bsdfImport);
    defines.add("TEST_BSDF", spec.bsdf);
    defines.add("TEST_BSDF_INIT", spec.bsdfInit);
    ctx.createProgram(kShaderFile, csEntry, defines);

    auto var = ctx["SamplingTestCB"]["gSamplingTest"];
    var["testCount"] = testCount;
    var["phiBinCount"] = spec.phiBinCount;
    var["cosThetaBinCount"] = spec.cosThetaBinCount;
    var["sampleCount"] = spec.sampleCount;
    var["threadSampleCount"] = spec.threadSampleCount;
    var["binSampleCount"] = spec.binSampleCount;

    auto testWi = BsdfConfig::getWi(spec.bsdfConfigs);
    auto pTestWiBuffer = pDevice->createStructuredBuffer(
        var["testWi"], testCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, testWi.data()
    );
    var["testWi"] = pTestWiBuffer;

    auto testParams = BsdfConfig::getParams(spec.bsdfConfigs);
    auto pTestParamsBuffer = pDevice->createStructuredBuffer(
        var["testParams"], testCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, testParams.data()
    );
    var["testParams"] = pTestParamsBuffer;
}

std::vector<std::vector<double>> tabulateHistogram(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
{
    ref<Device> pDevice = ctx.getDevice();

    uint32_t testCount = (uint32_t)spec.bsdfConfigs.size();
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    setupSamplingTest(ctx, spec, "tabulateHistogram");

    auto var = ctx["SamplingTestCB"]["gSamplingTest"];
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

    uint32_t testCount = (uint32_t)spec.bsdfConfigs.size();
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    setupSamplingTest(ctx, spec, "tabulatePdf");

    auto var = ctx["SamplingTestCB"]["gSamplingTest"];
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

std::vector<std::pair<std::vector<double>, std::vector<double>>> tabulateWeightAndPdfError(
    GPUUnitTestContext& ctx,
    const SamplingTestSpec& spec
)
{
    ref<Device> pDevice = ctx.getDevice();

    uint32_t testCount = (uint32_t)spec.bsdfConfigs.size();
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    setupSamplingTest(ctx, spec, "tabulateWeightAndPdfError");

    auto var = ctx["SamplingTestCB"]["gSamplingTest"];
    auto pHistogramWeightBuffer = pDevice->createStructuredBuffer(var["histogramWeightError"], testCount * binCount);
    var["histogramWeightError"] = pHistogramWeightBuffer;
    ctx.getRenderContext()->clearUAV(pHistogramWeightBuffer->getUAV().get(), uint4(0));

    auto pHistogramPdfBuffer = pDevice->createStructuredBuffer(var["histogramPdfError"], testCount * binCount);
    var["histogramPdfError"] = pHistogramPdfBuffer;
    ctx.getRenderContext()->clearUAV(pHistogramPdfBuffer->getUAV().get(), uint4(0));

    ctx.runProgram(spec.sampleCount / spec.threadSampleCount, 1, testCount);

    std::vector<float> histogramWeightData = pHistogramWeightBuffer->getElements<float>();
    std::vector<float> histogramPdfData = pHistogramPdfBuffer->getElements<float>();

    std::vector<std::pair<std::vector<double>, std::vector<double>>> histograms;

    for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
    {
        std::vector<double> histogramWeight(binCount);
        for (uint32_t i = 0; i < binCount; ++i)
            histogramWeight[i] = (double)histogramWeightData[testIndex * binCount + i];

        std::vector<double> histogramPdf(binCount);
        for (uint32_t i = 0; i < binCount; ++i)
            histogramPdf[i] = (double)histogramPdfData[testIndex * binCount + i];

        histograms.emplace_back(histogramWeight, histogramPdf);
    }

    return histograms;
}

void testSampling(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
{
    uint32_t testCount = (uint32_t)spec.bsdfConfigs.size();
    uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

    // Tabulate sampling and pdf histograms for chi^2 test.
    auto obsFrequencies = tabulateHistogram(ctx, spec);
    auto expFrequencies = tabulatePdf(ctx, spec);

    // Tabulate max relative error of weight and pdf values returned from sampling.
    auto weightAndPdfErrors = tabulateWeightAndPdfError(ctx, spec);

    for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
    {
        auto testName = spec.bsdf + "_" + spec.bsdfConfigs[testIndex].name;

        if (kPrintResults)
            std::cout << testName << std::endl;

        // Execute chi^2 test to verify sampling.
        const auto [success, report] = hypothesis::chi2_test(
            (int)binCount, obsFrequencies[testIndex].data(), expFrequencies[testIndex].data(), (int)spec.sampleCount, 5.0, 0.01, testCount
        );
        if (kPrintResults && !success)
            std::cout << report << std::endl;
        // TODO Enable this once tests are passing correctly.
        EXPECT(success);

        // Check maximum weight error, indicating mismatch between weights returned from BSDF::sample() vs. BSDF::eval() / BSDF::evalPdf().
        const auto& weightError = weightAndPdfErrors[testIndex].first;
        double maxWeightError = 0.0;
        for (size_t i = 0; i < weightError.size(); ++i)
            maxWeightError = std::max(maxWeightError, weightError[i]);
        if (kPrintResults && maxWeightError > kMaxWeightError)
            std::cout << "Sampling weight mismatch! Max weight error = " << maxWeightError << std::endl;
        // TODO Enable this once tests are passing correctly.
        EXPECT_LE(maxWeightError, kMaxWeightError);

        // Check maximum pdf error, indicating mismatch between pdf returned from BSDF::sample() vs. BSDF::evalPdf().
        const auto& pdfError = weightAndPdfErrors[testIndex].second;
        double maxPdfError = 0.0;
        for (size_t i = 0; i < pdfError.size(); ++i)
            maxPdfError = std::max(maxPdfError, pdfError[i]);
        if (kPrintResults && maxPdfError > kMaxPdfError)
            std::cout << "Sampling pdf mismatch! Max pdf error = " << maxPdfError << std::endl;
        // TODO Enable this once tests are passing correctly.
        EXPECT_LE(maxPdfError, kMaxPdfError);

        // Dump various histograms to EXR files.
        if (kDumpHistograms)
        {
            double factor = (double)binCount / (double)spec.sampleCount;
            dumpHistogram(testName + "_obs.exr", obsFrequencies[testIndex].data(), factor, spec.phiBinCount, spec.cosThetaBinCount);
            dumpHistogram(testName + "_exp.exr", expFrequencies[testIndex].data(), factor, spec.phiBinCount, spec.cosThetaBinCount);

            dumpHistogram(
                testName + "_weight_error.exr", weightAndPdfErrors[testIndex].first.data(), 1.0, spec.phiBinCount, spec.cosThetaBinCount
            );
            dumpHistogram(
                testName + "_pdf_error.exr", weightAndPdfErrors[testIndex].second.data(), 1.0, spec.phiBinCount, spec.cosThetaBinCount
            );
        }
    }
}
} // namespace

GPU_TEST(TestBsdf_DisneyDiffuseBRDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.DisneyDiffuseBRDF",
         "DisneyDiffuseBRDF",
         "bsdf.albedo = float3(1.f); bsdf.roughness = 0.5f;",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_FrostbiteDiffuseBRDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.FrostbiteDiffuseBRDF",
         "FrostbiteDiffuseBRDF",
         "bsdf.albedo = float3(1.f); bsdf.roughness = 0.5f;",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_LambertDiffuseBRDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.LambertDiffuseBRDF",
         "LambertDiffuseBRDF",
         "bsdf.albedo = float3(1.f);",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_LambertDiffuseBTDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.LambertDiffuseBTDF",
         "LambertDiffuseBTDF",
         "bsdf.albedo = float3(1.f);",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_OrenNayarBRDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.OrenNayarBRDF",
         "OrenNayarBRDF",
         "bsdf.albedo = float3(1.f); bsdf.roughness = 0.5f",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_SimpleBTDF, "Disabled, sampling test makes no sense for diract BSDF")
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.SimpleBTDF",
         "SimpleBTDF",
         "bsdf.transmittance = float3(1.f);",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_SpecularMicrofacetBRDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.SpecularMicrofacet",
         "SpecularMicrofacetBRDF",
         "bsdf.albedo = float3(1.f); bsdf.activeLobes = 0xff; bsdf.alpha = params.x;",
         {
             {"smooth_perp", perp, {0.05f, 0.f, 0.f, 0.f}},
             {"smooth_oblique", oblique, {0.05f, 0.f, 0.f, 0.f}},
             {"smooth_grazing", grazing, {0.05f, 0.f, 0.f, 0.f}},
             {"rough_perp", perp, {0.5f, 0.f, 0.f, 0.f}},
             {"rough_oblique", oblique, {0.5f, 0.f, 0.f, 0.f}},
             {"rough_grazing", grazing, {0.5f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_SpecularMicrofacetBSDF, "Disabled, not passing")
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.SpecularMicrofacet",
         "SpecularMicrofacetBSDF",
         "bsdf.activeLobes = 0xff; bsdf.transmissionAlbedo = float3(1.f); bsdf.alpha = params.x; bsdf.eta = params.y;",
         {
             {"to_glass_smooth_perp", perp, {0.05f, 0.67f, 0.f, 0.f}},
             {"to_glass_smooth_oblique", oblique, {0.05f, 0.67f, 0.f, 0.f}},
             {"to_glass_smooth_grazing", grazing, {0.05f, 0.67f, 0.f, 0.f}},
             {"to_glass_rough_perp", perp, {0.5f, 0.67f, 0.f, 0.f}},
             {"to_glass_rough_oblique", oblique, {0.5f, 0.67f, 0.f, 0.f}},
             {"to_glass_rough_grazing", grazing, {0.5f, 0.67f, 0.f, 0.f}},
             {"from_glass_smooth_perp", perp, {0.05f, 1.5f, 0.f, 0.f}},
             {"from_glass_smooth_oblique", oblique, {0.05f, 1.5f, 0.f, 0.f}},
             {"from_glass_smooth_grazing", grazing, {0.05f, 1.5f, 0.f, 0.f}},
             {"from_glass_rough_perp", perp, {0.5f, 1.5f, 0.f, 0.f}},
             {"from_glass_rough_oblique", oblique, {0.5f, 1.5f, 0.f, 0.f}},
             {"from_glass_rough_grazing", grazing, {0.5f, 1.5f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_SheenBSDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.SheenBSDF",
         "SheenBSDF",
         "bsdf.color = float3(1.f); bsdf.roughness = 0.5f",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}

GPU_TEST(TestBsdf_DiffuseSpecularBRDF)
{
    const float3 perp = normalize(float3(0.f, 0.f, 1.f));
    const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
    const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

    testSampling(
        ctx,
        {"Rendering.Materials.BSDFs.DiffuseSpecularBRDF",
         "DiffuseSpecularBRDF",
         "bsdf.diffuse = float3(0.5f); bsdf.specular = float3(0.04f); bsdf.roughness = 0.5f",
         {
             {"perp", perp, {0.f, 0.f, 0.f, 0.f}},
             {"oblique", oblique, {0.f, 0.f, 0.f, 0.f}},
             {"grazing", grazing, {0.f, 0.f, 0.f, 0.f}},
         }}
    );
}
} // namespace Falcor
