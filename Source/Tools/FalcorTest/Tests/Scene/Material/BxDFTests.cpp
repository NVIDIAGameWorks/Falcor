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
#include "Testing/UnitTest.h"
#include <hypothesis/hypothesis.h>
#include <random>
#include <fstream>
#include <iostream>

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Tests/Scene/Material/BxDFTests.cs.slang";

        // Print results as test is run (useful for development).
        const bool kPrintResults = false;

        // Dump the histograms generated for the chi^2 test and sampling weight/pdf error.
        const bool kDumpHistograms = false;

        const double kMaxWeightError = 1e-3;
        const double kMaxPdfError = 1e-3;

        struct BxdfConfig
        {
            std::string name;
            float3 wi;
            float4 params;

            static std::vector<float3> getWi(const std::vector<BxdfConfig>& configs)
            {
                std::vector<float3> wi;
                std::transform(configs.begin(), configs.end(), std::back_inserter(wi), [](auto const& config) { return config.wi; });
                return wi;
            }

            static std::vector<float4> getParams(const std::vector<BxdfConfig>& configs)
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
            Bitmap::saveImage(path, width, height, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, ResourceFormat::RGB32Float, true, img.data());
        }

        struct SamplingTestSpec
        {
            std::string bxdf;
            std::string bxdfInit;
            std::vector<BxdfConfig> bxdfConfigs;
            uint32_t phiBinCount = 128 + 2;
            uint32_t cosThetaBinCount = 64 + 1;
            uint32_t sampleCount = 64 * 1024 * 1024;
            uint32_t threadSampleCount = 128 * 1024;
            uint32_t binSampleCount = 128;
        };

        void setupSamplingTest(GPUUnitTestContext& ctx, const SamplingTestSpec& spec, const std::string& csEntry)
        {
            FALCOR_ASSERT(!spec.bxdf.empty());
            FALCOR_ASSERT(!spec.bxdfConfigs.empty());
            FALCOR_ASSERT(spec.sampleCount > spec.threadSampleCount && spec.sampleCount % spec.threadSampleCount == 0);

            uint32_t testCount = (uint32_t)spec.bxdfConfigs.size();
            uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

            Program::DefineList defines;
            defines.add("TEST_BxDF", spec.bxdf);
            defines.add("TEST_BxDF_INIT", spec.bxdfInit);
            ctx.createProgram(kShaderFile, csEntry, defines);

            auto var = ctx["SamplingTestCB"]["gSamplingTest"];
            var["testCount"] = testCount;
            var["phiBinCount"] = spec.phiBinCount;
            var["cosThetaBinCount"] = spec.cosThetaBinCount;
            var["sampleCount"] = spec.sampleCount;
            var["threadSampleCount"] = spec.threadSampleCount;
            var["binSampleCount"] = spec.binSampleCount;

            auto testWi = BxdfConfig::getWi(spec.bxdfConfigs);
            auto pTestWiBuffer = Buffer::createStructured(var["testWi"], testCount, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, testWi.data());
            var["testWi"] = pTestWiBuffer;

            auto testParams = BxdfConfig::getParams(spec.bxdfConfigs);
            auto pTestParamsBuffer = Buffer::createStructured(var["testParams"], testCount, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, testParams.data());
            var["testParams"] = pTestParamsBuffer;
        }

        std::vector<std::vector<double>> tabulateHistogram(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
        {
            uint32_t testCount = (uint32_t)spec.bxdfConfigs.size();
            uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

            setupSamplingTest(ctx, spec, "tabulateHistogram");

            auto var = ctx["SamplingTestCB"]["gSamplingTest"];
            auto pHistogramBuffer = Buffer::createStructured(var["histogramSampling"], testCount * binCount);
            var["histogramSampling"] = pHistogramBuffer;
            ctx.getRenderContext()->clearUAV(pHistogramBuffer->getUAV().get(), uint4(0));

            ctx.runProgram(spec.sampleCount / spec.threadSampleCount, 1, testCount);

            auto pHistogramData = reinterpret_cast<const uint32_t *>(pHistogramBuffer->map(Buffer::MapType::Read));

            std::vector<std::vector<double>> histograms;

            for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
            {
                std::vector<double> histogram(binCount);
                for (uint32_t i = 0; i < binCount; ++i) histogram[i] = (double)pHistogramData[testIndex * binCount + i];
                histograms.push_back(histogram);
            }

            pHistogramBuffer->unmap();

            return histograms;
        }

        std::vector<std::vector<double>> tabulatePdf(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
        {
            uint32_t testCount = (uint32_t)spec.bxdfConfigs.size();
            uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

            setupSamplingTest(ctx, spec, "tabulatePdf");

            auto var = ctx["SamplingTestCB"]["gSamplingTest"];
            auto pHistogramBuffer = Buffer::createStructured(var["histogramPdf"], testCount * binCount);
            var["histogramPdf"] = pHistogramBuffer;

            ctx.runProgram(spec.phiBinCount, spec.cosThetaBinCount, testCount);

            auto pHistogramData = reinterpret_cast<const double *>(pHistogramBuffer->map(Buffer::MapType::Read));

            std::vector<std::vector<double>> histograms;

            for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
            {
                size_t offset = testIndex * binCount;
                histograms.push_back(std::vector<double>(pHistogramData + offset, pHistogramData + offset + binCount));
            }

            pHistogramBuffer->unmap();

            return histograms;
        }

        std::vector<std::pair<std::vector<double>, std::vector<double>>> tabulateWeightAndPdfError(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
        {
            uint32_t testCount = (uint32_t)spec.bxdfConfigs.size();
            uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

            setupSamplingTest(ctx, spec, "tabulateWeightAndPdfError");

            auto var = ctx["SamplingTestCB"]["gSamplingTest"];
            auto pHistogramWeightBuffer = Buffer::createStructured(var["histogramWeightError"], testCount * binCount);
            var["histogramWeightError"] = pHistogramWeightBuffer;
            ctx.getRenderContext()->clearUAV(pHistogramWeightBuffer->getUAV().get(), uint4(0));

            auto pHistogramPdfBuffer = Buffer::createStructured(var["histogramPdfError"], testCount * binCount);
            var["histogramPdfError"] = pHistogramPdfBuffer;
            ctx.getRenderContext()->clearUAV(pHistogramPdfBuffer->getUAV().get(), uint4(0));

            ctx.runProgram(spec.sampleCount / spec.threadSampleCount, 1, testCount);

            auto pHistogramWeightData = reinterpret_cast<const float *>(pHistogramWeightBuffer->map(Buffer::MapType::Read));
            auto pHistogramPdfData = reinterpret_cast<const float *>(pHistogramPdfBuffer->map(Buffer::MapType::Read));

            std::vector<std::pair<std::vector<double>, std::vector<double>>> histograms;

            for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
            {
                std::vector<double> histogramWeight(binCount);
                for (uint32_t i = 0; i < binCount; ++i) histogramWeight[i] = (double)pHistogramWeightData[testIndex * binCount + i];

                std::vector<double> histogramPdf(binCount);
                for (uint32_t i = 0; i < binCount; ++i) histogramPdf[i] = (double)pHistogramPdfData[testIndex * binCount + i];

                histograms.emplace_back(histogramWeight, histogramPdf);
            }

            pHistogramWeightBuffer->unmap();
            pHistogramPdfBuffer->unmap();

            return histograms;
        }

        void testSampling(GPUUnitTestContext& ctx, const SamplingTestSpec& spec)
        {
            uint32_t testCount = (uint32_t)spec.bxdfConfigs.size();
            uint32_t binCount = spec.phiBinCount * spec.cosThetaBinCount;

            // Tabulate sampling and pdf histograms for chi^2 test.
            auto obsFrequencies = tabulateHistogram(ctx, spec);
            auto expFrequencies = tabulatePdf(ctx, spec);

            // Tabulate max relative error of weight and pdf values returned from sampling.
            auto weightAndPdfErrors = tabulateWeightAndPdfError(ctx, spec);

            for (uint32_t testIndex = 0; testIndex < testCount; ++testIndex)
            {
                auto testName = spec.bxdf + "_" + spec.bxdfConfigs[testIndex].name;

                if (kPrintResults) std::cout << testName << std::endl;

                // Execute chi^2 test to verify sampling.
                const auto [success, report] = hypothesis::chi2_test((int)binCount, obsFrequencies[testIndex].data(), expFrequencies[testIndex].data(), (int)spec.sampleCount, 5.0, 0.01, testCount);
                if (kPrintResults && !success) std::cout << report << std::endl;
                // TODO Enable this once tests are passing correctly.
                // EXPECT(success);

                // Check maximum weight error, indicating mismatch between weights returned from BxDF::sample() vs. BxDF::eval() / BxDF::evalPdf().
                const auto& weightError = weightAndPdfErrors[testIndex].first;
                double maxWeightError = 0.0;
                for (size_t i = 0; i < weightError.size(); ++i) maxWeightError = std::max(maxWeightError, weightError[i]);
                if (kPrintResults && maxWeightError > kMaxWeightError) std::cout << "Sampling weight mismatch! Max weight error = " << maxWeightError << std::endl;
                // TODO Enable this once tests are passing correctly.
                // EXPECT(maxWeightError <= kMaxWeightError);

                // Check maximum pdf error, indicating mismatch between pdf returned from BxDF::sample() vs. BxDF::evalPdf().
                const auto& pdfError = weightAndPdfErrors[testIndex].second;
                double maxPdfError = 0.0;
                for (size_t i = 0; i < pdfError.size(); ++i) maxPdfError = std::max(maxPdfError, pdfError[i]);
                if (kPrintResults && maxPdfError > kMaxPdfError) std::cout << "Sampling pdf mismatch! Max pdf error = " << maxPdfError << std::endl;
                // TODO Enable this once tests are passing correctly.
                // EXPECT(maxPdfError <= kMaxPdfError);

                // Dump various histograms to EXR files.
                if (kDumpHistograms)
                {
                    double factor = (double)binCount / (double)spec.sampleCount;
                    dumpHistogram(testName + "_obs.exr", obsFrequencies[testIndex].data(), factor, spec.phiBinCount, spec.cosThetaBinCount);
                    dumpHistogram(testName + "_exp.exr", expFrequencies[testIndex].data(), factor, spec.phiBinCount, spec.cosThetaBinCount);

                    dumpHistogram(testName + "_weight_error.exr", weightAndPdfErrors[testIndex].first.data(), 1.0, spec.phiBinCount, spec.cosThetaBinCount);
                    dumpHistogram(testName + "_pdf_error.exr", weightAndPdfErrors[testIndex].second.data(), 1.0, spec.phiBinCount, spec.cosThetaBinCount);
                }
            }
        }
    }

    GPU_TEST(BxDF_Sampling)
    {
        const float3 perp = normalize(float3(0.f, 0.f, 1.f));
        const float3 oblique = normalize(float3(0.5f, 0.f, 0.5f));
        const float3 grazing = normalize(float3(0.f, 1.f, 0.01f));

        testSampling(
            ctx,
            {
                "DiffuseReflectionLambert",
                "bxdf.albedo = float3(1.f);",
                {
                    { "perp",    perp,      { 0.f, 0.f, 0.f, 0.f } },
                    { "oblique", oblique,   { 0.f, 0.f, 0.f, 0.f } },
                    { "grazing", grazing,   { 0.f, 0.f, 0.f, 0.f } },
                }
            });

        testSampling(
            ctx,
            {
                "DiffuseReflectionDisney",
                "bxdf.albedo = float3(1.f); bxdf.roughness = 0.5f;",
                {
                    { "perp",    perp,      { 0.f, 0.f, 0.f, 0.f } },
                    { "oblique", oblique,   { 0.f, 0.f, 0.f, 0.f } },
                    { "grazing", grazing,   { 0.f, 0.f, 0.f, 0.f } },
                }
            });

        testSampling(
            ctx,
            {
                "DiffuseReflectionFrostbite",
                "bxdf.albedo = float3(1.f); bxdf.roughness = 0.5f;",
                {
                    { "perp",    perp,      { 0.f, 0.f, 0.f, 0.f } },
                    { "oblique", oblique,   { 0.f, 0.f, 0.f, 0.f } },
                    { "grazing", grazing,   { 0.f, 0.f, 0.f, 0.f } },
                }
            });

        testSampling(
            ctx,
            {
                "DiffuseTransmissionLambert",
                "bxdf.albedo = float3(1.f);",
                {
                    { "perp",    perp,      { 0.f, 0.f, 0.f, 0.f } },
                    { "oblique", oblique,   { 0.f, 0.f, 0.f, 0.f } },
                    { "grazing", grazing,   { 0.f, 0.f, 0.f, 0.f } },
                }
            });

        testSampling(
            ctx,
            {
                "SpecularReflectionMicrofacet",
                "bxdf.albedo = float3(1.f); bxdf.activeLobes = 0xff; bxdf.alpha = params.x;",
                {
                    { "smooth_perp",    perp,       { 0.05f, 0.f, 0.f, 0.f } },
                    { "smooth_oblique", oblique,    { 0.05f, 0.f, 0.f, 0.f } },
                    { "smooth_grazing", grazing,    { 0.05f, 0.f, 0.f, 0.f } },
                    { "rough_perp",     perp,       { 0.5f, 0.f, 0.f, 0.f } },
                    { "rough_oblique",  oblique,    { 0.5f, 0.f, 0.f, 0.f } },
                    { "rough_grazing",  grazing,    { 0.5f, 0.f, 0.f, 0.f } },
                }
            });

        testSampling(
            ctx,
            {
                "SpecularReflectionTransmissionMicrofacet",
                "bxdf.activeLobes = 0xff; bxdf.transmissionAlbedo = float3(1.f); bxdf.alpha = params.x; bxdf.eta = params.y;",
                {
                    { "to_glass_smooth_perp",       perp,       { 0.05f, 0.67f, 0.f, 0.f } },
                    { "to_glass_smooth_oblique",    oblique,    { 0.05f, 0.67f, 0.f, 0.f } },
                    { "to_glass_smooth_grazing",    grazing,    { 0.05f, 0.67f, 0.f, 0.f } },
                    { "to_glass_rough_perp",        perp,       { 0.5f, 0.67f, 0.f, 0.f } },
                    { "to_glass_rough_oblique",     oblique,    { 0.5f, 0.67f, 0.f, 0.f } },
                    { "to_glass_rough_grazing",     grazing,    { 0.5f, 0.67f, 0.f, 0.f } },
                    { "from_glass_smooth_perp",     perp,       { 0.05f, 1.5f, 0.f, 0.f } },
                    { "from_glass_smooth_oblique",  oblique,    { 0.05f, 1.5f, 0.f, 0.f } },
                    { "from_glass_smooth_grazing",  grazing,    { 0.05f, 1.5f, 0.f, 0.f } },
                    { "from_glass_rough_perp",      perp,       { 0.5f, 1.5f, 0.f, 0.f } },
                    { "from_glass_rough_oblique",   oblique,    { 0.5f, 1.5f, 0.f, 0.f } },
                    { "from_glass_rough_grazing",   grazing,    { 0.5f, 1.5f, 0.f, 0.f } },
                }
            });
    }

}  // namespace Falcor
