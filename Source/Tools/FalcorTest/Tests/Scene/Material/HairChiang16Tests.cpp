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
#include "Utils/Sampling/SampleGenerator.h"
#include <random>
#include <fstream>
#include <iostream>

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Tests/Scene/Material/HairChiang16Tests.cs.slang";

        void testWhiteFurnace(GPUUnitTestContext& ctx, const std::string& funcName, const double threshold)
        {
            uint32_t testCount = 0;
            std::vector<float2> testRoughness;

            for (float betaM = 0.1f; betaM < 1.f; betaM += 0.2f)
            {
                for (float betaN = 0.1f; betaN < 1.f; betaN += 0.2f)
                {
                    testRoughness.push_back(float2(betaM, betaN));
                    testCount++;
                }
            }

            // Create sample generator.
            SampleGenerator::SharedPtr pSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

            // Setup GPU test.
            auto defines = pSampleGenerator->getDefines();
            ctx.createProgram(kShaderFile, funcName, defines, Shader::CompilerFlags::None, "6_2");

            pSampleGenerator->setShaderData(ctx.vars().getRootVar());

            ctx.allocateStructuredBuffer("roughness", testCount, testRoughness.data(), testRoughness.size() * sizeof(float2));
            ctx.allocateStructuredBuffer("result", testCount);
            ctx["TestCB"]["resultSize"] = testCount;
            ctx["TestCB"]["sampleCount"] = 300000;

            ctx.runProgram(testCount);

            const float* result = ctx.mapBuffer<const float>("result");
            for (uint32_t i = 0; i < testCount; i++)
            {
                EXPECT_LE(std::abs(result[i] - 1.f), threshold) << "WhiteFurnaceTestCase" << i << ", expected " << 1 << ", got " << result[i];
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(HairChiang16_PbrtReference)
    {
        uint32_t testCount = 50000;

        // Load reference data.
        std::ifstream fin;

        std::filesystem::path fullPath;
        findFileInDataDirectories("pbrt_hair_bsdf.dat", fullPath);
        fin.open(fullPath, std::ios::in | std::ios::binary);
        if (!fin.is_open())
        {
            throw ErrorRunningTestException("Cannot find reference data file 'pbrt_hair_bsdf.dat'.");
        }

        std::vector<float> buf(testCount * 17);
        fin.read((char*)buf.data(), buf.size() * sizeof(float));
        fin.close();

        std::vector<float3> sigmaA(testCount);
        std::vector<float3> wi(testCount);
        std::vector<float3> wo(testCount);
        std::vector<float3> resultRef(testCount);
        for (uint32_t i = 0; i < testCount; i++)
        {
            sigmaA[i] = float3(buf[4 * testCount + i], buf[5 * testCount + i], buf[6 * testCount + i]);
            wi[i] = float3(buf[8 * testCount + i], buf[9 * testCount + i], buf[10 * testCount + i]);
            wo[i] = float3(buf[11 * testCount + i], buf[12 * testCount + i], buf[13 * testCount + i]);
            resultRef[i] = float3(buf[14 * testCount + i], buf[15 * testCount + i], buf[16 * testCount + i]);
        }

        // Create sample generator.
        SampleGenerator::SharedPtr pSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

        // Setup GPU test.
        auto defines = pSampleGenerator->getDefines();
        ctx.createProgram(kShaderFile, "testPbrtReference", defines, Shader::CompilerFlags::None, "6_2");

        ctx.allocateStructuredBuffer("gBetaM", testCount, buf.data());
        ctx.allocateStructuredBuffer("gBetaN", testCount, buf.data() + testCount);
        ctx.allocateStructuredBuffer("gAlpha", testCount, buf.data() + 2 * testCount);
        ctx.allocateStructuredBuffer("gIoR", testCount, buf.data() + 3 * testCount);
        ctx.allocateStructuredBuffer("gSigmaA", testCount, sigmaA.data(), sigmaA.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("gH", testCount, buf.data() + 7 * testCount);
        ctx.allocateStructuredBuffer("gWi", testCount, wi.data(), wi.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("gWo", testCount, wo.data(), wo.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("gResultOurs", testCount);
        ctx["TestCB"]["resultSize"] = testCount;

        ctx.runProgram(testCount);

        const float3* result = ctx.mapBuffer<const float3>("gResultOurs");
        for (uint32_t i = 0; i < testCount; i++)
        {
            for (uint32_t c = 0; c < 3; c++)
            {
                float relAbsError = std::abs(resultRef[i][c]) < 1e-6f ? 0.f : std::abs(result[i][c] - resultRef[i][c]) / resultRef[i][c];
                EXPECT_LE(relAbsError, 1e-3f) << "PbrtReferenceTestCase(" << i << ", " << c << "), expected " << resultRef[i][c] << ", got " << result[i][c];
            }
        }
        ctx.unmapBuffer("gResultOurs");
    }

    GPU_TEST(HairChiang16_WhiteFurnaceUniform)
    {
        testWhiteFurnace(ctx, "testWhiteFurnaceUniform", 0.05f);
    }

    GPU_TEST(HairChiang16_WhiteFurnaceImportanceSampling)
    {
        testWhiteFurnace(ctx, "testWhiteFurnaceImportanceSampling", 0.01f);
    }

    GPU_TEST(HairChiang16_ImportanceSamplingWeights)
    {
        uint32_t sampleCount = 10000;
        uint32_t testCount = 0;
        std::vector<float2> testRoughness;

        for (float betaM = 0.1f; betaM < 1.f; betaM += 0.2f)
        {
            for (float betaN = 0.1f; betaN < 1.f; betaN += 0.2f)
            {
                testRoughness.push_back(float2(betaM, betaN));
                testCount++;
            }
        }

        // Create sample generator.
        SampleGenerator::SharedPtr pSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

        // Setup GPU test.
        auto defines = pSampleGenerator->getDefines();
        ctx.createProgram(kShaderFile, "testImportanceSamplingWeights", defines, Shader::CompilerFlags::None, "6_2");

        pSampleGenerator->setShaderData(ctx.vars().getRootVar());

        ctx.allocateStructuredBuffer("roughness", testCount, testRoughness.data(), testRoughness.size() * sizeof(float2));
        ctx.allocateStructuredBuffer("result", testCount * sampleCount);
        ctx["TestCB"]["resultSize"] = testCount;
        ctx["TestCB"]["sampleCount"] = sampleCount;

        ctx.runProgram(testCount, sampleCount);

        const float* result = ctx.mapBuffer<const float>("result");
        for (uint32_t i = 0; i < testCount; i++)
        {
            for (uint32_t j = 0; j < sampleCount; j++)
            {
                uint32_t idx = i * sampleCount + j;
                if (std::abs(result[idx] + 1.f) < 1e-6f)
                {
                    // Importance sampling failed: NaNs appear in the sampling procedure.
                    std::cerr << "Importance sampling failed. Ignore this test case." << std::endl;
                    continue;
                }
                EXPECT_LE(std::abs(result[idx] - 1.f), 1e-3f) << "ImportanceSamplingWeightsTestCase(" << i << ", " << j << "), expected " << 1 << ", got " << result[idx];
            }
        }
        ctx.unmapBuffer("result");
    }

    GPU_TEST(HairChiang16_SamplingConsistency)
    {
        uint32_t sampleCount = 300000;
        uint32_t testCount = 0;
        std::vector<float2> testRoughness;

        for (float betaM = 0.1f; betaM < 1.f; betaM += 0.2f)
        {
            for (float betaN = 0.1f; betaN < 1.f; betaN += 0.2f)
            {
                testRoughness.push_back(float2(betaM, betaN));
                testCount++;
            }
        }

        // Create sample generator.
        SampleGenerator::SharedPtr pSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

        // Setup GPU test.
        auto defines = pSampleGenerator->getDefines();
        ctx.createProgram(kShaderFile, "testSamplingConsistency", defines, Shader::CompilerFlags::None, "6_2");

        pSampleGenerator->setShaderData(ctx.vars().getRootVar());

        ctx.allocateStructuredBuffer("roughness", testCount, testRoughness.data(), testRoughness.size() * sizeof(float2));
        ctx.allocateStructuredBuffer("result", testCount);
        ctx["TestCB"]["resultSize"] = testCount;
        ctx["TestCB"]["sampleCount"] = sampleCount;

        ctx.runProgram(testCount);

        const float* result = ctx.mapBuffer<const float>("result");
        for (uint32_t i = 0; i < testCount; i++)
        {
            EXPECT_LE(result[i], 0.05f) << "SamplingConsistencyTestCase" << i << ", expected " << 0 << ", got " << result[i];
        }
        ctx.unmapBuffer("result");
    }

}  // namespace Falcor
