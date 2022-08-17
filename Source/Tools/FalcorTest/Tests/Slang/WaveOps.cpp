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
#include <random>

namespace Falcor
{
    namespace
    {
        const char kShaderFilename[] = "Tests/Slang/WaveOps.cs.slang";
        const uint32_t kNumElems = 32 * 128;
        std::mt19937 rng;

        std::vector<uint32_t> generateMatchData(size_t numElems, size_t numUnique = 20)
        {
            std::vector<uint32_t> elems(numUnique);
            for (auto& e : elems) e = rng();

            std::uniform_int_distribution<size_t> select(0, numUnique-1);
            std::vector<uint32_t> data(numElems);
            for (auto& d : data) d = elems[select(rng)];

            return data;
        }

        std::vector<uint4> computeMatchResult(const std::vector<uint32_t>& data, uint32_t laneCount)
        {
            FALCOR_ASSERT(laneCount >= 4 && laneCount <= 128);
            std::vector<uint4> masks(data.size(), uint4(0));

            for (size_t i = 0; i < data.size(); i++)
            {
                size_t firstLane = (i / laneCount) * laneCount;
                uint32_t currentLaneValue = data[i];
                uint4& mask = masks[i];

                for (uint32_t j = 0; j < laneCount; j++)
                {
                    if (data[firstLane + j] == currentLaneValue)
                    {
                        mask[j >> 5] |= (1u << (j & 0x1f));
                    }
                }
            }

            return masks;
        }

        std::vector<float> computeMinMaxResult(const std::vector<float>& data, uint32_t laneCount, bool conditional)
        {
            FALCOR_ASSERT(laneCount >= 4 && laneCount <= 128);
            std::vector<float> result(data.size() * 2);

            for (size_t i = 0; i < data.size(); i += laneCount)
            {
                float minVal = INFINITY;
                float maxVal = -INFINITY;
                for (uint32_t j = 0; j < laneCount; j++)
                {
                    float val = data[i + j];
                    if (!conditional || (val - std::floor(val) < 0.5f))
                    {
                        minVal = std::min(minVal, data[i + j]);
                        maxVal = std::max(maxVal, data[i + j]);
                    }
                }
                for (uint32_t j = 0; j < laneCount; j++)
                {
                    float val = data[i + j];
                    if (!conditional || (val - std::floor(val) < 0.5f))
                    {
                        result[2 * (i + j) + 0] = minVal;
                        result[2 * (i + j) + 1] = maxVal;
                    }
                    else
                    {
                        result[2 * (i + j) + 0] = 0.f;
                        result[2 * (i + j) + 1] = 0.f;
                    }
                }
            }

            return result;
        }

        void testWaveMinMax(GPUUnitTestContext& ctx, bool conditional)
        {
            Program::DefineList defines = { { "CONDITIONAL", conditional ? "1" : "0" } };
            ctx.createProgram(kShaderFilename, "testWaveMinMax", defines, Shader::CompilerFlags::None, "6_0");
            ctx.allocateStructuredBuffer("result", kNumElems * 2);

            auto var = ctx.vars().getRootVar();
            uint32_t zero = 0;
            auto pLaneCount = Buffer::createTyped<uint32_t>(1, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
            var["laneCount"] = pLaneCount;

            std::uniform_real_distribution<float> u(0.f, 1.f);
            std::vector<float> testData(kNumElems);
            for (size_t i = 0; i < testData.size(); i += 32)
            {
                float offset = 10.f * u(rng) - 5.f;
                for (size_t j = 0; j < 32; j++) testData[i + j] = offset + 2.f * u(rng) - 1.f;
            }
            var["testData"] = Buffer::createTyped<uint32_t>(kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, (uint32_t*)testData.data());

            ctx.runProgram(kNumElems, 1, 1);

            // Get the lane count. We abort the test if it is an unsupported count.
            const uint32_t laneCount = *(const uint32_t*)pLaneCount->map(Buffer::MapType::Read);
            pLaneCount->unmap();
            if (laneCount < 4 || laneCount > 128) throw RuntimeError("Unsupported wave lane count");

            // Verify results of wave min/max.
            std::vector<float> expectedResult = computeMinMaxResult(testData, laneCount, conditional);
            FALCOR_ASSERT(expectedResult.size() == testData.size() * 2);

            const float4* result = ctx.mapBuffer<const float4>("result");
            for (size_t i = 0; i < testData.size(); i++)
            {
                EXPECT_EQ(result[2 * i + 0].x, expectedResult[2 * i + 0]) << "WaveActiveMin (i = " << i << ")";
                EXPECT_EQ(result[2 * i + 1].x, expectedResult[2 * i + 1]) << "WaveActiveMax (i = " << i << ")";
            }
            ctx.unmapBuffer("result");
        }

        uint32_t queryLaneCount(GPUUnitTestContext& ctx)
        {
            ctx.createProgram(kShaderFilename, "testWaveGetLaneCount", Program::DefineList(), Shader::CompilerFlags::None, "6_0");

            auto var = ctx.vars().getRootVar();
            uint32_t zero = 0;
            auto pLaneCount = Buffer::createTyped<uint32_t>(1, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
            var["laneCount"] = pLaneCount;

            ctx.runProgram(1, 1, 1);

            const uint32_t laneCount = *(const uint32_t*)pLaneCount->map(Buffer::MapType::Read);
            pLaneCount->unmap();

            return laneCount;
        }
    }

    GPU_TEST(WaveGetLaneCount)
    {
        uint32_t laneCount = queryLaneCount(ctx);
        EXPECT_GE(laneCount, 4u);
        EXPECT_LE(laneCount, 128u);
    }

    // WaveMatch intrinsic is available only on D3D12.
    GPU_TEST_D3D12(WaveMatch)
    {
        ctx.createProgram(kShaderFilename, "testWaveMatch", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ctx.allocateStructuredBuffer("result", kNumElems);

        auto var = ctx.vars().getRootVar();
        uint32_t zero = 0;
        auto pLaneCount = Buffer::createTyped<uint32_t>(1, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
        var["laneCount"] = pLaneCount;

        std::vector<uint32_t> matchData = generateMatchData(kNumElems);
        FALCOR_ASSERT(matchData.size() == kNumElems);
        var["testData"] = Buffer::createTyped<uint32_t>(kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, matchData.data());

        ctx.runProgram(kNumElems, 1, 1);

        // Get the lane count. We abort the test if it is an unsupported count.
        const uint32_t laneCount = *(const uint32_t*)pLaneCount->map(Buffer::MapType::Read);
        pLaneCount->unmap();
        if (laneCount < 4 || laneCount > 128) throw RuntimeError("Unsupported wave lane count");

        // Verify results of wave match.
        std::vector<uint4> expectedResult = computeMatchResult(matchData, laneCount);
        FALCOR_ASSERT(expectedResult.size() == matchData.size());

        const uint4* result = ctx.mapBuffer<const uint4>("result");
        for (size_t i = 0; i < matchData.size(); i++)
        {
            EXPECT_EQ(result[i].x, expectedResult[i].x) << "i = " << i;
            EXPECT_EQ(result[i].y, expectedResult[i].y) << "i = " << i;
            EXPECT_EQ(result[i].z, expectedResult[i].z) << "i = " << i;
            EXPECT_EQ(result[i].w, expectedResult[i].w) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }

    GPU_TEST(WaveMinMax)
    {
        testWaveMinMax(ctx, false);
    }

    GPU_TEST(WaveMinMaxConditional, "Disabled due to compiler issues")
    {
        testWaveMinMax(ctx, true);
    }

    GPU_TEST(WaveMaxSimpleFloat, "Disabled due to compiler issues")
    {
        // Minimal test for floating point WaveActiveMax inside control flow.
        // The max across all lanes with value <= -2 is computed, the rest are unmodified.
        // Expected outcome:
        // Input:  -15,-14, ..., -3, -2, -1, ..., 16
        // Output:  -2, -2, ..., -2, -2, -1, ..., 16

        if (uint32_t laneCount = queryLaneCount(ctx); laneCount != 32) throw SkippingTestException("Test assumes warp size 32");

        std::vector<float> testData(32);
        for (size_t i = 0; i < testData.size(); i++) testData[i] = (float)i - 15;

        ctx.createProgram(kShaderFilename, "testWaveMaxSimpleFloat", Program::DefineList(), Shader::CompilerFlags::None, "6_0");
        ctx.allocateStructuredBuffer("result", 32);

        auto var = ctx.vars().getRootVar();
        var["testData"] = Buffer::createTyped<uint32_t>(32, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, (uint32_t*)testData.data());
        ctx.runProgram(32, 1, 1);

        // Verify result.
        const float4* result = ctx.mapBuffer<const float4>("result");
        for (size_t i = 0; i < 32; i++)
        {
            float expected = testData[i] <= -2.f ? -2.f : testData[i];
            EXPECT_EQ(result[i].x, expected) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }

    GPU_TEST(WaveMaxSimpleInt)
    {
        // Minimal test for integer WaveActiveMax inside control flow.
        // The max across all lanes with value <= -2 is computed, the rest are unmodified.
        // Expected outcome:
        // Input:  -15,-14, ..., -3, -2, -1, ..., 16
        // Output:  -2, -2, ..., -2, -2, -1, ..., 16

        if (uint32_t laneCount = queryLaneCount(ctx); laneCount != 32) throw SkippingTestException("Test assumes warp size 32");

        std::vector<int> testData(32);
        for (size_t i = 0; i < testData.size(); i++) testData[i] = (int)i - 15;

        ctx.createProgram(kShaderFilename, "testWaveMaxSimpleInt", Program::DefineList(), Shader::CompilerFlags::None, "6_0");
        ctx.allocateStructuredBuffer("result", 32);

        auto var = ctx.vars().getRootVar();
        var["testData"] = Buffer::createTyped<uint32_t>(32, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, (uint32_t*)testData.data());
        ctx.runProgram(32, 1, 1);

        // Verify result.
        const int4* result = ctx.mapBuffer<const int4>("result");
        for (size_t i = 0; i < 32; i++)
        {
            int expected = testData[i] <= -2 ? -2 : testData[i];
            EXPECT_EQ(result[i].x, expected) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }
}
