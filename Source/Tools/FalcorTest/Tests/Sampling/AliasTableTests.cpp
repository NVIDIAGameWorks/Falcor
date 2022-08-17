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
#include "Utils/Sampling/AliasTable.h"

#include <hypothesis/hypothesis.h>

#include <iostream>

namespace Falcor
{
    namespace
    {
        void testAliasTable(GPUUnitTestContext& ctx, uint32_t N, std::vector<float> specificWeights = {})
        {
            std::mt19937 rng;
            std::uniform_real_distribution<float> uniform;

            // Use specificed weights or generate pseudo-random weights.
            std::vector<float> weights(N);
            for (uint32_t i = 0; i < N; ++i) weights[i] = i < specificWeights.size() ? specificWeights[i] : uniform(rng);

            // Add a few zero weights.
            if (N >= 100)
            {
                for (uint32_t i = 0; i < N / 100; ++i) weights[(size_t)(uniform(rng) * N)] = 0.f;
            }

            // Create alias table.
            auto aliasTable = AliasTable::create(weights, rng);
            EXPECT(aliasTable != nullptr);

            // Compute weight sum.
            double weightSum = 0.0;
            for (const auto& weight : weights) weightSum += weight;

            EXPECT_EQ(aliasTable->getCount(), weights.size());
            EXPECT_EQ(aliasTable->getWeightSum(), weightSum);

            // Test sampling the alias table.
            {
                const uint32_t samplesPerWeight = 10000;
                uint32_t resultCount = N * samplesPerWeight;
                uint32_t randomCount = resultCount * 2;

                // Create uniform random numbers as input.
                std::vector<float> random(randomCount);
                std::generate(random.begin(), random.end(), [&uniform, &rng]() { return uniform(rng); });

                // Setup and run GPU test.
                ctx.createProgram("Tests/Sampling/AliasTableTests.cs.slang", "testAliasTableSample");
                ctx.allocateStructuredBuffer("sampleResult", resultCount);
                ctx.allocateStructuredBuffer("random", randomCount, random.data());
                aliasTable->setShaderData(ctx["CB"]["aliasTable"]);
                ctx["CB"]["resultCount"] = resultCount;
                ctx.runProgram(resultCount);

                // Build histogram.
                std::vector<uint32_t> histogram(N, 0);
                const uint32_t* result = ctx.mapBuffer<const uint32_t>("sampleResult");
                for (uint32_t i = 0; i < resultCount; ++i)
                {
                    uint32_t item = result[i];
                    EXPECT(item >= 0u && item < N);
                    histogram[item]++;
                }
                ctx.unmapBuffer("sampleResult");

                // Verify histogram using a chi-square test.
                std::vector<double> expFrequencies(N);
                std::vector<double> obsFrequencies(N);
                for (uint32_t i = 0; i < N; ++i)
                {
                    expFrequencies[i] = (weights[i] / weightSum) * N * samplesPerWeight;
                    obsFrequencies[i] = (double)histogram[i];
                }

                // Special case for N == 1
                if (N == 1)
                {
                    EXPECT(histogram[0] == samplesPerWeight);
                }
                else
                {
                    const auto& [success, report] = hypothesis::chi2_test(N, obsFrequencies.data(), expFrequencies.data(), N * samplesPerWeight, 5, 0.1);
                    if (!success) std::cout << report << std::endl;
                    EXPECT(success);
                }
            }

            // Test getting weights.
            {
                uint32_t resultCount = N;

                // Setup and run GPU test.
                ctx.createProgram("Tests/Sampling/AliasTableTests.cs.slang", "testAliasTableWeight");
                ctx.allocateStructuredBuffer("weightResult", resultCount);
                aliasTable->setShaderData(ctx["CB"]["aliasTable"]);
                ctx["CB"]["resultCount"] = resultCount;
                ctx.runProgram(resultCount);

                // Verify weights.
                const float* weightResult = ctx.mapBuffer<const float>("weightResult");
                for (uint32_t i = 0; i < resultCount; ++i)
                {
                    EXPECT_EQ(weightResult[i], weights[i]);
                }
                ctx.unmapBuffer("weightResult");
            }
        }
    }

    GPU_TEST(AliasTable)
    {
        testAliasTable(ctx, 1, { 1.f });
        testAliasTable(ctx, 2, { 1.f, 2.f });
        testAliasTable(ctx, 100);
        testAliasTable(ctx, 1000);
    }
}
