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

/** GPU tests for the SampleGenerator utility class.
*/

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Tests/Sampling/SampleGeneratorTests.cs.slang";

        // The shader uses the first two dispatch dimensions as spatial seed and the last as instance index.
        // For each sample generator instance, it generates kDimensions samples.
        const uint3 kDispatchDim = { 64, 64, 16 };
        const uint32_t kDimensions = 32;

        /** Estimates the population Pearson correlation between pairs of
            measurements of a random variable stored in an array 'elems'.
            The two values in each pair are separated a distance 'stride'.
            The function iterates over all samples i, measuring correlation
            between sample i and i+stride, so each value may be part of multiple pairs.
            \return Estimated Pearson correlation coefficient in [-1,1], where 0.0 means no correlation.
        */
        double correlation(const float* elems, const size_t numElems, const size_t stride)
        {
            double sum_x = 0.0, sum_y = 0.0;
            double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0;
            size_t n = 0;
            for (size_t i = 0; i + stride < numElems; i++)
            {
                float x = elems[i];
                float y = elems[i + stride];
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_yy += y * y;
                sum_xy += x * y;
                n++;
            }
            FALCOR_ASSERT(n > 0);
            double r_xy = ((double)n * sum_xy - sum_x * sum_y) /
                (std::sqrt(n * sum_xx - sum_x * sum_x) * std::sqrt(n * sum_yy - sum_y * sum_y));
            return r_xy;
        }

        void testSampleGenerator(GPUUnitTestContext& ctx, uint32_t type, const double corrThreshold, bool testInstances)
        {
            // Create sample generator.
            SampleGenerator::SharedPtr pSampleGenerator = SampleGenerator::create(type);

            // Setup GPU test.
            // We defer the creation of the vars until after shader specialization.
            auto defines = pSampleGenerator->getDefines();
            ctx.createProgram(kShaderFile, "test", defines, Shader::CompilerFlags::None, "6_2");

            pSampleGenerator->setShaderData(ctx.vars().getRootVar());

            const size_t numSamples = kDispatchDim.x * kDispatchDim.y * kDispatchDim.z * kDimensions;
            ctx.allocateStructuredBuffer("result", uint32_t(numSamples));
            ctx["CB"]["gDispatchDim"] = kDispatchDim;
            ctx["CB"]["gDimensions"] = kDimensions;

            // Run the test.
            ctx.runProgram(kDispatchDim);

            // Readback results.
            const float* result = ctx.mapBuffer<const float>("result");

            // Check that all samples are in the [0,1) range,
            // and that their mean is roughly 0.5.
            double mean = 0.0;
            for (size_t i = 0; i < numSamples; i++)
            {
                float u = result[i];
                mean += u;
                EXPECT(u >= 0.f && u < 1.f) << u;
            }
            mean /= numSamples;
            EXPECT_GE(mean, 0.499);
            EXPECT_LE(mean, 0.501);

            // Check correlation between adjacent samples along different dimensions in the sample set.
            // This is not really a robust statistical test, but it should detect if something is fundamentally wrong.
            auto corr = [&](size_t stride) -> double
            {
                return std::abs(correlation(result, numSamples, stride));
            };

            // Test nearby dimensions.
            for (size_t i = 1; i <= 8; i++)
            {
                EXPECT_LE(corr(i), corrThreshold) << "i = " << i;
            }

            // Test nearby pixels.
            const size_t xStride = kDimensions;
            const size_t yStride = kDispatchDim.x * kDimensions;
            for (size_t y = 0; y < 4; y++)
            {
                for (size_t x = 0; x < 4; x++)
                {
                    if (x == 0 && y == 0) continue;
                    EXPECT_LE(corr(x * xStride + y * yStride), corrThreshold) << "x = " << x << " y = " << y;
                }
            }

            // Test nearby instances, if they are expected to be uncorrelated.
            if (testInstances)
            {
                const size_t instanceStride = kDispatchDim.x * kDispatchDim.y * kDimensions;
                for (size_t i = 1; i <= 4; i++)
                {
                    EXPECT_LE(corr(i * instanceStride), corrThreshold) << "i = " << i;
                }
            }

            ctx.unmapBuffer("result");
        }
    }

    /** Tests for the different types of sample generators.

        For each one, we specify the maximum allowed absolute correlation between samples.
        The values have been tweaked based on observed correlations at these sample counts.
    */

    GPU_TEST(SampleGenerator_TinyUniform)
    {
        testSampleGenerator(ctx, SAMPLE_GENERATOR_TINY_UNIFORM, 0.0025, true);
    }

    GPU_TEST(SampleGenerator_Uniform)
    {
        testSampleGenerator(ctx, SAMPLE_GENERATOR_UNIFORM, 0.002, true);
    }

}
