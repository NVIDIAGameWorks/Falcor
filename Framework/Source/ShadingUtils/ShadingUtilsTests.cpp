/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "UnitTest.h"

namespace Falcor
{

    // Just check the first four values.
    GPU_TEST(RadicalInverse)
    {
        ctx.createProgram("ShadingUtilsTests.cs.hlsl", "testRadicalInverse");
        ctx.allocateStructuredBuffer("result", 4);
        ctx["TestCB"]["resultSize"] = 4;
        ctx.runProgram();

        const float *s = ctx.mapBuffer<const float>("result");
        EXPECT_EQ(s[0], 0.f);
        EXPECT_EQ(s[1], 0.5f);
        EXPECT_EQ(s[2], 0.25f);
        EXPECT_EQ(s[3], 0.75f);
        ctx.unmapBuffer("result");
    }

    GPU_TEST(Random)
    {
        ctx.createProgram("ShadingUtilsTests.cs.hlsl", "testRand");
        const int32_t n = 4 * 1024 * 1024;
        ctx.allocateStructuredBuffer("result", n);
        ctx["TestCB"]["resultSize"] = n;
        ctx.runProgram();

        // A fairly crude test: bucket the range [0,1] into nBuckets buckets
        // and make sure that all of them have more or less 1/nBuckets of the
        // total values.  This doesn't really test the quality of the PRNG very
        // well, but will at least detect if it's totally borked.
        const float* r = ctx.mapBuffer<const float>("result");
        constexpr int32_t nBuckets = 64;
        int32_t counts[nBuckets] = { 0 };
        for (int32_t i = 0; i < n; ++i)
        {
            EXPECT(r[i] >= 0 && r[i] < 1.f) << r[i];
            ++counts[int32_t(r[i] * nBuckets)];
        }
        ctx.unmapBuffer("result");

        for (int32_t i = 0; i < nBuckets; ++i)
        {
            EXPECT_GT(counts[i], .98 * n / nBuckets);
            EXPECT_LT(counts[i], 1.02 * n / nBuckets);
        }
    }

    GPU_TEST(SphericalCoordinates)
    {
        ctx.createProgram("ShadingUtilsTests.cs.hlsl", "testSphericalCoordinates");
        constexpr int32_t n = 1024 * 1024;
        ctx.allocateStructuredBuffer("result", n);
        ctx["TestCB"]["resultSize"] = n;
        // The shader runs threadgroups of 1024 threads.
        ctx.runProgram(n);

        // The shader generates a bunch of random vectors, converts them to
        // spherical coordinates and back, and computes the dot product with
        // the original vector.  Here, we'll check that the dot product is
        // pretty close to one.
        const float* r = ctx.mapBuffer<const float>("result");
        for (int32_t i = 0; i < n; ++i)
        {
            EXPECT_GT(r[i], .999f) << "i = " << i;
            EXPECT_LT(r[i], 1.001f) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }

}  // namespace Falcor
