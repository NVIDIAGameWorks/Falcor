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
#include "Utils/Math/ScalarMath.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace Falcor
{
GPU_TEST(MathHelpers_SphericalCoordinates)
{
    ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testSphericalCoordinates");
    constexpr int32_t n = 1024 * 1024;
    ctx.allocateStructuredBuffer("result", n);
    // The shader runs threadgroups of 1024 threads.
    ctx.runProgram(n);

    // The shader generates a bunch of random vectors, converts them to
    // spherical coordinates and back, and computes the dot product with
    // the original vector.  Here, we'll check that the dot product is
    // pretty close to one.
    std::vector<float> r = ctx.readBuffer<float>("result");
    for (int32_t i = 0; i < n; ++i)
    {
        EXPECT_GT(r[i], .999f) << "i = " << i;
        EXPECT_LT(r[i], 1.001f) << "i = " << i;
    }
}

GPU_TEST(MathHelpers_SphericalCoordinatesRad)
{
    ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testSphericalCoordinatesRad");
    constexpr int32_t n = 1024 * 1024;
    ctx.allocateStructuredBuffer("result", n);
    // The shader runs threadgroups of 1024 threads.
    ctx.runProgram(n);

    // The shader generates a bunch of random vectors, converts them to
    // spherical coordinates and back, and computes the dot product with
    // the original vector.  Here, we'll check that the dot product is
    // pretty close to one.
    std::vector<float> r = ctx.readBuffer<float>("result");
    for (int32_t i = 0; i < n; ++i)
    {
        EXPECT_GT(r[i], .999f) << "i = " << i;
        EXPECT_LT(r[i], 1.001f) << "i = " << i;
    }
}

GPU_TEST(MathHelpers_ErrorFunction)
{
    // Test the approximate implementation of `erf` against
    // the C++ standard library.
    ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testErrorFunction");
    constexpr int32_t n = 25;
    std::vector<float> input(n);
    std::vector<float> ref(n);
    for (int32_t i = 0; i < n; ++i)
    {
        float t = i / (float)(n - 1);
        float x = math::lerp<float>(-5, 5, t);
        input[i] = x;
        ref[i] = std::erf(x);
    }

    ctx.allocateStructuredBuffer("result", n);
    ctx.allocateStructuredBuffer("input", (uint32_t)input.size(), input.data());

    ctx.runProgram(n);

    std::vector<float> r = ctx.readBuffer<float>("result");
    float epsilon = 1e-6f;
    for (int32_t i = 0; i < n; ++i)
    {
        EXPECT_GE(r[i], ref[i] - epsilon) << "i = " << i;
        EXPECT_LE(r[i], ref[i] + epsilon) << "i = " << i;
    }
}

GPU_TEST(MathHelpers_InverseErrorFunction)
{
    // The C++ standard library does not have a reference for `erfinv`,
    // but we can test erf(erfinv(x)) = x instead.
    ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testInverseErrorFunction");
    constexpr int32_t n = 25;
    std::vector<float> input(n);
    for (int32_t i = 0; i < n; ++i)
    {
        float t = i / (float)(n - 1);
        input[i] = math::lerp<float>(-1, 1, t);
    }

    ctx.allocateStructuredBuffer("result", n);
    ctx.allocateStructuredBuffer("input", (uint32_t)input.size(), input.data());

    ctx.runProgram(n);

    std::vector<float> r = ctx.readBuffer<float>("result");
    float epsilon = 1e-6f;
    for (int32_t i = 0; i < n; ++i)
    {
        EXPECT_GE(std::erf(r[i]), input[i] - epsilon) << "i = " << i;
        EXPECT_LE(std::erf(r[i]), input[i] + epsilon) << "i = " << i;
    }
}
} // namespace Falcor
