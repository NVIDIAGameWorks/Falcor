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
#include <cmath>

namespace Falcor
{
GPU_TEST(SphericalHarmonics_Normalization)
{
    const std::vector<float3> directions = {
        float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f), float3(0.f, 0.f, 1.f), normalize(float3(1.f, 2.f, 3.f))
    };
    constexpr uint32_t basisCount = 16;
    ctx.createProgram("Tests/Utils/SphericalHarmonicsTests.cs.slang", "testSphericalHarmonics");
    ctx.allocateStructuredBuffer("directions", (uint32_t)directions.size(), directions.data());
    ctx.allocateStructuredBuffer("result", (uint32_t)directions.size() * basisCount);
    ctx.runProgram((uint32_t)directions.size() * basisCount);

    const auto result = ctx.readBuffer<float>("result");
    const float pi = std::acos(-1.f);
    constexpr float epsilon = 1e-5f;
    for (uint32_t sample = 0; sample < directions.size(); ++sample)
    {
        const float y00 = result[sample * basisCount];
        const float expectedY00 = 1.f / std::sqrt(4.f * pi);
        EXPECT_GE(y00, expectedY00 - epsilon) << "sample = " << sample;
        EXPECT_LE(y00, expectedY00 + epsilon) << "sample = " << sample;

        // Addition theorem for an orthonormal real SH basis on the unit sphere.
        for (uint32_t degree = 0; degree < 4; ++degree)
        {
            float sum = 0.f;
            for (uint32_t index = degree * degree; index < (degree + 1) * (degree + 1); ++index)
            {
                const float value = result[sample * basisCount + index];
                sum += value * value;
            }
            const float expected = (2.f * degree + 1.f) / (4.f * pi);
            EXPECT_GE(sum, expected - epsilon) << "sample = " << sample << ", degree = " << degree;
            EXPECT_LE(sum, expected + epsilon) << "sample = " << sample << ", degree = " << degree;
        }
    }
}
} // namespace Falcor
