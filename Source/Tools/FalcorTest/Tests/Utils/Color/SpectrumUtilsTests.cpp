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
#include "Utils/Color/SpectrumUtils.h"
#include <random>

namespace Falcor
{
    namespace
    {
        const float kTestMinWavelength = 300.f;
        const float kTestMaxWavelength = 900.f;
    }

    GPU_TEST(WavelengthToXYZ)
    {
        std::mt19937 rng;
        auto dist = std::uniform_real_distribution<float>();
        auto u = [&]() { return dist(rng); };

        const uint32_t n = 20000;
        std::vector<float> wavelengths(n);

        for (uint32_t i = 0; i < n; i++)
        {
            float w = ((float)i + u()) / n;
            wavelengths[i] = kTestMinWavelength + w * (kTestMaxWavelength - kTestMinWavelength);
        }

        // Run GPU test.
        ctx.createProgram("Tests/Utils/Color/SpectrumUtilsTests.cs.slang", "testWavelengthToXYZ");
        ctx.allocateStructuredBuffer("result", n);
        ctx.allocateStructuredBuffer("wavelengths", n, wavelengths.data());
        ctx["CB"]["n"] = n;
        ctx.runProgram(uint3(n, 1, 1));

        // Verify results.
        float3 maxSqrError = {};
        const float3* result = ctx.mapBuffer<const float3>("result");
        for (uint32_t i = 0; i < n; i++)
        {
            float lambda = wavelengths[i];
            float3 res = result[i];
            float3 ref = SpectrumUtils::wavelengthToXYZ_CIE1931(wavelengths[i]);

            EXPECT_GE(res.x, 0.f);
            EXPECT_GE(res.y, 0.f);
            EXPECT_GE(res.z, 0.f);

            float3 e = ref - res;
            maxSqrError = glm::max(maxSqrError, e * e);
        }
        ctx.unmapBuffer("result");

        EXPECT_LE(maxSqrError.x, 2.0e-4f);
        EXPECT_LE(maxSqrError.y, 6.6e-5f);
        EXPECT_LE(maxSqrError.z, 5.2e-4f);
    }
}
