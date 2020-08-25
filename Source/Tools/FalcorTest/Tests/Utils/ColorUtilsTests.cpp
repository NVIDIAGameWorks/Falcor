/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Color/ColorUtils.h"
#include <random>

namespace Falcor
{
    // Some shared test utils.
    namespace
    {
        const float kMaxError = 1e-5f;

        auto maxAbsDiff = [](float3 a, float3 b) -> float
        {
            float3 d = abs(a - b);
            return std::max(std::max(d.x, d.y), d.z);
        };
    }

    CPU_TEST(ColorTransforms)
    {
        const uint32_t n = 10000;

        // Prepare for tests.
        std::default_random_engine rng;
        auto dist = std::uniform_real_distribution<float>();
        auto u = [&]() -> float { return dist(rng); };

        const glm::mat3 LMS_CAT02 = kColorTransform_LMStoXYZ_CAT02 * kColorTransform_XYZtoLMS_CAT02;
        const glm::mat3 LMS_Bradford = kColorTransform_LMStoXYZ_Bradford * kColorTransform_XYZtoLMS_Bradford;

        // Run test code that transforms random colors between different spaces.
        for (uint32_t i = 0; i < n; i++)
        {
            const float3 c = { u(), u(), u() };

            // Test RGB<->XYZ by transforming random colors back and forth.
            float3 res1 = XYZtoRGB_Rec709(RGBtoXYZ_Rec709(c));
            EXPECT_LE(maxAbsDiff(res1, c), kMaxError);

            // Test XYZ<->LMS using the CAT02 transform.
            float3 res2 = LMS_CAT02 * c;
            EXPECT_LE(maxAbsDiff(res2, c), kMaxError);

            // Test XYZ<->LMS using the Bradford transform
            float3 res3 = LMS_Bradford * c;
            EXPECT_LE(maxAbsDiff(res3, c), kMaxError);
        }
    }

    CPU_TEST(WhiteBalance)
    {
        const float3 white = { 1, 1, 1 };

        // The white point should be 6500K. Verify that we get pure white back.
        float3 wbWhite = calculateWhiteBalanceTransformRGB_Rec709(6500.f) * white;
        EXPECT_LE(maxAbsDiff(wbWhite, white), kMaxError);

        // Test white balance transform at a few different color temperatures.
        // This is a  very crude test just to see we're not entirely off.
        //
        // Color correcting white @ 6500K to these targets should yield:
        // - Cloudy (7000K) => yellowish tint (r > g > b)
        // - Sunny  (5500K) => blueish tint (r < g < b)
        // - Indoor (3000K) => stronger bluish tint (r < g < b)
        float3 wbCloudy = calculateWhiteBalanceTransformRGB_Rec709(7000.f) * white;
        float3 wbSunny = calculateWhiteBalanceTransformRGB_Rec709(5500.f) * white;
        float3 wbIndoor = calculateWhiteBalanceTransformRGB_Rec709(3000.f) * white;

        EXPECT_GE(wbCloudy.r, wbCloudy.g);
        EXPECT_GE(wbCloudy.g, wbCloudy.b);

        EXPECT_LE(wbSunny.r, wbSunny.g);
        EXPECT_LE(wbSunny.g, wbSunny.b);

        EXPECT_LE(wbIndoor.r, wbIndoor.g);
        EXPECT_LE(wbIndoor.g, wbIndoor.b);

        // Normalize the returned RGB to max 1.0 to be able to compare the scale.
        wbSunny /= wbSunny.b;
        wbIndoor /= wbIndoor.b;

        EXPECT_LE(wbIndoor.r, wbSunny.r);
        EXPECT_LE(wbIndoor.g, wbSunny.g);
    }
}
