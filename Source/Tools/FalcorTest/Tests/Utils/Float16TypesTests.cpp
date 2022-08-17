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
#include "Utils/Math/Float16.h"
#include <fstd/bit.h> // TODO C++20: Replace with <bit>
#include <random>

namespace Falcor
{
    namespace
    {
        std::mt19937 rng;

        template<size_t N>
        void testVector(CPUUnitTestContext& ctx)
        {
            using floatN = glm::vec<N, float, glm::defaultp>;
            using float16_tN = tfloat16_vec<N>;

            std::uniform_real_distribution<float> dist(-65504.f, 65504.f); // Numerical range of float16
            auto u = [&]() { return dist(rng); };

            // Test construction from scalar.
            {
                float16_t s = float16_t(u());
                float16_tN v = float16_tN(s);
                for (auto i = 0; i < N; i++) EXPECT_EQ((float)v[i], (float)s);
            }

            // Test construction from float vector.
            floatN f;
            for (auto i = 0; i < N; i++) f[i] = u();

            float16_tN v = float16_tN(f);
            for (auto i = 0; i < N; i++) EXPECT_EQ((float)v[i], (float)((float16_t)f[i]));

            // Test cast to float vector.
            f = (floatN)v;
            for (auto i = 0; i < N; i++) EXPECT_EQ((float)v[i], f[i]);
        }
    }

    CPU_TEST(Float16Vector)
    {
        // Check expected size.
        EXPECT_EQ(sizeof(float16_t2), 4);
        EXPECT_EQ(sizeof(float16_t3), 6);
        EXPECT_EQ(sizeof(float16_t4), 8);

        // Test direct element access.
        float16_t2 a = float16_t2(1.f, 2.f);
        EXPECT_EQ((float)a.x, 1.f);
        EXPECT_EQ((float)a.y, 2.f);

        float16_t3 b = float16_t3(1.f, 2.f, 3.f);
        EXPECT_EQ((float)b.x, 1.f);
        EXPECT_EQ((float)b.y, 2.f);
        EXPECT_EQ((float)b.z, 3.f);

        float16_t4 c = float16_t4(1.f, 2.f, 3.f, 4.f);
        EXPECT_EQ((float)c.x, 1.f);
        EXPECT_EQ((float)c.y, 2.f);
        EXPECT_EQ((float)c.z, 3.f);
        EXPECT_EQ((float)c.w, 4.f);

        // More extensive testing of vectors with 2-4 components.
        for (size_t i = 0; i < 1000; i++)
        {
            testVector<2>(ctx);
            testVector<3>(ctx);
            testVector<4>(ctx);
        }
    }

    CPU_TEST(Float16Scalar)
    {
        // Check expected size.
        EXPECT_EQ(sizeof(float16_t), 2);

        // Test cast to float for all bit patterns.
        for (uint32_t bits = 0; bits < 0x10000; bits++)
        {
            const float16_t v = fstd::bit_cast<float16_t>((uint16_t)bits);
            const float f = (float)v;

            uint s = bits >> 15;
            uint e = (bits >> 10) & 0x1f;
            uint m = bits & 0x3ff;
            float sign = s == 0 ? 1.f : -1.f;

            if (e == 0) // denorm
            {
                if (m == 0) // +-zero
                {
                    uint32_t expected = fstd::bit_cast<uint32_t>(s == 0 ? 0.f : -0.f);
                    EXPECT_EQ(f, 0.f);
                    EXPECT_EQ(fstd::bit_cast<uint32_t>(f), expected);
                }
                else
                {
                    float expected = sign * 0x1p-14f * ((float)m / 1024.f);
                    EXPECT_EQ(f, expected);
                }
            }
            else if (e == 0x1f) // inf/nan
            {
                if (m == 0) // +-inf
                {
                    uint32_t expected = fstd::bit_cast<uint32_t>(s == 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
                    EXPECT(std::isinf(f));
                    EXPECT_EQ(fstd::bit_cast<uint32_t>(f), expected);
                }
                else // nan
                {
                    EXPECT(std::isnan(f));
                }
            }
            else // normalized
            {
                float expected = sign * ((float)(1u << e) * 0x1p-15f) * (1.f + (float)m / 1024.f);
                EXPECT_EQ(f, expected);
            }
        }

        // Test cast to/from float for all bit patterns.
        for (uint32_t bits = 0; bits < 0x10000; bits++)
        {
            float16_t expected = fstd::bit_cast<float16_t>((uint16_t)bits);
            float f = (float)expected;
            float16_t result = (float16_t)f;

            EXPECT_EQ(fstd::bit_cast<uint16_t>(result), fstd::bit_cast<uint16_t>(expected));
        }
    }
}
