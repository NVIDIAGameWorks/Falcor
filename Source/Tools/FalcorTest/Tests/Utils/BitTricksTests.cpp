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
#include <random>

namespace Falcor
{
    namespace
    {
        // Reference function to interleave m bits from x and y.
        // The result is a bit sequence: 0 ... 0 ym xm ... y1 x1 y0 x0.
        uint32_t referenceBitInterleave(uint32_t x, uint32_t y, uint32_t m)
        {
            uint32_t result = 0;
            for (uint32_t i = 0; i < m; i++)
            {
                result |= ((x >> i) & 1) << (2 * i);
                result |= ((y >> i) & 1) << (2 * i + 1);
            }
            return result;
        }
    }

    GPU_TEST(BitInterleave)
    {
        const uint32_t tests = 5;
        const uint32_t n = 1 << 16;

        // First test the reference function itself against a manually constructed example.
        EXPECT_EQ(referenceBitInterleave(0xe38e, 0xbe8b, 16), 0xdeadc0de);
        EXPECT_EQ(referenceBitInterleave(0xe38e, 0xbe8b, 12), 0x00adc0de);

        // Create a buffer of random bits to use as test data.
        std::vector<uint32_t> testData(n);
        std::mt19937 r;
        for (auto& it : testData) it = r();

        Buffer::SharedPtr pTestDataBuffer = Buffer::create(n * sizeof(uint32_t), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, testData.data());

        // Setup and run GPU test.
        ctx.createProgram("Tests/Utils/BitTricksTests.cs.slang", "testBitInterleave");
        ctx.allocateStructuredBuffer("result", n * tests);
        ctx["testData"] = pTestDataBuffer;
        ctx.runProgram(n);

        // Verify results.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        for (uint32_t i = 0; i < n; i++)
        {
            const uint32_t bits = testData[i];
            const uint32_t interleavedBits = referenceBitInterleave(bits, bits >> 16, 16);

            // Check result of interleave functions.
            EXPECT_EQ(result[tests * i + 0], interleavedBits);
            EXPECT_EQ(result[tests * i + 1], (interleavedBits & 0xffff));

            // Check result of de-interleave functions.
            EXPECT_EQ(result[tests * i + 2], (bits & 0x00ff00ff));
            EXPECT_EQ(result[tests * i + 3], (bits & 0x000f000f));
            EXPECT_EQ(result[tests * i + 4], (bits & 0x0f0f0f0f));
        }
        ctx.unmapBuffer("result");
    }
}
