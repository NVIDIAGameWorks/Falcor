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

/** GPU tests for the various pseusorandom number generators.

    We create a number of parallel instances of the PRNGs, seeded randomly,
    and let each instance create a sequence of random numbers up to a certain
    dimension. The result is compared against CPU reference implementations.
*/

namespace Falcor
{
    // Reference implementation of the xoshiro128** algorithm.
    namespace xoshiro128starstar
    {
        // This sources is unmodified from the author's website.
        // We import it into a namespace to be able to use it from the test function.
        #include "xoshiro/xoshiro128starstar.c"
    }

    // Reference implementation of the SplitMix64 algorithm.
    namespace splitmix64
    {
        // This sources is unmodified from the author's website.
        // We import it into a namespace to be able to use it from the test function.
        #include "xoshiro/splitmix64.c"
    }

    // Reference implementation of the LCG from Numerical Recipes.
    // See https://en.wikipedia.org/wiki/Linear_congruential_generator
    namespace lcg
    {
        static uint32_t state;
        uint32_t next()
        {
            const uint32_t a = 1664525;
            const uint32_t c = 1013904223;
            state = a * state + c;
            return state;
        }
    }

    // Shared test utils.
    namespace
    {
        const char kShaderFile[] = "Tests/Sampling/PseudorandomTests.cs.slang";

        const uint32_t kInstances = 256;
        const uint32_t kDimensions = 64;

        Buffer::SharedPtr createSeed(size_t elements, std::vector<uint32_t>& seed)
        {
            // Initialize buffer of random seed data.
            seed.resize(elements);
            std::mt19937 rng;
            for (auto& it : seed) it = rng();

            // Upload seeds to the GPU.
            Buffer::SharedPtr pSeedBuf = Buffer::create(seed.size() * sizeof(seed[0]), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, seed.data());
            assert(pSeedBuf);
            return pSeedBuf;
        }
    }

    /** GPU test for Xoshiro pseudorandom number generator.
    */
    GPU_TEST(XoshiroPRNG)
    {
        // Create random seed (128 bits per instance).
        std::vector<uint32_t> seed;
        auto pSeedBuf = createSeed(kInstances * 4, seed);

        // Setup and run GPU test.
        ctx.createProgram(kShaderFile, "testXoshiro");
        ctx.allocateStructuredBuffer("result", kInstances * kDimensions);
        ctx["seed"] = pSeedBuf;
        ctx.runProgram(kInstances);

        // Compare result against reference implementation.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        for (uint32_t i = 0; i < kInstances; i++)
        {
            // Set seed.
            xoshiro128starstar::s[0] = seed[i * 4 + 0];
            xoshiro128starstar::s[1] = seed[i * 4 + 1];
            xoshiro128starstar::s[2] = seed[i * 4 + 2];
            xoshiro128starstar::s[3] = seed[i * 4 + 3];

            for (uint32_t j = 0; j < kDimensions; j++)
            {
                const uint32_t ref = xoshiro128starstar::next();
                const uint32_t res = result[j * kInstances + i];
                EXPECT_EQ(ref, res) << "instance = " << i << " dimension = " << j;
            }
        }
        ctx.unmapBuffer("result");
    }

    /** GPU test for SplitMix64 pseudorandom number generator.
    */
    GPU_TEST(SplitMixPRNG)
    {
        // Create random seed (64 bits per instance).
        std::vector<uint32_t> seed;
        auto pSeedBuf = createSeed(kInstances * 2, seed);

        // Setup and run GPU test. Note it requires SM 6.0 or higher.
        ctx.createProgram(kShaderFile, "testSplitMix", Program::DefineList(), Shader::CompilerFlags::None);
        ctx.allocateStructuredBuffer("result64", kInstances * kDimensions);
        ctx["seed"] = pSeedBuf;
        ctx.runProgram(kInstances);

        // Compare result against reference implementation.
        const uint64_t* result = ctx.mapBuffer<const uint64_t>("result64");
        for (uint32_t i = 0; i < kInstances; i++)
        {
            // Set seed.
            splitmix64::x = (uint64_t(seed[i * 2 + 1]) << 32) | uint64_t(seed[i * 2 + 0]);

            for (uint32_t j = 0; j < kDimensions; j++)
            {
                const uint64_t ref = splitmix64::next();
                const uint64_t res = result[j * kInstances + i];
                EXPECT_EQ(ref, res) << "instance = " << i << " dimension = " << j;
            }
        }
        ctx.unmapBuffer("result64");
    }

    /** GPU test for LCG pseudorandom number generator.
    */
    GPU_TEST(LCGPRNG)
    {
        // Create random seed (32 bits per instance).
        std::vector<uint32_t> seed;
        auto pSeedBuf = createSeed(kInstances, seed);

        // Setup and run GPU test.
        ctx.createProgram(kShaderFile, "testLCG");
        ctx.allocateStructuredBuffer("result", kInstances * kDimensions);
        ctx["seed"] = pSeedBuf;
        ctx.runProgram(kInstances);

        // Compare result against reference implementation.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        for (uint32_t i = 0; i < kInstances; i++)
        {
            // Set seed.
            lcg::state = seed[i];

            for (uint32_t j = 0; j < kDimensions; j++)
            {
                const uint32_t ref = lcg::next();
                const uint32_t res = result[j * kInstances + i];
                EXPECT_EQ(ref, res) << "instance = " << i << " dimension = " << j;
            }
        }
        ctx.unmapBuffer("result");
    }
}
