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
#include <random>

/** GPU tests for the various pseusorandom number generators.

    We create a number of parallel instances of the PRNGs, seeded randomly,
    and let each instance create a sequence of random numbers up to a certain
    dimension. The result is compared against CPU reference implementations.
*/

namespace Falcor
{
// Reference implementation of the xoshiro128** algorithm.
// See http://xoshiro.di.unimi.it/xoshiro128starstar.c
struct xoshiro128starstar
{
    static inline uint32_t rotl(const uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

    uint32_t s[4];

    uint32_t next()
    {
        const uint32_t result_starstar = rotl(s[0] * 5, 7) * 9;

        const uint32_t t = s[1] << 9;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 11);

        return result_starstar;
    }
};

// Reference implementation of the SplitMix64 algorithm.
// See http://xoshiro.di.unimi.it/splitmix64.c
struct splitmix64
{
    uint64_t x;

    uint64_t next()
    {
        uint64_t z = (x += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
};

// Reference implementation of the LCG from Numerical Recipes.
// See https://en.wikipedia.org/wiki/Linear_congruential_generator
struct lcg
{
    uint32_t state;

    uint32_t next()
    {
        const uint32_t a = 1664525;
        const uint32_t c = 1013904223;
        state = a * state + c;
        return state;
    }
};

// Shared test utils.
namespace
{
const char kShaderFile[] = "Tests/Sampling/PseudorandomTests.cs.slang";

const uint32_t kInstances = 256;
const uint32_t kDimensions = 64;

ref<Buffer> createSeed(ref<Device> pDevice, size_t elements, std::vector<uint32_t>& seed)
{
    // Initialize buffer of random seed data.
    seed.resize(elements);
    std::mt19937 rng;
    for (auto& it : seed)
        it = rng();

    // Upload seeds to the GPU.
    ref<Buffer> pSeedBuf =
        pDevice->createBuffer(seed.size() * sizeof(seed[0]), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, seed.data());
    FALCOR_ASSERT(pSeedBuf);
    return pSeedBuf;
}
} // namespace

/** GPU test for Xoshiro pseudorandom number generator.
 */
GPU_TEST(XoshiroPRNG)
{
    // Create random seed (128 bits per instance).
    std::vector<uint32_t> seed;
    auto pSeedBuf = createSeed(ctx.getDevice(), kInstances * 4, seed);

    // Setup and run GPU test.
    ctx.createProgram(kShaderFile, "testXoshiro");
    ctx.allocateStructuredBuffer("result", kInstances * kDimensions);
    ctx["seed"] = pSeedBuf;
    ctx.runProgram(kInstances);

    // Compare result against reference implementation.
    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kInstances; i++)
    {
        xoshiro128starstar rng;
        // Set seed.
        rng.s[0] = seed[i * 4 + 0];
        rng.s[1] = seed[i * 4 + 1];
        rng.s[2] = seed[i * 4 + 2];
        rng.s[3] = seed[i * 4 + 3];

        for (uint32_t j = 0; j < kDimensions; j++)
        {
            const uint32_t ref = rng.next();
            const uint32_t res = result[j * kInstances + i];
            EXPECT_EQ(ref, res) << "instance = " << i << " dimension = " << j;
        }
    }
}

/** GPU test for SplitMix64 pseudorandom number generator.
 */
GPU_TEST(SplitMixPRNG)
{
    // Create random seed (64 bits per instance).
    std::vector<uint32_t> seed;
    auto pSeedBuf = createSeed(ctx.getDevice(), kInstances * 2, seed);

    // Setup and run GPU test. Note it requires SM 6.0 or higher.
    ctx.createProgram(kShaderFile, "testSplitMix");
    ctx.allocateStructuredBuffer("result64", kInstances * kDimensions);
    ctx["seed"] = pSeedBuf;
    ctx.runProgram(kInstances);

    // Compare result against reference implementation.
    std::vector<uint64_t> result = ctx.readBuffer<uint64_t>("result64");
    for (uint32_t i = 0; i < kInstances; i++)
    {
        splitmix64 rng;
        // Set seed.
        rng.x = (uint64_t(seed[i * 2 + 1]) << 32) | uint64_t(seed[i * 2 + 0]);

        for (uint32_t j = 0; j < kDimensions; j++)
        {
            const uint64_t ref = rng.next();
            const uint64_t res = result[j * kInstances + i];
            EXPECT_EQ(ref, res) << "instance = " << i << " dimension = " << j;
        }
    }
}

/** GPU test for LCG pseudorandom number generator.
 */
GPU_TEST(LCGPRNG)
{
    // Create random seed (32 bits per instance).
    std::vector<uint32_t> seed;
    auto pSeedBuf = createSeed(ctx.getDevice(), kInstances, seed);

    // Setup and run GPU test.
    ctx.createProgram(kShaderFile, "testLCG");
    ctx.allocateStructuredBuffer("result", kInstances * kDimensions);
    ctx["seed"] = pSeedBuf;
    ctx.runProgram(kInstances);

    // Compare result against reference implementation.
    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kInstances; i++)
    {
        lcg rng;
        // Set seed.
        rng.state = seed[i];

        for (uint32_t j = 0; j < kDimensions; j++)
        {
            const uint32_t ref = rng.next();
            const uint32_t res = result[j * kInstances + i];
            EXPECT_EQ(ref, res) << "instance = " << i << " dimension = " << j;
        }
    }
}
} // namespace Falcor
