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
#include <vector>

// The perfect hash tests are disabled by default as they take a really long time to run.
// We test a subset of the space instead. The full tests are useful to re-run if the hash is modified.
// Running the full GPU test may require increasing the TDR delay.
//#define RUN_PERFECT_HASH_TESTS

namespace Falcor
{
    namespace
    {
        /** Jenkins hash. This should match HashUtils.slang.
        */
        uint32_t jenkinsHash(uint32_t a)
        {
            a = (a + 0x7ed55d16) + (a << 12);
            a = (a ^ 0xc761c23c) ^ (a >> 19);
            a = (a + 0x165667b1) + (a << 5);
            a = (a + 0xd3a2646c) ^ (a << 9);
            a = (a + 0xfd7046c5) + (a << 3);
            a = (a ^ 0xb55a4f09) ^ (a >> 16);
            return a;
        }
    }

    GPU_TEST(JenkinsHash_CompareToCPU)
    {
        // Allocate results buffer (64k dwords).
        Buffer::SharedPtr pResultBuffer = Buffer::createTyped<uint32_t>(1 << 16, ResourceBindFlags::UnorderedAccess);
        ctx.getRenderContext()->clearUAV(pResultBuffer->getUAV().get(), uint4(0));

        // Setup and run GPU test.
        ctx.createProgram("Tests/Utils/HashUtilsTests.cs.slang", "testJenkinsHash");
        ctx["result"] = pResultBuffer;
        ctx.runProgram(1 << 16, 1, 1);

        // Verify that the generated hashes match the CPU version.
        const uint32_t* result = (const uint32_t*)pResultBuffer->map(Buffer::MapType::Read);
        FALCOR_ASSERT(result);
        for (uint32_t i = 0; i < pResultBuffer->getElementCount(); i++)
        {
            EXPECT_EQ(result[i], jenkinsHash(i)) << "i = " << i;
        }
        pResultBuffer->unmap();
    }

#ifdef RUN_PERFECT_HASH_TESTS
    CPU_TEST(JenkinsHash_PerfectHashCPU)
#else
    CPU_TEST(JenkinsHash_PerfectHashCPU, "Disabled for performance reasons")
#endif
    {
        std::vector<uint32_t> result(1 << 27, 0);
        for (uint64_t i = 0; i < (1ull << 32); i++)
        {
            uint32_t h = jenkinsHash((uint32_t)i);
            result[h >> 5] |= 1u << (h & 0x1f);
        }
        for (size_t i = 0; i < result.size(); i++)
        {
            EXPECT_EQ(result[i], 0xffffffff) << "i = " << i;
        }
    }

#ifdef RUN_PERFECT_HASH_TESTS
    GPU_TEST(JenkinsHash_PerfectHashGPU)
#else
    GPU_TEST(JenkinsHash_PerfectHashGPU, "Disabled for performance reasons")
#endif
    {
        // Allocate results buffer (2^27 dwords).
        Buffer::SharedPtr pResultBuffer = Buffer::createTyped<uint32_t>(1 << 27, ResourceBindFlags::UnorderedAccess);
        ctx.getRenderContext()->clearUAV(pResultBuffer->getUAV().get(), uint4(0));

        // Setup and run GPU test.
        ctx.createProgram("Tests/Utils/HashUtilsTests.cs.slang", "testJenkinsHash_PerfectHash");
        ctx["result"] = pResultBuffer;
        ctx.runProgram(1 << 16, 1 << 16, 1);

        // Verify that all possible 32-bit hashes has occured (all bits set).
        const uint32_t* result = (const uint32_t*)pResultBuffer->map(Buffer::MapType::Read);
        FALCOR_ASSERT(result);
        for (uint32_t i = 0; i < pResultBuffer->getElementCount(); i++)
        {
            EXPECT_EQ(result[i], 0xffffffff) << "i = " << i;
        }
        pResultBuffer->unmap();
    }
}
