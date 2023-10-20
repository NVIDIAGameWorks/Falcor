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
#include "Utils/Algorithm/PrefixSum.h"
#include <random>

namespace Falcor
{
namespace
{
uint32_t prefixSumRef(std::vector<uint32_t>& elems)
{
    // Perform exclusive scan. Return sum of all elements.
    uint32_t sum = 0;
    for (auto& it : elems)
    {
        uint32_t tmp = it;
        it = sum;
        sum += tmp;
    }
    return sum;
}

void testPrefixSum(GPUUnitTestContext& ctx, PrefixSum& prefixSum, uint32_t numElems)
{
    ref<Device> pDevice = ctx.getDevice();

    // Create a buffer of random data to use as test data.
    // We make sure the total sum fits in 32 bits.
    FALCOR_ASSERT(numElems > 0);
    const uint32_t maxVal = std::numeric_limits<uint32_t>::max() / numElems;
    std::vector<uint32_t> testData(numElems);
    std::mt19937 r;
    for (auto& it : testData)
        it = r() % maxVal;

    ref<Buffer> pTestDataBuffer =
        pDevice->createBuffer(numElems * sizeof(uint32_t), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, testData.data());

    // Allocate buffer for the total sum on the GPU.
    uint32_t nullValue = 0;
    ref<Buffer> pSumBuffer =
        pDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &nullValue);

    // Execute prefix sum on the GPU.
    uint32_t sum = 0;
    prefixSum.execute(ctx.getRenderContext(), pTestDataBuffer, numElems, &sum, pSumBuffer, 0);

    // Compute prefix sum on the CPU for comparison.
    const uint32_t refSum = prefixSumRef(testData);

    // Compare results.
    EXPECT_EQ(sum, refSum);

    uint32_t resultSum = pSumBuffer->getElement<uint32_t>(0);
    EXPECT_EQ(resultSum, refSum);

    std::vector<uint32_t> result = pTestDataBuffer->getElements<uint32_t>();
    for (uint32_t i = 0; i < numElems; i++)
    {
        EXPECT_EQ(testData[i], result[i]) << "i = " << i;
    }
}
} // namespace

GPU_TEST(PrefixSum)
{
    // Quick test of our reference function.
    std::vector<uint32_t> x({5, 17, 2, 9, 23});
    uint32_t sum = prefixSumRef(x);
    FALCOR_ASSERT(x[0] == 0 && x[1] == 5 && x[2] == 22 && x[3] == 24 && x[4] == 33);
    FALCOR_ASSERT(sum == 56);

    // Create helper class.
    PrefixSum prefixSum(ctx.getDevice());

    // Test prefix sums on varying size buffers.
    testPrefixSum(ctx, prefixSum, 1);
    testPrefixSum(ctx, prefixSum, 27);
    testPrefixSum(ctx, prefixSum, 64);
    testPrefixSum(ctx, prefixSum, 2049);
    testPrefixSum(ctx, prefixSum, 10201);
    testPrefixSum(ctx, prefixSum, 231917);
    testPrefixSum(ctx, prefixSum, 1088921);
    testPrefixSum(ctx, prefixSum, 13912615);
}
} // namespace Falcor
