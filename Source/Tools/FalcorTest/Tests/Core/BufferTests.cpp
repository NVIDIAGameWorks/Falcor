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
#include "Utils/Math/Common.h"

namespace Falcor
{
namespace
{
const uint32_t kIterations = 4;

enum class Type
{
    ByteAddressBuffer = 0,
    TypedBuffer = 1,
    StructuredBuffer = 2,
};

template<Type type>
void testBuffer(GPUUnitTestContext& ctx, uint32_t numElems, uint32_t index = 0, uint32_t count = 0)
{
    static_assert(type == Type::ByteAddressBuffer || type == Type::TypedBuffer || type == Type::StructuredBuffer);

    ref<Device> pDevice = ctx.getDevice();

    numElems = div_round_up(numElems, 256u) * 256u; // Make sure we run full thread groups.

    // Create a data blob for the test.
    // We fill it some numbers that don't overlap with the test buffers values.
    std::vector<uint32_t> blob;
    if (count > 0)
    {
        blob.resize(count);
        for (uint32_t i = 0; i < blob.size(); i++)
            blob[i] = (count - i) * 3;
    }

    // Create clear program.
    DefineList defines = {{"TYPE", std::to_string((uint32_t)type)}};
    ctx.createProgram("Tests/Core/BufferTests.cs.slang", "clearBuffer", defines);

    // Create test buffer.
    ref<Buffer> pBuffer;
    if constexpr (type == Type::ByteAddressBuffer)
        pBuffer = pDevice->createBuffer(numElems * sizeof(uint32_t), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal);
    else if constexpr (type == Type::TypedBuffer)
        pBuffer = pDevice->createTypedBuffer<uint32_t>(numElems, ResourceBindFlags::UnorderedAccess);
    else if constexpr (type == Type::StructuredBuffer)
        pBuffer = pDevice->createStructuredBuffer(ctx.getVars()->getRootVar()["buffer"], numElems, ResourceBindFlags::UnorderedAccess);

    ctx["buffer"] = pBuffer;

    // Run kernel to clear the buffer.
    // We clear explicitly instead of using clearUAV() as the latter is not compatible with RWStructuredBuffer.
    ctx.runProgram(numElems, 1, 1);

    // Create update program.
    ctx.createProgram("Tests/Core/BufferTests.cs.slang", "updateBuffer", defines);
    ctx["buffer"] = pBuffer;

    // Run kernel N times to update the buffer (RMW).
    for (uint32_t i = 0; i < kIterations; i++)
        ctx.runProgram(numElems, 1, 1);

    // Use setBlob() to update part of the buffer from the CPU.
    if (count > 0)
    {
        FALCOR_ASSERT(index + blob.size() <= numElems);
        pBuffer->setBlob(blob.data(), index * sizeof(uint32_t), blob.size() * sizeof(uint32_t));
    }

    // Run kernel N times to update the buffer (RMW).
    for (uint32_t i = 0; i < kIterations; i++)
        ctx.runProgram(numElems, 1, 1);

    // Run kernel to read values in the buffer.
    ctx.createProgram("Tests/Core/BufferTests.cs.slang", "readBuffer", defines);
    ctx.allocateStructuredBuffer("result", numElems);
    ctx["buffer"] = pBuffer;

    ctx.runProgram(numElems, 1, 1);

    // Verify results.
    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < numElems; i++)
    {
        // Each RMW pass adds i+1 to the element at index i.
        // We run kIterations passes, then replace part of the buffer, followed by kIterations more passes.
        uint32_t expected = (i + 1) * kIterations * 2;
        if (i >= index && i < index + blob.size())
            expected = blob[i - index] + (i + 1) * kIterations;
        uint32_t actual = result[i];

        EXPECT_EQ(actual, expected) << "i = " << i << " (numElems = " << numElems << " index = " << index << " count = " << count << ")";
    }
}
} // namespace

GPU_TEST(RawBuffer)
{
    auto testFunc = testBuffer<Type::ByteAddressBuffer>;
    for (uint32_t numElems = 1u << 8; numElems <= (1u << 16); numElems <<= 4)
    {
        testFunc(ctx, numElems, 0, 0);
        testFunc(ctx, numElems, 0, 1);
        testFunc(ctx, numElems, 0, numElems / 2);
        testFunc(ctx, numElems, 1, 1);
        testFunc(ctx, numElems, numElems / 2 + 3, numElems / 4 - 1);
    }
}

GPU_TEST(TypedBuffer)
{
    auto testFunc = testBuffer<Type::TypedBuffer>;
    for (uint32_t numElems = 1u << 8; numElems <= (1u << 16); numElems <<= 4)
    {
        testFunc(ctx, numElems, 0, 0);
        testFunc(ctx, numElems, 0, 1);
        testFunc(ctx, numElems, 0, numElems / 2);
        testFunc(ctx, numElems, 1, 1);
        testFunc(ctx, numElems, numElems / 2 + 3, numElems / 4 - 1);
    }
}

GPU_TEST(StructuredBuffer)
{
    auto testFunc = testBuffer<Type::StructuredBuffer>;
    for (uint32_t numElems = 1u << 8; numElems <= (1u << 16); numElems <<= 4)
    {
        testFunc(ctx, numElems, 0, 0);
        testFunc(ctx, numElems, 0, 1);
        testFunc(ctx, numElems, 0, numElems / 2);
        testFunc(ctx, numElems, 1, 1);
        testFunc(ctx, numElems, numElems / 2 + 3, numElems / 4 - 1);
    }
}

GPU_TEST(BufferUpdate)
{
    const uint4 a = {1, 2, 3, 4};
    const uint4 b = {5, 6, 7, 8};

    auto pBuffer = ctx.getDevice()->createBuffer(16);

    pBuffer->setBlob(&a, 0, 16);

    uint4 resA = pBuffer->getElement<uint4>(0);
    EXPECT_EQ(a, resA);

    pBuffer->setBlob(&b, 0, 16);

    uint4 resB = pBuffer->getElement<uint4>(0);
    EXPECT_EQ(b, resB);
}

GPU_TEST(BufferWrite)
{
    auto testWrite = [&ctx](uint4 testData, bool useInitData)
    {
        ref<Buffer> bufA =
            ctx.getDevice()->createBuffer(16, ResourceBindFlags::None, MemoryType::Upload, useInitData ? &testData : nullptr);
        ref<Buffer> bufB = ctx.getDevice()->createBuffer(16, ResourceBindFlags::None);

        if (!useInitData)
        {
            uint4* data = reinterpret_cast<uint4*>(bufA->map(Buffer::MapType::Write));
            std::memcpy(data, &testData, 16);
            bufA->unmap();
        }

        ctx.getDevice()->getRenderContext()->copyResource(bufB.get(), bufA.get());

        {
            uint4 result = bufB->getElement<uint4>(0);
            EXPECT_EQ(result, testData);
        }

        uint4 testData2 = testData * 10u;

        {
            uint4* data = reinterpret_cast<uint4*>(bufA->map(Buffer::MapType::Write));
            std::memcpy(data, &testData2, 16);
            bufA->unmap();
        }

        ctx.getDevice()->getRenderContext()->copyResource(bufB.get(), bufA.get());

        {
            uint4 result = bufB->getElement<uint4>(0);
            EXPECT_EQ(result, testData2);
        }
    };

    testWrite(uint4(1, 2, 3, 4), false);
    testWrite(uint4(3, 4, 5, 6), true);
}

} // namespace Falcor
