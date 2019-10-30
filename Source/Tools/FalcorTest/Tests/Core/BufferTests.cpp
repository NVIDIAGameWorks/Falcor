/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Testing/UnitTest.h"

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

        template <Type type, typename BufferT>
        void testBuffer(GPUUnitTestContext& ctx, uint32_t numElems, uint32_t index = 0, uint32_t count = 0)
        {
            numElems = div_round_up(numElems, 256u) * 256u; // Make sure we run full thread groups.

            // Create a data blob for the test.
            // We fill it some numbers that don't overlap with the test buffers values.
            std::vector<uint32_t> blob;
            if (count > 0)
            {
                blob.resize(count);
                for (uint32_t i = 0; i < blob.size(); i++) blob[i] = (count - i) * 3;
            }

            // Create program and test buffer.
            Program::DefineList defines = { { "TYPE", std::to_string((uint32_t)type) } };
            ctx.createProgram("Tests/Core/BufferTests.cs.slang", "clearBuffer", defines);

            typename BufferT::SharedPtr pBuffer;
            if constexpr (type == Type::ByteAddressBuffer) pBuffer = BufferT::create(numElems * sizeof(uint32_t), ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            else if constexpr (type == Type::TypedBuffer) pBuffer = BufferT::create(numElems, ResourceBindFlags::UnorderedAccess);
            else if constexpr (type == Type::StructuredBuffer) pBuffer = StructuredBuffer::create(ctx.getProgram(), "buffer", numElems, ResourceBindFlags::UnorderedAccess);
            else static_assert(false);

            if constexpr (type == Type::ByteAddressBuffer) ctx.vars().setRawBuffer("buffer", pBuffer);
            else if constexpr (type == Type::TypedBuffer) ctx.vars().setTypedBuffer("buffer", pBuffer);
            else if constexpr (type == Type::StructuredBuffer) ctx.vars().setStructuredBuffer("buffer", pBuffer);
            else static_assert(false);

            // Run kernel to clear the buffer.
            // We clear explicitly instead of using clearUAV() as the latter is not compatible with RWStructuredBuffer.
            ctx.runProgram(numElems, 1, 1);

            ctx.createProgram("Tests/Core/BufferTests.cs.slang", "updateBuffer", defines);

            if constexpr (type == Type::ByteAddressBuffer) ctx.vars().setRawBuffer("buffer", pBuffer);
            else if constexpr (type == Type::TypedBuffer) ctx.vars().setTypedBuffer("buffer", pBuffer);
            else if constexpr (type == Type::StructuredBuffer) ctx.vars().setStructuredBuffer("buffer", pBuffer);
            else static_assert(false);

            // Run kernel N times to update the buffer (RMW).
            for (uint32_t i = 0; i < kIterations; i++) ctx.runProgram(numElems, 1, 1);

            // Use setBlob() to update part of the buffer from the CPU.
            if (count > 0)
            {
                assert(index + blob.size() <= numElems);
                pBuffer->setBlob(blob.data(), index * sizeof(uint32_t), blob.size() * sizeof(uint32_t));
            }

            // Run kernel N times to update the buffer (RMW).
            for (uint32_t i = 0; i < kIterations; i++) ctx.runProgram(numElems, 1, 1);

            // Run kernel to read values in the buffer.
            ctx.createProgram("Tests/Core/BufferTests.cs.slang", "readBuffer", defines);
            ctx.allocateStructuredBuffer("result", numElems);

            if constexpr (type == Type::ByteAddressBuffer) ctx.vars().setRawBuffer("buffer", pBuffer);
            else if constexpr (type == Type::TypedBuffer) ctx.vars().setTypedBuffer("buffer", pBuffer);
            else if constexpr (type == Type::StructuredBuffer) ctx.vars().setStructuredBuffer("buffer", pBuffer);
            else static_assert(false);

            ctx.runProgram(numElems, 1, 1);

            // Verify results.
            const uint32_t* pResult = ctx.mapBuffer<const uint32_t>("result");
            for (uint32_t i = 0; i < numElems; i++)
            {
                // Each RMW pass adds i+1 to the element at index i.
                // We run kIterations passes, then replace part of the buffer, followed by kIterations more passes.
                uint32_t expected = (i + 1) * kIterations * 2;
                if (i >= index && i < index + blob.size()) expected = blob[i - index] + (i + 1) * kIterations;
                uint32_t result = pResult[i];

                EXPECT_EQ(result, expected) << "i = " << i << " (numElems = " << numElems << " index = " << index << " count = " << count << ")";
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(RawBuffer)
    {
        auto testFunc = testBuffer<Type::ByteAddressBuffer, Buffer>;
        for (uint32_t numElems = 1u << 8; numElems <= (1u << 20); numElems <<= 4)
        {
            testFunc(ctx, numElems, 0, 0);
            testFunc(ctx, numElems, 0, 1);
            testFunc(ctx, numElems, 0, numElems / 2);
            testFunc(ctx, numElems, 1, 1);
            testFunc(ctx, numElems, numElems / 2 + 3, numElems / 4 - 1);
        }
    }

#if 0
    // Tests disables as setBlob() on TypedBuffer and StructuredBuffer doesn't work the same way as on Buffer.
    // For TypedBuffer and StructuredBuffer, the CPU copy of the entire buffer is re-uploaded to the GPU,
    // which invalidates previous data we may have populated them with on the GPU.
    // The Buffer class doesn't keep an internal CPU copy, so it won't invalidate the contents.

    GPU_TEST(TypedBuffer)
    {
        auto testFunc = testBuffer<Type::TypedBuffer, TypedBuffer<uint32_t>>;
        for (uint32_t numElems = 1u << 8; numElems <= (1u << 20); numElems <<= 4)
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
        auto testFunc = testBuffer<Type::StructuredBuffer, StructuredBuffer>;
        for (uint32_t numElems = 1u << 8; numElems <= (1u << 20); numElems <<= 4)
        {
            testFunc(ctx, numElems, 0, 0);
            testFunc(ctx, numElems, 0, 1);
            testFunc(ctx, numElems, 0, numElems / 2);
            testFunc(ctx, numElems, 1, 1);
            testFunc(ctx, numElems, numElems / 2 + 3, numElems / 4 - 1);
        }
    }

#endif
}
