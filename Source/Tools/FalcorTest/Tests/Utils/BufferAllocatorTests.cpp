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
#include "Utils/BufferAllocator.h"

namespace Falcor
{
struct S
{
    float a;
    float b;
    uint32_t c;
    S(float _a, float _b, float _c) : a(_a), b(_b), c(_c) {}
};

GPU_TEST(BufferAllocatorNoAlign)
{
    // Raw buffer without any alignment requirements. Everything is tightly packed in memory.
    BufferAllocator buf(0, 0, 0);
    EXPECT_EQ(buf.getSize(), 0);

    // Make some allocations.
    {
        size_t offset = buf.allocate(4);
        buf.set<uint32_t>(offset, 11);

        offset = buf.allocate<uint32_t>();
        buf.set<uint32_t>(offset, 99);

        offset = buf.allocate<float4>();
        buf.set(offset, float4(1.1f, 2.5f, 13.3f, -1.2f));

        EXPECT_EQ(offset, 8);

        buf.emplaceBack<float>(18.4f);
        buf.emplaceBack<uint16_t>(33391);

        EXPECT_EQ(buf.getSize(), 30);

        offset = buf.allocate<float>(25);
        EXPECT_EQ(offset, 30);

        float d[25];
        for (size_t i = 0; i < 25; i++)
        {
            d[i] = (float)i + 0.11f;
        }
        buf.setBlob(d, offset, 25 * sizeof(float));

        buf.emplaceBack<S>(5.9f, 3.3f, 19);

        S obj(9.1f, 10.f, 333);
        buf.pushBack(obj);
    }

    // Validate CPU buffer.
    {
        EXPECT_EQ(buf.getSize(), 154);

        const uint8_t* ptr = buf.getStartPointer();

        EXPECT_EQ(*(uint32_t*)(ptr + 0), 11);
        EXPECT_EQ(*(uint32_t*)(ptr + 4), 99);
        EXPECT_EQ(*(float*)(ptr + 8), 1.1f);
        EXPECT_EQ(*(float*)(ptr + 12), 2.5f);
        EXPECT_EQ(*(float*)(ptr + 16), 13.3f);
        EXPECT_EQ(*(float*)(ptr + 20), -1.2f);
        EXPECT_EQ(*(float*)(ptr + 24), 18.4f);
        EXPECT_EQ(*(uint16_t*)(ptr + 28), 33391);

        float* d = (float*)(ptr + 30);
        for (size_t i = 0; i < 25; i++)
        {
            EXPECT_EQ(d[i], (float)i + 0.11f);
        }

        EXPECT_EQ(*(float*)(ptr + 130), 5.9f);
        EXPECT_EQ(*(float*)(ptr + 134), 3.3f);
        EXPECT_EQ(*(uint32_t*)(ptr + 138), 19);

        EXPECT_EQ(*(float*)(ptr + 142), 9.1f);
        EXPECT_EQ(*(float*)(ptr + 146), 10.f);
        EXPECT_EQ(*(uint32_t*)(ptr + 150), 333);
    }

    // Validate GPU buffer.
    auto validateGpuBuffer = [&]()
    {
        ref<Buffer> pBuffer = buf.getGPUBuffer(ctx.getDevice());

        EXPECT(!pBuffer->isStructured());
        EXPECT(!pBuffer->isTyped());
        EXPECT_EQ(pBuffer->getSize(), 156); // Size should be padded to the next 4B boundary.

        const uint8_t* ref = buf.getStartPointer();
        std::vector<uint8_t> data = pBuffer->getElements<uint8_t>(0, buf.getSize());
        for (size_t i = 0; i < buf.getSize(); i++)
        {
            EXPECT_EQ((uintptr_t)data[i], (uintptr_t)ref[i]) << "i=" << i;
        }
    };

    validateGpuBuffer();

    // Low-level access.
    {
        uint8_t* ptr = buf.getStartPointer();

        float* a = (float*)(ptr + 24);
        *a = 55.4f;
        buf.modified<float>(24);

        float* b = (float*)(ptr + 130);
        *b = 0.004f;
        buf.modified(130, 4);
    }

    validateGpuBuffer();
}

GPU_TEST(BufferAllocatorAlign)
{
    // Raw buffer with alignment and cacheline alignment.
    BufferAllocator buf(16, 0, 128);
    EXPECT_EQ(buf.getSize(), 0);

    // Make some allocations. Check that we get the expected alignment.
    {
        size_t offset = buf.allocate(20);
        EXPECT_EQ(offset, 0);
        EXPECT_EQ(buf.getSize(), 20);

        offset = buf.allocate(4);
        EXPECT_EQ(offset, 32);
        EXPECT_EQ(buf.getSize(), 36);

        // Allocation that would stride two cache lines. It should get placed at the start of the next line.
        offset = buf.allocate(100);
        EXPECT_EQ(offset, 128);
        EXPECT_EQ(buf.getSize(), 228);

        offset = buf.allocate(4);
        EXPECT_EQ(offset, 240);
        EXPECT_EQ(buf.getSize(), 244);

        // Allocation that would stride two cache lines. It should get placed at the start of the next line.
        offset = buf.allocate(128);
        EXPECT_EQ(offset, 256);
        EXPECT_EQ(buf.getSize(), 384);

        // Allocation that strides five cache lines. It would not help to move it.
        offset = buf.allocate(590);
        EXPECT_EQ(offset, 384);
        EXPECT_EQ(buf.getSize(), 974);

        offset = buf.allocate(20);
        EXPECT_EQ(offset, 976);
        EXPECT_EQ(buf.getSize(), 996);

        // Allocation that would stride two cache lines. It should get placed at the start of the next line.
        offset = buf.allocate(24);
        EXPECT_EQ(offset, 1024);
        EXPECT_EQ(buf.getSize(), 1048);

        // Allocation that strides three cache lines. It would not help to move it.
        offset = buf.allocate(130);
        EXPECT_EQ(offset, 1056);
        EXPECT_EQ(buf.getSize(), 1186);
    }

    // Get the GPU buffer. Make sure it's the expected size.
    ref<Buffer> pBuffer = buf.getGPUBuffer(ctx.getDevice());

    EXPECT(!pBuffer->isStructured());
    EXPECT(!pBuffer->isTyped());
    EXPECT_EQ(pBuffer->getSize(), 1188); // Size should be padded to the next 4B boundary.
}

GPU_TEST(BufferAllocatorStructNoAlign)
{
    // Structured buffer without any alignment requirements.
    BufferAllocator buf(0, 16, 0);
    EXPECT_EQ(buf.getSize(), 0);

    // Make some allocations.
    {
        buf.emplaceBack<float4>(1, 2, 3, 4);
        buf.emplaceBack<float4>(5, 6, 7, 8);
        buf.emplaceBack<float4>(9, 10, 11, 12);
        buf.emplaceBack<float>(13);           // We can still allocate smaller (or larger) blocks than the struct size.
        buf.pushBack(float4(14, 15, 16, 17)); // Tightly packed after the previous allocation.

        EXPECT_EQ(buf.getSize(), 68);
    }

    // Validate CPU buffer.
    {
        const float* data = reinterpret_cast<const float*>(buf.getStartPointer());
        for (size_t i = 0; i < 17; i++)
        {
            EXPECT_EQ(data[i], (float)i + 1);
        }
    }

    // Validate GPU buffer.
    {
        ref<Buffer> pBuffer = buf.getGPUBuffer(ctx.getDevice());

        EXPECT(pBuffer->isStructured());
        EXPECT(!pBuffer->isTyped());
        EXPECT_EQ(pBuffer->getStructSize(), 16);
        EXPECT_EQ(pBuffer->getSize(), 80); // Size should be padded to a whole number of structs.

        std::vector<float> data = pBuffer->getElements<float>(0, 17);
        for (size_t i = 0; i < 17; i++)
        {
            EXPECT_EQ(data[i], (float)i + 1);
        }
    }
}

GPU_TEST(BufferAllocatorStructAlign)
{
    // Structured buffer with alignment and cacheline alignment.
    // Note that the fact that it is a structured buffer instead of raw buffer should have no effect on the computed
    // alignments.
    BufferAllocator buf(16, 32, 128);
    EXPECT_EQ(buf.getSize(), 0);

    // Make some allocations.
    {
        size_t offset = buf.emplaceBack<float>(1.2f);
        EXPECT_EQ(offset, 0);
        EXPECT_EQ(buf.getSize(), 4);

        offset = buf.pushBack<float>(3.4f);
        EXPECT_EQ(offset, 16);
        EXPECT_EQ(buf.getSize(), 20);

        offset = buf.allocate(12);
        buf.set(offset, 4.7f);
        buf.set(offset + 4, 5.7f);
        buf.set(offset + 8, 6.7f);
        EXPECT_EQ(offset, 32);
        EXPECT_EQ(buf.getSize(), 44);

        // Allocation that would stride two cache lines. It should get placed at the start of the next line.
        offset = buf.allocate(84);
        float d[21];
        for (size_t i = 0; i < 21; i++)
            d[i] = (float)i + 0.5f;
        buf.setBlob(d, offset, 21 * sizeof(float));
        EXPECT_EQ(offset, 128);
        EXPECT_EQ(buf.getSize(), 212);
    }

    // Validate CPU buffer.
    {
        const float* data = reinterpret_cast<const float*>(buf.getStartPointer());

        EXPECT_EQ(data[0], 1.2f);
        EXPECT_EQ(data[4], 3.4f);
        EXPECT_EQ(data[8], 4.7f);
        EXPECT_EQ(data[9], 5.7f);
        EXPECT_EQ(data[10], 6.7f);

        for (size_t i = 0; i < 21; i++)
        {
            EXPECT_EQ(data[32 + i], (float)i + 0.5f);
        }
    }

    // Validate GPU buffer.
    {
        ref<Buffer> pBuffer = buf.getGPUBuffer(ctx.getDevice());

        EXPECT(pBuffer->isStructured());
        EXPECT(!pBuffer->isTyped());
        EXPECT_EQ(pBuffer->getStructSize(), 32);
        EXPECT_EQ(pBuffer->getSize(), 224); // Size should be padded to a whole number of structs.

        const float* ref = reinterpret_cast<const float*>(buf.getStartPointer());
        std::vector<float> data = pBuffer->getElements<float>(0, buf.getSize() / 4);
        for (size_t i = 0; i < buf.getSize() / 4; i++)
        {
            EXPECT_EQ(data[i], ref[i]);
        }
    }
}

} // namespace Falcor
