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

namespace Falcor
{
namespace
{
const uint32_t kElementCount = 256;

std::vector<uint32_t> getTestData()
{
    std::vector<uint32_t> data(kElementCount);
    for (uint32_t i = 0; i < kElementCount; i++)
        data[i] = i;
    return data;
}

const std::vector<uint32_t> kTestData = getTestData();

/** Create buffer with the given CPU access and elements initialized to 0,1,2,...
 */
ref<Buffer> createTestBuffer(GPUUnitTestContext& ctx, MemoryType memoryType, bool initialize)
{
    return ctx.getDevice()->createBuffer(
        kElementCount * sizeof(uint32_t),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        memoryType,
        initialize ? kTestData.data() : nullptr
    );
}

void checkData(GPUUnitTestContext& ctx, const uint32_t* pData)
{
    for (uint32_t i = 0; i < kElementCount; i++)
        EXPECT_EQ(pData[i], i) << "i = " << i;
}

void initBufferIndirect(GPUUnitTestContext& ctx, ref<Buffer> pBuffer)
{
    auto pInitData = createTestBuffer(ctx, MemoryType::DeviceLocal, true);
    ctx.getRenderContext()->copyResource(pBuffer.get(), pInitData.get());
    ctx.getRenderContext()->submit(true);
}

std::vector<uint32_t> readBufferIndirect(GPUUnitTestContext& ctx, ref<Buffer> pBuffer)
{
    ref<Buffer> pResult = createTestBuffer(ctx, MemoryType::DeviceLocal, false);
    ctx.getRenderContext()->copyResource(pResult.get(), pBuffer.get());
    ctx.getRenderContext()->submit(true);
    return pResult->getElements<uint32_t>();
}

void checkBufferIndirect(GPUUnitTestContext& ctx, ref<Buffer> pBuffer)
{
    return checkData(ctx, readBufferIndirect(ctx, pBuffer).data());
}

} // namespace

GPU_TEST(BufferDeviceLocalWrite)
{
    // Create without init data, then set data using setBlob().
    {
        ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::DeviceLocal, false);
        pBuffer->setBlob(kTestData.data(), 0, kTestData.size() * sizeof(uint32_t));
        checkBufferIndirect(ctx, pBuffer);
    }

    // Create with init data.
    {
        ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::DeviceLocal, true);
        checkBufferIndirect(ctx, pBuffer);
    }
}

GPU_TEST(BufferDeviceLocalRead)
{
    ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::DeviceLocal, false);
    initBufferIndirect(ctx, pBuffer);

    std::vector<uint32_t> data(kElementCount);
    pBuffer->getBlob(data.data(), 0, kElementCount * sizeof(uint32_t));
    checkData(ctx, data.data());
}

GPU_TEST(BufferUploadWrite)
{
    // Create without init data, then set data using setBlob().
    {
        ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::Upload, false);
        pBuffer->setBlob(kTestData.data(), 0, kTestData.size() * sizeof(uint32_t));
        checkBufferIndirect(ctx, pBuffer);
    }

    // Create with init data.
    {
        ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::Upload, true);
        checkBufferIndirect(ctx, pBuffer);
    }
}

GPU_TEST(BufferUploadMap)
{
    ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::Upload, false);
    uint32_t* pData = reinterpret_cast<uint32_t*>(pBuffer->map(Buffer::MapType::Write));
    for (uint32_t i = 0; i < kElementCount; ++i)
        pData[i] = kTestData[i];
    pBuffer->unmap();
    checkBufferIndirect(ctx, pBuffer);
}

GPU_TEST(BufferReadbackRead)
{
    ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::ReadBack, false);
    initBufferIndirect(ctx, pBuffer);

    std::vector<uint32_t> data(kElementCount);
    pBuffer->getBlob(data.data(), 0, kElementCount * sizeof(uint32_t));
    checkData(ctx, data.data());
}

GPU_TEST(BufferReadbackMap)
{
    ref<Buffer> pBuffer = createTestBuffer(ctx, MemoryType::ReadBack, false);
    initBufferIndirect(ctx, pBuffer);

    const uint32_t* pData = reinterpret_cast<const uint32_t*>(pBuffer->map(Buffer::MapType::Read));
    checkData(ctx, pData);
    pBuffer->unmap();
}

} // namespace Falcor
