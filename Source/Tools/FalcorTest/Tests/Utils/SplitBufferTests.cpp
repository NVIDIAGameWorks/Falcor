/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/SplitBuffer.h"

#include <random>
#include <chrono>

namespace Falcor
{
namespace
{

/// Creates a Structured GPU buffer from span.
template<typename T>
ref<Buffer> createBufferFromVector(
    std::string_view name,
    const ref<Device>& device,
    fstd::span<T> data,
    ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
    MemoryType memoryType = MemoryType::DeviceLocal
)
{
    if (data.empty())
        return ref<Buffer>();
    auto result = device->createStructuredBuffer(sizeof(T), data.size(), bindFlags, memoryType, data.data());
    result->setName(std::string(name));
    return result;
}

/// Creates a Structured GPU buffer from span.
template<typename T>
ref<Buffer> createBufferFromVector(
    std::string_view name,
    const ref<Device>& device,
    const std::vector<T>& data,
    ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
    MemoryType memoryType = MemoryType::DeviceLocal
)
{
    return createBufferFromVector(name, device, fstd::span<const T>(data.data(), data.size()), bindFlags, memoryType);
}

struct S32B
{
    S32B() = default;
    S32B(float _f) : f0(_f), f1(_f) {}
    float4 f0, f1;

    bool operator==(const S32B& rhs) const { return math::all(f0 == rhs.f0) && math::all(f1 == rhs.f1); }

    friend std::ostream& operator<<(std::ostream& os, const S32B& x)
    {
        os << fmt::format("({},{})", x.f0, x.f1);
        return os;
    }
};

struct S4B
{
    S4B() = default;
    S4B(float _f) : f(_f) {}
    float f;

    bool operator==(const S4B& rhs) const { return f == rhs.f; }

    friend std::ostream& operator<<(std::ostream& os, const S4B& x)
    {
        os << x.f;
        return os;
    }
};

using uint16_t3 = uint16_t[3];
using uint32_t3 = uint32_t[3];

template<typename T>
struct BaseTypeTrait
{
    using BaseType = T;
};

template<>
struct BaseTypeTrait<S4B>
{
    using BaseType = float;
};

template<>
struct BaseTypeTrait<S32B>
{
    using BaseType = float;
};

template<>
struct BaseTypeTrait<uint16_t3>
{
    using BaseType = uint16_t;
};

template<>
struct BaseTypeTrait<uint32_t3>
{
    using BaseType = uint32_t;
};

struct RangeDesc
{
    // The tested concept
    uint offset;
    uint count;
    // The debugging info
    uint bufferIndex;
    uint bufferOffset;
};

template<typename T, typename U, bool TByteBuffer>
RangeDesc insertData(SplitBuffer<U, TByteBuffer>& buffer, uint32_t count)
{
    using BaseType = typename BaseTypeTrait<T>::BaseType;
    std::vector<U> dataToInsert;
    dataToInsert.resize((count * sizeof(T) + sizeof(U) - 1) / sizeof(U), U(0));
    auto data = reinterpret_cast<BaseType*>(dataToInsert.data());
    for (uint32_t i = 0; i < (count * sizeof(T) / sizeof(BaseType)); ++i)
        data[i] = BaseType(i);

    RangeDesc result;
    result.offset = buffer.insert(dataToInsert.begin(), dataToInsert.end());
    result.count = count;
    result.bufferIndex = buffer.getBufferIndex(result.offset);
    result.bufferOffset = buffer.getElementIndex(result.offset);
    return result;
}

template<typename T, typename U, bool TByteBuffer>
RangeDesc insertEmpty(SplitBuffer<U, TByteBuffer>& buffer, uint32_t count)
{
    RangeDesc result;
    result.offset = buffer.insertEmpty(count);
    result.count = count;
    result.bufferIndex = buffer.getBufferIndex(result.offset);
    result.bufferOffset = buffer.getElementIndex(result.offset);
    return result;
}

} // namespace

GPU_TEST(SplitBuffer_ByteBuffer_Large48b)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = uint32_t;
    using DataElementType = uint16_t3;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, true> buffer;
    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    insertEmpty<DataElementType>(buffer, size_t(k2GB - buffer.getByteSize() * 1.5) / sizeof(BufferElementType));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_BYTE_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitByteBuffer48b", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitByteBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gByteBufferUint16_t3", ctx.getDevice(), empty);
            ;
        }

        ctx["gByteBufferUint16_t3"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_ByteBuffer_Large96b)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = uint32_t;
    using DataElementType = uint32_t3;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, true> buffer;
    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    insertEmpty<DataElementType>(buffer, size_t(k2GB - buffer.getByteSize() * 1.5) / sizeof(BufferElementType));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_BYTE_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitByteBuffer96b", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitByteBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gByteBufferUint32_t3", ctx.getDevice(), empty);
            ;
        }

        ctx["gByteBufferUint32_t3"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_StructuredBuffer_Large4B)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = S4B;
    using DataElementType = S4B;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, false> buffer;
    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    insertEmpty<DataElementType>(buffer, size_t(k2GB - buffer.getByteSize() * 1.5) / sizeof(BufferElementType));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_STRUCT_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    if (sizeof(BufferElementType) == 4)
        defines.add("USE_4B_SIZE", "1");
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitStructuredBuffer", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitStructBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gStructuredBuffer", ctx.getDevice(), empty);
            ;
        }

        ctx["gStructuredBuffer"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_StructuredBuffer_Large32B)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = S32B;
    using DataElementType = S32B;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, false> buffer;
    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    insertEmpty<DataElementType>(buffer, size_t(k4GB - buffer.getByteSize() * 1.5) / sizeof(BufferElementType));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_STRUCT_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    if (sizeof(BufferElementType) == 4)
        defines.add("USE_4B_SIZE", "1");
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitStructuredBuffer", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitStructBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gStructuredBuffer", ctx.getDevice(), empty);
            ;
        }

        ctx["gStructuredBuffer"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_ByteBuffer_Many48b)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = uint32_t;
    using DataElementType = uint16_t3;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, true> buffer;
    buffer.setBufferCount(buffer.getMaxBufferCount());

    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));

    buffer.setBufferCountDefinePrefix("SPLIT_BYTE_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitByteBuffer48b", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitByteBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gByteBufferUint16_t3", ctx.getDevice(), empty);
            ;
        }

        ctx["gByteBufferUint16_t3"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_ByteBuffer_Many96b)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = uint32_t;
    using DataElementType = uint32_t3;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, true> buffer;
    buffer.setBufferCount(buffer.getMaxBufferCount());

    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_BYTE_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitByteBuffer96b", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitByteBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gByteBufferUint32_t3", ctx.getDevice(), empty);
            ;
        }

        ctx["gByteBufferUint32_t3"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_StructuredBuffer_Many4B)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = S4B;
    using DataElementType = S4B;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, false> buffer;
    buffer.setBufferCount(buffer.getMaxBufferCount());

    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_STRUCT_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    if (sizeof(BufferElementType) == 4)
        defines.add("USE_4B_SIZE", "1");
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitStructuredBuffer", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitStructBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gStructuredBuffer", ctx.getDevice(), empty);
            ;
        }

        ctx["gStructuredBuffer"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

GPU_TEST(SplitBuffer_StructuredBuffer_Many32B)
{
    static size_t k2GB = UINT64_C(1) << UINT64_C(31);
    static size_t k4GB = UINT64_C(1) << UINT64_C(32);

    using BufferElementType = S32B;
    using DataElementType = S32B;
    size_t minCount = 32 * 1024;
    size_t maxCount = 256 * 1024;

    SplitBuffer<BufferElementType, false> buffer;
    buffer.setBufferCount(buffer.getMaxBufferCount());

    std::vector<RangeDesc> ranges;
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    for (size_t i = 0; i < 16; ++i)
        ranges.push_back(insertData<DataElementType>(buffer, (32 + i) * 1024 + i));
    EXPECT_EQ(ranges.front().bufferIndex, 0u);
    EXPECT_GT(ranges.back().bufferIndex, 0u);

    buffer.setBufferCountDefinePrefix("SPLIT_STRUCT_BUFFER");
    buffer.createGpuBuffers(ctx.getDevice(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    std::vector<ref<Buffer>> results(buffer.getBufferCount());

    DefineList defines;
    buffer.getShaderDefines(defines);
    if (sizeof(BufferElementType) == 4)
        defines.add("USE_4B_SIZE", "1");
    ctx.createProgram("Tests/Utils/SplitBufferTests.cs.slang", "testSplitStructuredBuffer", defines);
    ctx["gRangeCount"] = uint32_t(ranges.size());
    ctx["gRangeDescs"] = createBufferFromVector("gRangeDescs", ctx.getDevice(), ranges);
    buffer.bindShaderData(ctx["gSplitStructBuffer"]);
    std::vector<BufferElementType> empty;
    for (size_t i = 0; i < buffer.getBufferCount(); ++i)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(i);
        if (!cpuBuffer.empty())
        {
            empty.assign(cpuBuffer.size(), BufferElementType{0});
            results[i] = createBufferFromVector("gStructuredBuffer", ctx.getDevice(), empty);
            ;
        }

        ctx["gStructuredBuffer"][i] = results[i];
    }

    ctx.runProgram(ranges.size());

    for (const RangeDesc& it : ranges)
    {
        auto& cpuBuffer = buffer.getCpuBuffer(it.bufferIndex);
        EXPECT_FALSE(cpuBuffer.empty());
        auto fromGpu = results[it.bufferIndex]->getElements<BufferElementType>(it.bufferOffset, it.count);
        fstd::span<const BufferElementType> fromCpu(cpuBuffer.data() + it.bufferOffset, cpuBuffer.data() + it.bufferOffset + it.count);

        EXPECT_EQ(fromCpu.size(), fromGpu.size());
        EXPECT(memcmp(fromCpu.data(), fromGpu.data(), fromCpu.size() * sizeof(BufferElementType)) == 0);
    }
}

} // namespace Falcor
