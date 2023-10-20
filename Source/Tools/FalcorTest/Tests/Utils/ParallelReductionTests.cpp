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
#include "Utils/Algorithm/ParallelReduction.h"
#include <random>

namespace Falcor
{
namespace
{
// Utility classes for compact data types on the GPU.
// TODO: Make types like these part of new math library.

// Unsigned normalized integer.
template<typename T, int bits>
struct unorm_t
{
    unorm_t(float v) { _val = static_cast<T>(std::min(std::max(v, 0.f), 1.f) * _scale + 0.5f); }
    unorm_t& operator=(float v)
    {
        _val = unorm_t(v);
        return *this;
    }
    operator float() { return static_cast<float>(_val) / _scale; }

private:
    static const uint32_t _scale = (1u << bits) - 1;
    T _val;
};

using unorm8_t = unorm_t<uint8_t, 8>;
using unorm16_t = unorm_t<uint16_t, 16>;

// Signed normalized integer.
template<typename T, int bits>
struct snorm_t
{
    snorm_t(float v) { _val = static_cast<T>(std::min(std::max(v, -1.f), 1.f) * _scale + (v >= 0.f ? 0.5f : -0.5f)); }
    snorm_t& operator=(float v)
    {
        _val = snorm_t(v);
        return *this;
    }
    operator float() { return static_cast<float>(_val) / _scale; }

private:
    static const uint32_t _scale = (1u << (bits - 1)) - 1;
    T _val;
};

using snorm8_t = snorm_t<int8_t, 8>;
using snorm16_t = snorm_t<int16_t, 16>;

// Quantize and store value. The de-quantized value is returned in 'val'.
template<typename T>
void store(void* ptr, size_t idx, float& val)
{
    T qval = static_cast<T>(val);
    val = static_cast<float>(qval);
    reinterpret_cast<T*>(ptr)[idx] = qval;
}

template<typename DataType, typename RefType>
void testReduction(GPUUnitTestContext& ctx, ParallelReduction& reduction, ResourceFormat format, uint32_t width, uint32_t height)
{
    ref<Device> pDevice = ctx.getDevice();

    // Create random test data.
    const uint32_t channels = getFormatChannelCount(format);
    const size_t elems = width * height * channels;
    const uint32_t sz = getFormatBytesPerBlock(format) / channels;
    const FormatType type = getFormatType(format);

    FALCOR_ASSERT(getFormatPixelsPerBlock(format) == 1);
    FALCOR_ASSERT(!isCompressedFormat(format));
    FALCOR_ASSERT(channels >= 1);
    FALCOR_ASSERT(elems > 0);
    FALCOR_ASSERT(sz * channels == getFormatBytesPerBlock(format));
    FALCOR_ASSERT(sz == 1 || sz == 2 || sz == 4);
    FALCOR_ASSERT(type != FormatType::Unknown && type != FormatType::UnormSrgb);

    std::default_random_engine rng;
    auto dist = std::uniform_real_distribution<float>();
    auto u = [&]() -> float { return dist(rng); };

    auto pInitData = std::make_unique<uint8_t[]>(elems * sz);
    void* ptr = pInitData.get();

    RefType refSum[4] = {};
    RefType absSum[4] = {};
    RefType minValue = std::numeric_limits<RefType>::lowest();
    RefType maxValue = std::numeric_limits<RefType>::max();
    RefType refMin[4] = {maxValue, maxValue, maxValue, maxValue};
    RefType refMax[4] = {minValue, minValue, minValue, minValue};

    for (size_t i = 0; i < elems; i++)
    {
        // Compute random quantized number.
        // The values are in a range that is exactly representable in float.
        float value = 0.f;
        if (type == FormatType::Float)
        {
            value = u() * 200.f - 100.f;
            if (sz == 2)
                store<float16_t>(ptr, i, value);
            else if (sz == 4)
                store<float>(ptr, i, value);
            else
                FALCOR_UNREACHABLE();
        }
        else if (type == FormatType::Sint)
        {
            value = u() * 200.f - 100.f;
            if (sz == 1)
                store<int8_t>(ptr, i, value);
            else if (sz == 2)
                store<int16_t>(ptr, i, value);
            else if (sz == 4)
                store<int32_t>(ptr, i, value);
            else
                FALCOR_UNREACHABLE();
        }
        else if (type == FormatType::Uint)
        {
            value = u() * 200.f;
            if (sz == 1)
                store<uint8_t>(ptr, i, value);
            else if (sz == 2)
                store<uint16_t>(ptr, i, value);
            else if (sz == 4)
                store<uint32_t>(ptr, i, value);
            else
                FALCOR_UNREACHABLE();
        }
        else if (type == FormatType::Unorm)
        {
            value = u();
            if (sz == 1)
                store<unorm8_t>(ptr, i, value);
            else if (sz == 2)
                store<unorm16_t>(ptr, i, value);
            else
                FALCOR_UNREACHABLE();
        }
        else if (type == FormatType::Snorm)
        {
            value = u() * 2.f - 1.f;
            if (sz == 1)
                store<snorm8_t>(ptr, i, value);
            else if (sz == 2)
                store<snorm16_t>(ptr, i, value);
            else
                FALCOR_UNREACHABLE();
        }
        else
            FALCOR_UNREACHABLE();

        // Compute reference sum (per channel).
        refSum[i % channels] += (RefType)value;
        absSum[i % channels] += (RefType)std::abs(value);
        refMin[i % channels] = std::min(refMin[i % channels], (RefType)value);
        refMax[i % channels] = std::max(refMax[i % channels], (RefType)value);
    }

    // Create a texture with test data.
    ref<Texture> pTexture = pDevice->createTexture2D(width, height, format, 1, 1, pInitData.get());

    // Test Sum operation.
    {
        // Allocate buffer for the result on the GPU.
        DataType nullValue = {};
        ref<Buffer> pResultBuffer = pDevice->createBuffer(16, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &nullValue);

        // Perform reduction operation.
        DataType result;
        reduction.execute(ctx.getRenderContext(), pTexture, ParallelReduction::Type::Sum, &result, pResultBuffer, 0);

        // Verify that returned result is identical to result stored to GPU buffer.
        DataType resultBuffer = pResultBuffer->getElement<DataType>(0);
        for (uint32_t i = 0; i < 4; i++)
        {
            EXPECT_EQ(resultBuffer[i], result[i % 4]) << "i = " << i;
        }

        // Compare result to reference value computed on the CPU.
        for (uint32_t i = 0; i < 4; i++)
        {
            if (i < channels)
            {
                if constexpr (std::is_floating_point<RefType>::value)
                {
                    // For floating-point formats, calculate relative error with respect to the sum of absolute values.
                    double e = std::abs((RefType)result[i] - refSum[i]);
                    double relError = (double)e / absSum[i];
                    EXPECT_LE(relError, 1e-6) << "i = " << i;
                }
                else
                {
                    // For integer formats, we expect the exact result.
                    EXPECT_EQ(result[i], refSum[i]) << "i = " << i;
                }
            }
            else
            {
                EXPECT_EQ(result[i], 0) << "i = " << i;
            }
        }
    }

    // Test MinMax operation
    {
        // Allocate buffer for the result on the GPU.
        DataType nullValues[2] = {{}, {}};
        ref<Buffer> pResultBuffer = pDevice->createBuffer(32, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullValues);

        // Perform reduction operation.
        DataType result[2];
        reduction.execute(ctx.getRenderContext(), pTexture, ParallelReduction::Type::MinMax, result, pResultBuffer, 0);

        // Verify that returned result is identical to result stored to GPU buffer.
        std::vector<DataType> resultBuffer = pResultBuffer->getElements<DataType>();
        for (uint32_t i = 0; i < 2; i++)
        {
            for (uint32_t j = 0; j < 4; j++)
            {
                EXPECT_EQ(resultBuffer[i][j], result[i][j]) << "i = " << i << " j = " << j;
            }
        }

        // Compare result to reference value computed on the CPU.
        for (uint32_t i = 0; i < 4; i++)
        {
            if (i < channels)
            {
                // For integer formats, we expect the exact result.
                EXPECT_EQ(result[0][i], refMin[i]) << "i = " << i;
                EXPECT_EQ(result[1][i], refMax[i]) << "i = " << i;
            }
        }
    }
}

void testReduction(GPUUnitTestContext& ctx, ParallelReduction& reduction, ResourceFormat format, uint32_t width, uint32_t height)
{
    const FormatType type = getFormatType(format);
    if (type == FormatType::Uint)
        testReduction<uint4, uint32_t>(ctx, reduction, format, width, height);
    else if (type == FormatType::Sint)
        testReduction<int4, int32_t>(ctx, reduction, format, width, height);
    else
        testReduction<float4, double>(ctx, reduction, format, width, height);
}
} // namespace

GPU_TEST(ParallelReduction)
{
    // Quick test of the snorm/unorm data types we use.
    FALCOR_ASSERT((float)unorm8_t(163.499f / 255.f) == (163 / 255.f));
    FALCOR_ASSERT((float)unorm16_t(163.501f / 65535.f) == (164 / 65535.f));
    FALCOR_ASSERT((float)snorm8_t(10.499f / 127.f) == (10 / 127.f));
    FALCOR_ASSERT((float)snorm8_t(10.501f / 127.f) == (11 / 127.f));
    FALCOR_ASSERT((float)snorm8_t(-10.499f / 127.f) == (-10 / 127.f));
    FALCOR_ASSERT((float)snorm8_t(-10.501f / 127.f) == (-11 / 127.f));
    FALCOR_ASSERT((float)snorm16_t(-10.499f / 32767.f) == (-10 / 32767.f));
    FALCOR_ASSERT((float)snorm16_t(-10.501f / 32767.f) == (-11 / 32767.f));

    // Create reduction operation.
    ParallelReduction reduction(ctx.getDevice());

    // Test floating-point formats.
    testReduction(ctx, reduction, ResourceFormat::RGBA32Float, 1, 1);
    testReduction(ctx, reduction, ResourceFormat::RGBA32Float, 32, 64);
    testReduction(ctx, reduction, ResourceFormat::RGBA32Float, 127, 71);
    testReduction(ctx, reduction, ResourceFormat::RGBA8Unorm, 256, 192);
    testReduction(ctx, reduction, ResourceFormat::RGBA8Snorm, 91, 130);
    testReduction(ctx, reduction, ResourceFormat::RG16Float, 220, 121);
    testReduction(ctx, reduction, ResourceFormat::RG16Unorm, 256, 192);
    testReduction(ctx, reduction, ResourceFormat::RG16Snorm, 333, 101);

    // Test integer formats.
    testReduction(ctx, reduction, ResourceFormat::RGBA32Uint, 33, 99);
    testReduction(ctx, reduction, ResourceFormat::R32Uint, 22, 291);
    testReduction(ctx, reduction, ResourceFormat::R16Int, 64, 33);
    testReduction(ctx, reduction, ResourceFormat::RG8Int, 403, 57);
}
} // namespace Falcor
