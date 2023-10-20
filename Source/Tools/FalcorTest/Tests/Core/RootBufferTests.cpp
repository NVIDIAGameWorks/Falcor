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

namespace Falcor
{
namespace
{
const uint32_t kNumElems = 256;
const std::string kRootBufferName = "testBuffer";

std::mt19937 rng;
auto dist = std::uniform_int_distribution<uint32_t>(0, 100);

uint32_t c0 = 31;
float c1 = 2.5f;

struct S
{
    float a;
    uint32_t b;
};

void testRootBuffer(GPUUnitTestContext& ctx, ShaderModel shaderModel, bool useUav)
{
    ref<Device> pDevice = ctx.getDevice();

    auto nextRandom = [&]() -> uint32_t { return dist(rng); };

    DefineList defines = {{"USE_UAV", useUav ? "1" : "0"}};
    SlangCompilerFlags compilerFlags = SlangCompilerFlags::None;

    ctx.createProgram("Tests/Core/RootBufferTests.cs.slang", "main", defines, compilerFlags, shaderModel);
    ctx.allocateStructuredBuffer("result", kNumElems);

    auto var = ctx.vars().getRootVar();
    var["CB"]["c0"] = c0;
    var["CB"]["c1"] = c1;

    // Bind some regular buffers.
    std::vector<uint32_t> rawBuffer(kNumElems);
    {
        for (uint32_t i = 0; i < kNumElems; i++)
            rawBuffer[i] = nextRandom();
        var["rawBuffer"] = pDevice->createBuffer(
            kNumElems * sizeof(uint32_t), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, rawBuffer.data()
        );
    }

    std::vector<S> structBuffer(kNumElems);
    {
        for (uint32_t i = 0; i < kNumElems; i++)
            structBuffer[i] = {nextRandom() + 0.5f, nextRandom()};
        var["structBuffer"] = pDevice->createStructuredBuffer(
            var["structBuffer"], kNumElems, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, structBuffer.data()
        );
    }

    std::vector<uint32_t> typedBufferUint(kNumElems);
    {
        for (uint32_t i = 0; i < kNumElems; i++)
            typedBufferUint[i] = nextRandom();
        var["typedBufferUint"] = pDevice->createTypedBuffer<uint32_t>(
            kNumElems, ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, typedBufferUint.data()
        );
    }

    std::vector<float4> typedBufferFloat4(kNumElems);
    {
        for (uint32_t i = 0; i < kNumElems; i++)
            typedBufferFloat4[i] = {nextRandom() * 0.25f, nextRandom() * 0.5f, nextRandom() * 0.75f, float(nextRandom())};
        var["typedBufferFloat4"] = pDevice->createTypedBuffer<float4>(
            kNumElems, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, typedBufferFloat4.data()
        );
    }

    // Test binding buffer to root descriptor.
    std::vector<uint32_t> testBuffer(kNumElems);
    {
        for (uint32_t i = 0; i < kNumElems; i++)
            testBuffer[i] = nextRandom();
        auto pTestBuffer = pDevice->createBuffer(
            kNumElems * sizeof(uint32_t),
            useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            testBuffer.data()
        );
        var[kRootBufferName] = pTestBuffer;

        ref<Buffer> pBoundBuffer = var[kRootBufferName];
        EXPECT_EQ(pBoundBuffer, pTestBuffer);
    }

    auto verifyResults = [&](auto str)
    {
        std::vector<float> result = ctx.readBuffer<float>("result");
        for (uint32_t i = 0; i < kNumElems; i++)
        {
            float r = 0.f;
            r += c0;
            r += c1;
            r += rawBuffer[i];
            r += typedBufferUint[i] * 2;
            r += typedBufferFloat4[i].z * 3;
            r += structBuffer[i].a * 4;
            r += structBuffer[i].b * 5;
            r += testBuffer[i] * 6;
            EXPECT_EQ(result[i], r) << "i = " << i << " (" << str << ")";
        }
    };

    // Run the program to test that we can access the buffer.
    ctx.runProgram(kNumElems, 1, 1);
    verifyResults("step 1");

    // Change the binding of other resources to test that the root buffer stays correctly bound.
    for (uint32_t i = 0; i < kNumElems; i++)
        rawBuffer[i] = nextRandom();
    var["rawBuffer"] =
        pDevice->createBuffer(kNumElems * sizeof(uint32_t), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, rawBuffer.data());
    for (uint32_t i = 0; i < kNumElems; i++)
        typedBufferFloat4[i] = {nextRandom() * 0.25f, nextRandom() * 0.5f, nextRandom() * 0.75f, float(nextRandom())};
    var["typedBufferFloat4"] =
        pDevice->createTypedBuffer<float4>(kNumElems, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, typedBufferFloat4.data());
    var["CB"]["c0"] = ++c0;

    ctx.runProgram(kNumElems, 1, 1);
    verifyResults("step 2");

    // Test binding a new root buffer.
    {
        for (uint32_t i = 0; i < kNumElems; i++)
            testBuffer[i] = nextRandom();
        auto pTestBuffer = pDevice->createBuffer(
            kNumElems * sizeof(uint32_t),
            useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            testBuffer.data()
        );
        var[kRootBufferName] = pTestBuffer;

        ref<Buffer> pBoundBuffer = var[kRootBufferName];
        EXPECT_EQ(pBoundBuffer, pTestBuffer);
    }

    ctx.runProgram(kNumElems, 1, 1);
    verifyResults("step 3");
}
} // namespace

GPU_TEST(RootBufferSRV_6_0)
{
    testRootBuffer(ctx, ShaderModel::SM6_0, false);
}

GPU_TEST(RootBufferUAV_6_0)
{
    testRootBuffer(ctx, ShaderModel::SM6_0, true);
}

GPU_TEST(RootBufferSRV_6_3)
{
    testRootBuffer(ctx, ShaderModel::SM6_3, false);
}

GPU_TEST(RootBufferUAV_6_3)
{
    testRootBuffer(ctx, ShaderModel::SM6_3, true);
}
} // namespace Falcor
