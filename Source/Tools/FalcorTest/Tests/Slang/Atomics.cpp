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
#include "Core/Pass/ComputePass.h"
#include <random>

namespace Falcor
{
namespace
{
const std::string_view kShaderFile = "Tests/Slang/Atomics.cs.slang";

const uint32_t kNumElems = 256;
std::mt19937 r;
std::uniform_real_distribution<float> u;

void testInterlockedAddF16(GPUUnitTestContext& ctx, std::string_view entryPoint)
{
    ref<Device> pDevice = ctx.getDevice();

    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile).csEntry("testBufferAddF16");
    desc.setUseSPIRVBackend(); // NOTE: The SPIR-V backend is required for RWByteAddressBuffer.InterlockedAddF16() on Vulkan!
    ctx.createProgram(desc);

    std::vector<float16_t> elems(kNumElems * 2);
    for (auto& v : elems)
        v = (float16_t)u(r);
    auto dataBuf =
        pDevice->createBuffer(kNumElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, elems.data());

    float zeros[2] = {};
    auto resultBuf = pDevice->createBuffer(
        2 * sizeof(float), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, zeros
    );

    auto var = ctx.vars().getRootVar();
    var["data"] = dataBuf;
    var["resultBuf"] = resultBuf;

    ctx.runProgram(kNumElems, 1, 1);

    // Verify results.
    float16_t result[4] = {};
    resultBuf->getBlob(&result, 0, 2 * sizeof(float));

    float16_t a[2] = {}, b[2] = {};
    for (uint32_t i = 0; i < 2 * kNumElems; i += 2)
    {
        a[0] += elems[i];
        a[1] += elems[i + 1];
        b[0] -= elems[i];
        b[1] -= elems[i + 1];
    }
    float e = 1.f;
    EXPECT_GE(result[0] + e, a[0]);
    EXPECT_LE(result[0] - e, a[0]);
    EXPECT_GE(result[1] + e, a[1]);
    EXPECT_LE(result[1] - e, a[1]);
    EXPECT_GE(result[2] + e, b[0]);
    EXPECT_LE(result[2] - e, b[0]);
    EXPECT_GE(result[3] + e, b[1]);
    EXPECT_LE(result[3] - e, b[1]);
}
} // namespace

GPU_TEST(Atomics_Buffer_InterlockedAddF16)
{
    testInterlockedAddF16(ctx, "testBufferAddF16");
}

GPU_TEST(Atomics_Buffer_InterlockedAddF16_2)
{
    testInterlockedAddF16(ctx, "testBufferAddF16_2");
}

GPU_TEST(Atomics_Buffer_InterlockedAddF32)
{
    ref<Device> pDevice = ctx.getDevice();

    ctx.createProgram(kShaderFile, "testBufferAddF32");

    std::vector<float> elems(kNumElems);
    for (auto& v : elems)
        v = u(r);
    auto dataBuf =
        pDevice->createBuffer(kNumElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, elems.data());

    float zeros[2] = {};
    auto resultBuf = pDevice->createBuffer(
        2 * sizeof(float), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, zeros
    );

    auto var = ctx.vars().getRootVar();
    var["data"] = dataBuf;
    var["resultBuf"] = resultBuf;

    ctx.runProgram(kNumElems, 1, 1);

    // Verify results.
    float result[2] = {};
    resultBuf->getBlob(&result, 0, 2 * sizeof(float));

    float a = 0.f, b = 0.f;
    for (uint32_t i = 0; i < kNumElems; i++)
    {
        a += elems[i];
        b -= elems[i];
    }
    float e = 1e-3f;
    EXPECT_GE(result[0] + e, a);
    EXPECT_LE(result[0] - e, a);
    EXPECT_GE(result[1] + e, b);
    EXPECT_LE(result[1] - e, b);
}

GPU_TEST(Atomics_Texture2D_InterlockedAddF32)
{
    ref<Device> pDevice = ctx.getDevice();

    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile).csEntry("testTextureAddF32");
    desc.setUseSPIRVBackend(); // NOTE: The SPIR-V backend is required for RWTexture2D.InterlockedAddF32() on Vulkan!
    ctx.createProgram(desc);

    std::vector<float> elems(kNumElems);
    for (auto& v : elems)
        v = u(r);
    auto dataBuf =
        pDevice->createBuffer(kNumElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, elems.data());

    float zeros[2] = {};
    auto resultTex = pDevice->createTexture2D(
        2, 1, ResourceFormat::R32Float, 1, 1, zeros, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    auto resultBuf = pDevice->createBuffer(
        2 * sizeof(float), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, zeros
    );

    auto var = ctx.vars().getRootVar();
    var["data"] = dataBuf;
    var["resultTex"] = resultTex;

    ctx.runProgram(kNumElems, 1, 1);

    // Copy result into readback buffer.
    {
        auto copyPass = ComputePass::create(pDevice, kShaderFile, "copyResult");
        auto copyVar = copyPass->getRootVar();
        copyVar["resultBuf"] = resultBuf;
        copyVar["resultTex"] = resultTex;
        copyPass->execute(pDevice->getRenderContext(), 256, 1);
    }

    // Verify results.
    float result[2] = {};
    resultBuf->getBlob(&result, 0, 2 * sizeof(float));

    float a = 0.f, b = 0.f;
    for (uint32_t i = 0; i < kNumElems; i++)
    {
        a += elems[i];
        b -= elems[i];
    }
    float e = 1e-3f;
    EXPECT_GE(result[0] + e, a);
    EXPECT_LE(result[0] - e, a);
    EXPECT_GE(result[1] + e, b);
    EXPECT_LE(result[1] - e, b);
}
} // namespace Falcor
