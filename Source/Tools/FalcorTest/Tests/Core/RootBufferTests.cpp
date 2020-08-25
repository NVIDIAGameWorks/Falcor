/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
        auto r = [&]() -> uint32_t { return dist(rng); };

        uint32_t c0 = 31;
        float c1 = 2.5f;

        struct S
        {
            float a;
            uint32_t b;
        };

        void testRootBuffer(GPUUnitTestContext& ctx, const std::string& shaderModel, bool useUav)
        {
            Program::DefineList defines = { {"USE_UAV", useUav ? "1" : "0"} };
            Shader::CompilerFlags compilerFlags = Shader::CompilerFlags::None; // Shader::CompilerFlags::DumpIntermediates;

            ctx.createProgram("Tests/Core/RootBufferTests.cs.slang", "main", defines, compilerFlags, shaderModel);
            ctx.allocateStructuredBuffer("result", kNumElems);

            auto var = ctx.vars().getRootVar();
            var["CB"]["c0"] = c0;
            var["CB"]["c1"] = c1;

            // Bind some regular buffers.
            std::vector<uint32_t> rawBuffer(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) rawBuffer[i] = r();
                var["rawBuffer"] = Buffer::create(kNumElems * sizeof(uint32_t), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, rawBuffer.data());
            }

            std::vector<S> structBuffer(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) structBuffer[i] = { r() + 0.5f, r() };
                var["structBuffer"] = Buffer::createStructured(var["structBuffer"], kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, structBuffer.data());
            }

            std::vector<uint32_t> typedBufferUint(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) typedBufferUint[i] = r();
                var["typedBufferUint"] = Buffer::createTyped<uint32_t>(kNumElems, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, typedBufferUint.data());
            }

            std::vector<float4> typedBufferFloat4(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) typedBufferFloat4[i] = { r() * 0.25f, r() * 0.5f, r() * 0.75f, r() };
                var["typedBufferFloat4"] = Buffer::createTyped<float4>(kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, typedBufferFloat4.data());
            }

            // Test binding buffer to root descriptor.
            std::vector<uint32_t> testBuffer(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) testBuffer[i] = r();
                auto pTestBuffer = Buffer::create(kNumElems * sizeof(uint32_t), useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, testBuffer.data());
                var[kRootBufferName] = pTestBuffer;

                Buffer::SharedPtr pBoundBuffer = var[kRootBufferName];
                EXPECT_EQ(pBoundBuffer, pTestBuffer);
            }

            auto verifyResults = [&](auto str) {
                const float* result = ctx.mapBuffer<const float>("result");
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
                ctx.unmapBuffer("result");
            };

            // Run the program to test that we can access the buffer.
            ctx.runProgram(kNumElems, 1, 1);
            verifyResults("step 1");

            // Change the binding of other resources to test that the root buffer stays correctly bound.
            for (uint32_t i = 0; i < kNumElems; i++) rawBuffer[i] = r();
            var["rawBuffer"] = Buffer::create(kNumElems * sizeof(uint32_t), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, rawBuffer.data());
            for (uint32_t i = 0; i < kNumElems; i++) typedBufferFloat4[i] = { r() * 0.25f, r() * 0.5f, r() * 0.75f, r() };
            var["typedBufferFloat4"] = Buffer::createTyped<float4>(kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, typedBufferFloat4.data());
            var["CB"]["c0"] = ++c0;

            ctx.runProgram(kNumElems, 1, 1);
            verifyResults("step 2");

            // Test binding a new root buffer.
            {
                for (uint32_t i = 0; i < kNumElems; i++) testBuffer[i] = r();
                auto pTestBuffer = Buffer::create(kNumElems * sizeof(uint32_t), useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, testBuffer.data());
                var[kRootBufferName] = pTestBuffer;

                Buffer::SharedPtr pBoundBuffer = var[kRootBufferName];
                EXPECT_EQ(pBoundBuffer, pTestBuffer);
            }

            ctx.runProgram(kNumElems, 1, 1);
            verifyResults("step 3");
        }
    }

    GPU_TEST(RootBufferSRV_5_1) { testRootBuffer(ctx, "5_1", false); }
    GPU_TEST(RootBufferUAV_5_1) { testRootBuffer(ctx, "5_1", true); }

    GPU_TEST(RootBufferSRV_6_0) { testRootBuffer(ctx, "6_0", false); }
    GPU_TEST(RootBufferUAV_6_0) { testRootBuffer(ctx, "6_0", true); }

    GPU_TEST(RootBufferSRV_6_3) { testRootBuffer(ctx, "6_3", false); }
    GPU_TEST(RootBufferUAV_6_3) { testRootBuffer(ctx, "6_3", true); }
}
