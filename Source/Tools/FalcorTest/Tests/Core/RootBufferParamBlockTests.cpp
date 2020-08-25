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
        const std::string kReflectionProgram = "Tests/Core/ParamBlockReflection.cs.slang";
        const std::string kTestProgram = "Tests/Core/RootBufferParamBlockTests.cs.slang";

        const uint32_t kNumElems = 256;
        const std::string kRootBufferName = "testBuffer";
        const std::string kGlobalRootBufferName = "globalTestBuffer";

        std::mt19937 rng;
        auto dist = std::uniform_int_distribution<uint32_t>(0, 100);
        auto r = [&]() -> uint32_t { return dist(rng); };

        void testRootBuffer(GPUUnitTestContext& ctx, const std::string& shaderModel, bool useUav)
        {
            Program::DefineList defines = { {"USE_UAV", useUav ? "1" : "0"} };
            Shader::CompilerFlags compilerFlags = Shader::CompilerFlags::None; // Shader::CompilerFlags::DumpIntermediates;

            // Create parameter block based on reflection of a dummy program.
            // This is to ensure that the register index/space here do not match those of the final program.
            Program::Desc reflDesc;
            reflDesc.addShaderLibrary(kReflectionProgram).csEntry("main");
            reflDesc.setShaderModel("5_1"); // Note: Using 5.1 for the reflection, and the specified higher shading model for the actual test to make sure the reflection isn't affected.
            auto pReflectionProgram = ComputePass::create(reflDesc, defines);
            EXPECT(pReflectionProgram != nullptr);
            auto pBlockReflection = pReflectionProgram->getProgram()->getReflector()->getParameterBlock("gParamBlock");
            EXPECT(pBlockReflection != nullptr);
            auto pParamBlock = ParameterBlock::create(pBlockReflection);
            EXPECT(pParamBlock != nullptr);

            // Bind non-root resources to the parameter block.
            auto block = pParamBlock->getRootVar();
            float c0 = (float)r();
            block["c0"] = c0;

            std::vector<uint32_t> bufA[2];
            for (uint32_t j = 0; j < 2; j++)
            {
                bufA[j].resize(kNumElems);
                for (uint32_t i = 0; i < kNumElems; i++) bufA[j][i] = r();
                block["bufA"][j] = Buffer::create(kNumElems * sizeof(uint32_t), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, bufA[j].data());
            }
            std::vector<float> bufB[3];
            for (uint32_t j = 0; j < 3; j++)
            {
                bufB[j].resize(kNumElems);
                for (uint32_t i = 0; i < kNumElems; i++) bufB[j][i] = (float)r();
                block["bufB"][j] = Buffer::createTyped<float>(kNumElems, Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, bufB[j].data());
            }
            std::vector<uint32_t> bufC[4];
            for (uint32_t j = 0; j < 4; j++)
            {
                bufC[j].resize(kNumElems);
                for (uint32_t i = 0; i < kNumElems; i++) bufC[j][i] = r();
                block["bufC"][j] = Buffer::createTyped<uint32_t>(kNumElems, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, bufC[j].data());
            }

            // Bind root buffer to the parameter block.
            std::vector<uint32_t> testBuffer(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) testBuffer[i] = r();
                auto pTestBuffer = Buffer::create(kNumElems * sizeof(uint32_t), useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, testBuffer.data());
                bool ret = pParamBlock->setBuffer(kRootBufferName, pTestBuffer);
                EXPECT(ret);

                Buffer::SharedPtr pBoundBuffer = pParamBlock->getBuffer(kRootBufferName);
                EXPECT_EQ(pBoundBuffer, pTestBuffer);
            }

            // Create test program and bind the parameter block.
            ctx.createProgram(kTestProgram, "main", defines, compilerFlags, shaderModel);
            ctx.allocateStructuredBuffer("result", kNumElems);

            auto var = ctx.vars().getRootVar();
            var["gParamBlock"] = pParamBlock;

            // Bind some buffers at the global scope, both root and non-root resources.
            std::vector<uint32_t> globalBufA;
            {
                globalBufA.resize(kNumElems);
                for (uint32_t i = 0; i < kNumElems; i++) globalBufA[i] = r();
                var["globalBufA"] = Buffer::createTyped<uint32_t>(kNumElems, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, globalBufA.data());
            }
            std::vector<uint32_t> globalTestBuffer(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) globalTestBuffer[i] = r();
                var[kGlobalRootBufferName] = Buffer::create(kNumElems * sizeof(uint32_t), useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, globalTestBuffer.data());
            }

            // Test that reading from all the resources in the block works.
            ctx.runProgram(kNumElems, 1, 1);

            const float* result = ctx.mapBuffer<const float>("result");
            for (uint32_t i = 0; i < kNumElems; i++)
            {
                float r = 0.f;
                r += c0;
                r += bufA[0][i];
                r += bufA[1][i] * 2;
                r += bufB[0][i] * 3;
                r += bufB[1][i] * 4;
                r += bufB[2][i] * 5;
                r += bufC[0][i] * 6;
                r += bufC[1][i] * 7;
                r += bufC[2][i] * 8;
                r += bufC[3][i] * 9;
                r += testBuffer[i] * 10;
                r += globalBufA[i] * 11;
                r += globalTestBuffer[i] * 12;
                EXPECT_EQ(result[i], r) << "i = " << i;
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(RootBufferParamBlockSRV_5_1) { testRootBuffer(ctx, "5_1", false); }
    GPU_TEST(RootBufferParamBlockUAV_5_1) { testRootBuffer(ctx, "5_1", true); }

    GPU_TEST(RootBufferParamBlockSRV_6_0) { testRootBuffer(ctx, "6_0", false); }
    GPU_TEST(RootBufferParamBlockUAV_6_0) { testRootBuffer(ctx, "6_0", true); }

    GPU_TEST(RootBufferParamBlockSRV_6_3) { testRootBuffer(ctx, "6_3", false); }
    GPU_TEST(RootBufferParamBlockUAV_6_3) { testRootBuffer(ctx, "6_3", true); }
}
