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
        const std::string kRootBufferName = "rootBuf";

        std::mt19937 rng;
        auto dist = std::uniform_int_distribution<uint32_t>(0, 100);
        auto r = [&]() -> uint32_t { return dist(rng); };

        void testRootBufferInStruct(GPUUnitTestContext& ctx, const std::string& shaderModel, bool useUav)
        {
            Program::DefineList defines = { {"USE_UAV", useUav ? "1" : "0"} };
            Shader::CompilerFlags compilerFlags = Shader::CompilerFlags::None; // Shader::CompilerFlags::DumpIntermediates;

            ctx.createProgram("Tests/Core/RootBufferStructTests.cs.slang", "main", defines, compilerFlags, shaderModel);
            ctx.allocateStructuredBuffer("result", kNumElems);

            auto data = ctx.vars().getRootVar()["CB"]["data"];

            // Bind some regular buffers.
            std::vector<uint32_t> buf(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) buf[i] = r();
                data["buf"] = Buffer::createTyped<uint32_t>(kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, buf.data());
            }
            std::vector<uint32_t> rwBuf(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) rwBuf[i] = r();
                data["rwBuf"] = Buffer::createTyped<uint32_t>(kNumElems, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, rwBuf.data());
            }

            // Test binding structured buffer to root descriptor inside struct in CB.
            std::vector<uint32_t> rootBuf(kNumElems);
            {
                for (uint32_t i = 0; i < kNumElems; i++) rootBuf[i] = r();

                auto pRootBuffer = Buffer::createStructured(
                    data[kRootBufferName],
                    kNumElems,
                    useUav ? ResourceBindFlags::UnorderedAccess : ResourceBindFlags::ShaderResource,
                    Buffer::CpuAccess::None,
                    rootBuf.data(),
                    false /* no UAV counter */);

                data[kRootBufferName] = pRootBuffer;

                Buffer::SharedPtr pBoundBuffer = data[kRootBufferName];
                EXPECT_EQ(pBoundBuffer, pRootBuffer);
            }

            // Run the program to test that we can access the buffer.
            ctx.runProgram(kNumElems, 1, 1);

            const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
            for (uint32_t i = 0; i < kNumElems; i++)
            {
                uint32_t r = 0;
                r += buf[i];
                r += rwBuf[i] * 2;
                r += rootBuf[i] * 3;
                EXPECT_EQ(result[i], r) << "i = " << i;
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(RootBufferStructSRV_5_1) { testRootBufferInStruct(ctx, "5_1", false); }
    GPU_TEST(RootBufferStructUAV_5_1) { testRootBufferInStruct(ctx, "5_1", true); }

    GPU_TEST(RootBufferStructSRV_6_0) { testRootBufferInStruct(ctx, "6_0", false); }
    GPU_TEST(RootBufferStructUAV_6_0) { testRootBufferInStruct(ctx, "6_0", true); }

    GPU_TEST(RootBufferStructSRV_6_3) { testRootBufferInStruct(ctx, "6_3", false); }
    GPU_TEST(RootBufferStructUAV_6_3) { testRootBufferInStruct(ctx, "6_3", true); }
}
