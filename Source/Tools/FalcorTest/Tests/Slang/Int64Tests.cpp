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

namespace Falcor
{
    namespace
    {
        std::vector<std::string> kShaderModels =
        {
            { "6_0" },
            { "6_1" },
            { "6_2" },
            { "6_3" },
        };

        const uint32_t kNumElems = 256;
        std::mt19937 r;

        void test(GPUUnitTestContext& ctx, const std::string& shaderModel, bool useUav)
        {
            Program::DefineList defines = { {"USE_UAV", useUav ? "1" : "0"} };

            ctx.createProgram("Tests/Slang/Int64Tests.cs.slang", "testInt64", defines, Shader::CompilerFlags::None, shaderModel);
            ctx.allocateStructuredBuffer("result", kNumElems * 2);

            std::vector<uint64_t> elems(kNumElems);
            for (auto& v : elems) v = ((uint64_t)r() << 32) | r();
            auto var = ctx.vars().getRootVar();
            auto pBuf = Buffer::createStructured(var["data"], kNumElems, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, elems.data());
            var["data"] = pBuf;

            ctx.runProgram(kNumElems, 1, 1);

            // Verify results.
            const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
            for (uint32_t i = 0; i < kNumElems; i++)
            {
                uint32_t lo = result[2 * i];
                uint32_t hi = result[2 * i + 1];
                uint64_t res = ((uint64_t)hi << 32) | lo;
                EXPECT_EQ(res, elems[i]) << "i = " << i << " shaderModel=" << shaderModel;
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(StructuredBufferLoadUInt64)
    {
        for (auto sm : kShaderModels) test(ctx, sm, false);
    }

    GPU_TEST(RWStructuredBufferLoadUInt64)
    {
        for (auto sm : kShaderModels) test(ctx, sm, true);
    }
}
