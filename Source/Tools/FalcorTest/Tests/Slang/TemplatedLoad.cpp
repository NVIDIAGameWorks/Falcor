/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/HostDeviceShared.slangh"
#include <random>

namespace Falcor
{
    namespace
    {
        std::mt19937 r;
        std::uniform_real_distribution u;

        std::vector<uint16_t> generateData(const size_t n)
        {
            std::vector<uint16_t> elems;
            for (size_t i = 0; i < n; i++) elems.push_back((uint16_t)f32tof16(float(u(r))));
            return elems;
        }

        void test(GPUUnitTestContext& ctx, const std::string& entryPoint, const size_t n)
        {
            std::vector<uint16_t> elems = generateData(n);

            ctx.createProgram("Tests/Slang/TemplatedLoad.cs.slang", entryPoint, Program::DefineList(), Shader::CompilerFlags::None, "6_5");
            ctx.allocateStructuredBuffer("result", (uint32_t)elems.size());

            auto var = ctx.vars().getRootVar();
            var["data"] = Buffer::create(elems.size() * sizeof(elems[0]), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, elems.data());

            ctx.runProgram(1, 1, 1);

            // Verify results.
            const uint16_t* result = ctx.mapBuffer<const uint16_t>("result");
            for (size_t i = 0; i < elems.size(); i++)
            {
                EXPECT_EQ(result[i], elems[i]) << "i = " << i;
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(TemplatedScalarLoad16)
    {
        test(ctx, "testTemplatedScalarLoad16", 20);
    }

    GPU_TEST(TemplatedVectorLoad16)
    {
        test(ctx, "testTemplatedVectorLoad16", 20);
    }

    GPU_TEST(TemplatedMatrixLoad16_2x4)
    {
        test(ctx, "testTemplatedMatrixLoad16_2x4", 8);
    }

    GPU_TEST(TemplatedMatrixLoad16_4x3)
    {
        test(ctx, "testTemplatedMatrixLoad16_4x3", 12);
    }
}
