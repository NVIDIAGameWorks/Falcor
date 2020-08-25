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
        const uint32_t kNumElems = 256;

        void test(GPUUnitTestContext& ctx, const std::string& shaderModel)
        {
            ctx.createProgram("Tests/Slang/ShaderModel.cs.slang", "main", Program::DefineList(), Shader::CompilerFlags::None, shaderModel);
            ctx.allocateStructuredBuffer("result", kNumElems);
            ctx.runProgram(kNumElems, 1, 1);

            const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
            for (uint32_t i = 0; i < kNumElems; i++)
            {
                EXPECT_EQ(result[i], 3 * i);
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(ShaderModel5_0) { test(ctx, "5_0"); }
    GPU_TEST(ShaderModel5_1) { test(ctx, "5_1"); }
    GPU_TEST(ShaderModel6_0) { test(ctx, "6_0"); }
    GPU_TEST(ShaderModel6_1) { test(ctx, "6_1"); }
    GPU_TEST(ShaderModel6_2) { test(ctx, "6_2"); }
    GPU_TEST(ShaderModel6_3) { test(ctx, "6_3"); }
    GPU_TEST(ShaderModel6_4, "Requires shader model 6.4") { test(ctx, "6_4"); }
    GPU_TEST(ShaderModel6_5, "Requires shader model 6.5") { test(ctx, "6_5"); }
}
