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

namespace Falcor
{
namespace
{
void runTest(GPUUnitTestContext& ctx, const std::string& entryPoint, DefineList defines)
{
    ref<Device> pDevice = ctx.getDevice();

    ProgramDesc desc;
    desc.addShaderLibrary("Tests/Slang/SlangGenerics.cs.slang").csEntry(entryPoint);
    ctx.createProgram(desc, defines);
    ctx.allocateStructuredBuffer("result", 128);

    // Run program.
    ctx.runProgram(32, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < 32; i++)
    {
        EXPECT_EQ(result[4 * i + 0], (i + 0) * 12);
        EXPECT_EQ(result[4 * i + 1], (i + 1) * 12);
        EXPECT_EQ(result[4 * i + 2], (i + 2) * 12);
        EXPECT_EQ(result[4 * i + 3], (i + 3) * 12);
    }
}
} // namespace

GPU_TEST(Slang_GenericsInterface_Int)
{
    runTest(ctx, "testGenericsInterface", DefineList{{"TEST_A", "1"}, {"USE_INT", "1"}});
}

GPU_TEST(Slang_GenericsInterface_UInt)
{
    runTest(ctx, "testGenericsInterface", DefineList{{"TEST_A", "1"}});
}

GPU_TEST(Slang_GenericsFunction_Int)
{
    runTest(ctx, "testGenericsFunction", DefineList{{"TEST_B", "1"}, {"USE_INT", "1"}});
}

GPU_TEST(Slang_GenericsFunction_UInt)
{
    runTest(ctx, "testGenericsFunction", DefineList{{"TEST_B", "1"}});
}
} // namespace Falcor
