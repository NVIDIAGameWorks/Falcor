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
#include "Falcor.h"
#include "Testing/UnitTest.h"

namespace Falcor
{
    namespace
    {
        void runTest2(GPUUnitTestContext& ctx, Program::DefineList defines)
        {
            ctx.createProgram("Tests/Slang/StructuredBufferMatrix.cs.slang", "testStructuredBufferMatrixLoad2", defines, Shader::CompilerFlags::DumpIntermediates, "6_5");
            ctx.allocateStructuredBuffer("result", 16);

            auto var = ctx.vars().getRootVar();
            auto pData = Buffer::createStructured(var["data2"], 1, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);

            EXPECT_EQ(pData->getElementCount(), 1);
            EXPECT_EQ(pData->getElementSize(), 32);

            std::vector<float16_t> initData(16);
            for (size_t i = 0; i < 16; i++) initData[i] = float16_t((float)i + 0.75f);
            pData->setBlob(initData.data(), 0, 32);

            var["data2"] = pData;

            ctx.runProgram(1, 1, 1);

            // Verify results.
            const float* result = ctx.mapBuffer<const float>("result");
            for (size_t i = 0; i < 16; i++)
            {
                EXPECT_EQ(result[i], (float)i + 0.75f) << "i = " << i;
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(StructuredBufferMatrixLoad1)
    {
        ctx.createProgram("Tests/Slang/StructuredBufferMatrix.cs.slang", "testStructuredBufferMatrixLoad1", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ctx.allocateStructuredBuffer("result", 32);

        auto var = ctx.vars().getRootVar();
        auto pData = Buffer::createStructured(var["data1"], 1, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);

        EXPECT_EQ(pData->getElementCount(), 1);
        EXPECT_EQ(pData->getElementSize(), 100);

        std::vector<uint8_t> initData(100);
        for (size_t i = 0; i < 18; i++) ((float*)initData.data())[i] = (float)i + 0.5f;
        for (size_t i = 0; i < 14; i++) ((float16_t*)(initData.data() + 72))[i] = float16_t((float)i + 18.5f);
        pData->setBlob(initData.data(), 0, 100);

        var["data1"] = pData;

        ctx.runProgram(1, 1, 1);

        // Verify results.
        const float* result = ctx.mapBuffer<const float>("result");
        for (size_t i = 0; i < 32; i++)
        {
            EXPECT_EQ(result[i], (float)i + 0.5f) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }

    GPU_TEST(StructuredBufferMatrixLoad2_1)
    {
        Program::DefineList defines = { {"LAYOUT", "1"} };
        runTest2(ctx, defines);
    }

    // TODO: Enable when https://github.com/microsoft/DirectXShaderCompiler/issues/4492 has been resolved.
    GPU_TEST(StructuredBufferMatrixLoad2_2, "Disabled due to compiler bug")
    {
        Program::DefineList defines = { {"LAYOUT", "2"} };
        runTest2(ctx, defines);
    }

    // TODO: Enable when https://github.com/microsoft/DirectXShaderCompiler/issues/4492 has been resolved.
    GPU_TEST(StructuredBufferMatrixLoad2_3, "Disabled due to compiler bug")
    {
        Program::DefineList defines = { {"LAYOUT", "3"} };
        runTest2(ctx, defines);
    }
}
