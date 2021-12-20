/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "SlangInheritance.cs.slang"

namespace Falcor
{
    GPU_TEST(SlangStructInheritanceReflection, "Not working yet")
    {
        ctx.createProgram("Tests/Slang/SlangInheritance.cs.slang", "main", Program::DefineList(), Shader::CompilerFlags::None, "6_5");

        // Reflection of struct A.
        auto typeA = ctx.getProgram()->getReflector()->findType("A");
        EXPECT(typeA != nullptr);
        if (typeA)
        {
            EXPECT_EQ(typeA->getByteSize(), 4);

            auto scalarA = typeA->findMember("scalar");
            EXPECT(scalarA != nullptr);
            if (scalarA)
            {
                EXPECT_EQ(scalarA->getByteOffset(), 0);
                EXPECT_EQ(scalarA->getType()->getByteSize(), 4);
            }
        }

        // Reflection of struct B inheriting from A
        // Expect A's members to be placed first before B's.
        auto typeB = ctx.getProgram()->getReflector()->findType("B");
        EXPECT(typeB != nullptr);
        if (typeB)
        {
            EXPECT_EQ(typeB->getByteSize(), 16);

            auto scalarB = typeB->findMember("scalar");
            EXPECT(scalarB != nullptr);
            if (scalarB)
            {
                EXPECT_EQ(scalarB->getByteOffset(), 0);
                EXPECT_EQ(scalarB->getType()->getByteSize(), 4);
            }

            auto vectorB = typeB->findMember("vector");
            EXPECT(vectorB != nullptr);
            if (vectorB)
            {
                EXPECT_EQ(vectorB->getByteOffset(), 4);
                EXPECT_EQ(vectorB->getType()->getByteSize(), 12);
            }
        }
    }

    GPU_TEST(SlangStructInheritanceLayout)
    {
        ctx.createProgram("Tests/Slang/SlangInheritance.cs.slang", "main", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ShaderVar var = ctx.vars().getRootVar();

        // TODO: Use built-in buffer when reflection of struct inheritance works (see #1306).
        //ctx.allocateStructuredBuffer("result", 1);
        auto pResult = Buffer::createStructured(16, 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        var["result"] = pResult;

        std::vector<uint32_t> initData(4);
        initData[0] = 59941431;
        initData[1] = asuint(3.13f);
        initData[2] = asuint(5.11f);
        initData[3] = asuint(7.99f);

        var["data"] = Buffer::createTyped<uint>((uint32_t)initData.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, initData.data());

        ctx.runProgram();

        // Check struct layout on the host.
        // Expect A's members to be placed first before B's.
        EXPECT_EQ(sizeof(B), 16);
        EXPECT_EQ(sizeof(B::scalar), 4);
        EXPECT_EQ(sizeof(B::vector), 12);
        EXPECT_EQ(offsetof(B, scalar), 0);
        EXPECT_EQ(offsetof(B, vector), 4);

        //const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        const uint32_t* result = (const uint32_t*)pResult->map(Buffer::MapType::Read);

        // Check struct fields read back from the GPU.
        // Slang uses the same struct layout as the host.
        EXPECT_EQ(result[0], initData[0]);
        EXPECT_EQ(result[1], initData[1]);
        EXPECT_EQ(result[2], initData[2]);
        EXPECT_EQ(result[3], initData[3]);

        //ctx.unmapBuffer("result");
        pResult->unmap();
    }
}
