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
#include "SlangReinterpretCast.cs.slang"
#include <random>

namespace Falcor
{
    namespace
    {
        static_assert(sizeof(Blob) == 48, "Unpexected size of struct Blob");
        static_assert(sizeof(A) == 48, "Unpexected size of struct A");
        static_assert(sizeof(B) == 48, "Unpexected size of struct B");
        static_assert(sizeof(C) == 48, "Unpexected size of struct C");
        static_assert(sizeof(D) == 48, "Unpexected size of struct D");
        static_assert(sizeof(E) == 48, "Unpexected size of struct E");
        static_assert(sizeof(F) == 48, "Unpexected size of struct F");

        const uint32_t kElems = 128;
    }

    GPU_TEST(SlangReinterpretCast)
    {
        ctx.createProgram("Tests/Slang/SlangReinterpretCast.cs.slang", "main", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ctx.allocateStructuredBuffer("resultA", kElems);
        ctx.allocateStructuredBuffer("resultB", kElems);
        ctx.allocateStructuredBuffer("resultC", kElems);
        ctx.allocateStructuredBuffer("resultD", kElems);
        ctx.allocateStructuredBuffer("resultE", kElems);
        ctx.allocateStructuredBuffer("resultF", kElems);

        std::mt19937 r;
        std::uniform_real_distribution<float> u;

        std::vector<A> data(kElems);
        for (auto& v : data)
        {
            v.a = r();
            v.b = u(r);
            v.c = float16_t(u(r));
            v.d = int16_t(r());
            v.e = { r(), r() };
            v.f = { u(r), u(r), u(r) };
            v.g = r();
            v.h = { float16_t(u(r)), float16_t(u(r)), float16_t(u(r)) };
            v.i = uint16_t(r());
            v.j = { float16_t(u(r)), float16_t(u(r)) };
        }

        ShaderVar var = ctx.vars().getRootVar();
        var["data"] = Buffer::createStructured(sizeof(A), kElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, data.data());

        ctx.runProgram(kElems);

        // Verify final result matches our input.
        const A* result = ctx.mapBuffer<const A>("resultA");
        for (size_t i = 0; i < data.size(); i++)
        {
            EXPECT_EQ(result[i].a, data[i].a);
            EXPECT_EQ(result[i].b, data[i].b);
            EXPECT_EQ((float)result[i].c, (float)data[i].c);
            EXPECT_EQ(result[i].d, data[i].d);
            EXPECT_EQ(result[i].e.x, data[i].e.x);
            EXPECT_EQ(result[i].e.y, data[i].e.y);
            EXPECT_EQ((float)result[i].f.x, (float)data[i].f.x);
            EXPECT_EQ((float)result[i].f.y, (float)data[i].f.y);
            EXPECT_EQ((float)result[i].f.z, (float)data[i].f.z);
            EXPECT_EQ(result[i].g, data[i].g);
            EXPECT_EQ((float)result[i].h.x, (float)data[i].h.x);
            EXPECT_EQ((float)result[i].h.y, (float)data[i].h.y);
            EXPECT_EQ((float)result[i].h.z, (float)data[i].h.z);
            EXPECT_EQ(result[i].i, data[i].i);
            EXPECT_EQ((float)result[i].j.x, (float)data[i].j.x);
            EXPECT_EQ((float)result[i].j.y, (float)data[i].j.y);
        }
        ctx.unmapBuffer("resultA");

        // Verify the intermediate results. We'll just do a binary comparison for simplicity.
        auto verify = [&](const char* bufferName)
        {
            const uint32_t* result = ctx.mapBuffer<const uint32_t>(bufferName);
            const uint32_t* rawData = reinterpret_cast<const uint32_t*>(data.data());
            for (size_t i = 0; i < data.size() * sizeof(data[0]) / 4; i++)
            {
                EXPECT_EQ(result[i], rawData[i]) << "i = " << i << " buffer " << bufferName;
            }
            ctx.unmapBuffer(bufferName);
        };

        verify("resultA");
        verify("resultB");
        verify("resultC");
        verify("resultD");
        verify("resultE");
        verify("resultF");
    }
}
