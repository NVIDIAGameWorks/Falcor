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
        const uint32_t kNumElems = 256;
        std::mt19937 r;
        std::uniform_real_distribution u;
    }

    GPU_TEST(CastFloat16)
    {
        ctx.createProgram("Tests/Slang/CastFloat16.cs.slang", "testCastFloat16", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ctx.allocateStructuredBuffer("result", kNumElems);

        std::vector<uint16_t> elems(kNumElems * 2);
        for (auto& v : elems) v = f32tof16(float(u(r)));
        auto var = ctx.vars().getRootVar();
        var["data"] = Buffer::createStructured(var["data"], (uint32_t)elems.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, elems.data());

        ctx.runProgram(kNumElems, 1, 1);

        // Verify results.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        for (uint32_t i = 0; i < kNumElems; i++)
        {
            uint16_t ix = elems[2 * i];
            uint16_t iy = elems[2 * i + 1];
            uint32_t expected = (uint32_t(iy) << 16) | ix;
            EXPECT_EQ(result[i], expected) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }
}
