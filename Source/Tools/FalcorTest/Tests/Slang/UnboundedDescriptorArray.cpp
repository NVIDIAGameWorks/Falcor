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

namespace Falcor
{
    GPU_TEST(UnboundedDescriptorArray, "Unbounded arrays are not yet supported")
    {
        const uint32_t kTexCount = 4;

        ctx.createProgram("Tests/Slang/UnboundedDescriptorArray.cs.slang", "main", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ctx.allocateStructuredBuffer("result", kTexCount);

        auto var = ctx.vars().getRootVar()["resources"];
        for (size_t i = 0; i < kTexCount; i++)
        {
            float initData = (float)(i + 1);
            var["textures"][i] = Texture::create2D(1, 1, ResourceFormat::R32Float, 1, 1, &initData);
        }

        ctx.runProgram(kTexCount, 1, 1);

        const float* result = ctx.mapBuffer<const float>("result");
        for (size_t i = 0; i < kTexCount; i++)
        {
            float expected = (float)(i + 1);
            EXPECT_EQ(result[i], expected) << "i = " << i;

        }
        ctx.unmapBuffer("result");
    }
}
