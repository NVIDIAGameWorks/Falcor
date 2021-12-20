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
    namespace
    {
        void testRayFlags(GPUUnitTestContext& ctx, bool useDXR_1_1)
        {
            std::vector<uint32_t> expected =
            {
                (uint32_t)RayFlags::None,
                (uint32_t)RayFlags::ForceOpaque,
                (uint32_t)RayFlags::ForceNonOpaque,
                (uint32_t)RayFlags::AcceptFirstHitAndEndSearch,
                (uint32_t)RayFlags::SkipClosestHitShader,
                (uint32_t)RayFlags::CullBackFacingTriangles,
                (uint32_t)RayFlags::CullFrontFacingTriangles,
                (uint32_t)RayFlags::CullOpaque,
                (uint32_t)RayFlags::CullNonOpaque
            };

            Program::DefineList defines;
            std::string shaderModel = "6_3";

            if (useDXR_1_1)
            {
                expected.push_back((uint32_t)RayFlags::SkipTriangles);
                expected.push_back((uint32_t)RayFlags::SkipProceduralPrimitives);
                defines.add("DXR_1_1");
                shaderModel = "6_5";
            }

            ctx.createProgram("Tests/Slang/TraceRayFlags.cs.slang", "testRayFlags", defines, Shader::CompilerFlags::None, shaderModel);
            ctx.allocateStructuredBuffer("result", (uint32_t)expected.size());
            ctx.runProgram(1, 1, 1);

            const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
            for (size_t i = 0; i < expected.size(); ++i)
            {
                EXPECT_EQ(result[i], expected[i]);
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(TraceRayFlagsDXR1_0)
    {
        testRayFlags(ctx, false);
    }

    GPU_TEST(TraceRayFlagsDXR1_1)
    {
        testRayFlags(ctx, true);
    }
}
