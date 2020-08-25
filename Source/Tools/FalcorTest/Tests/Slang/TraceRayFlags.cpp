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
        uint32_t getRayFlags1_0()
        {
            return D3D12_RAY_FLAG_NONE |
                D3D12_RAY_FLAG_FORCE_OPAQUE |
                D3D12_RAY_FLAG_FORCE_NON_OPAQUE |
                D3D12_RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                D3D12_RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |
                D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES |
                D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES |
                D3D12_RAY_FLAG_CULL_OPAQUE |
                D3D12_RAY_FLAG_CULL_NON_OPAQUE;
        }
#if 0
        uint32_t getRayFlags1_1()
        {
            return getRayFlags1_0() |
                D3D12_RAY_FLAG_SKIP_TRIANGLES |
                D3D12_RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
        }
#endif
        void testRayFlags(GPUUnitTestContext& ctx, uint32_t expected, const Program::DefineList& defines, const std::string& shaderModel)
        {
            ctx.createProgram("Tests/Slang/TraceRayFlags.cs.slang", "testRayFlags", defines, Shader::CompilerFlags::None, shaderModel);
            ctx.allocateStructuredBuffer("result", 1);
            ctx.runProgram(1, 1, 1);

            const uint32_t result = *ctx.mapBuffer<const uint32_t>("result");
            EXPECT_EQ(result, expected);
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(TraceRayFlagsDXR1_0)
    {
        testRayFlags(ctx, getRayFlags1_0(), {}, "6_3");
    }
#if 0
    GPU_TEST(TraceRayFlagsDXR1_1, "Requires shader model 6.5")
    {
        testRayFlags(ctx, getRayFlags1_1(), { { "DXR_1_1", ""} }, "6_5");
    }
#endif
}
