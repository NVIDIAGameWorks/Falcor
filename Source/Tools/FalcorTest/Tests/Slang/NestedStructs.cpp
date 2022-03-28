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
#include "SlangInheritance.cs.slang"

namespace Falcor
{
    /** Test nested structs in constant buffers.
        This makes sure Slang reflection is accurate and that assign-by-name
        works correctly for all basic types without manually added padding.
    */
    GPU_TEST(NestedStructs)
    {
        ctx.createProgram("Tests/Slang/NestedStructs.cs.slang", "main");
        ctx.allocateStructuredBuffer("result", 27);

        ShaderVar var = ctx.vars().getRootVar()["CB"];

        var["a"] = 1.1f;
        var["s3"]["a"] = 17;
        var["s3"]["b"] = true;
        var["s3"]["s2"]["a"] = bool3(true, false, true);
        var["s3"]["s2"]["s1"]["a"] = float2(9.3f, 2.1f);
        var["s3"]["s2"]["s1"]["b"] = 23;
        var["s3"]["s2"]["b"] = 0.99f;
        var["s3"]["s2"]["c"] = uint2(4, 8);
        var["s3"]["c"] = float3(0.1f, 0.2f, 0.3f);
        var["s3"]["s1"]["a"] = float2(1.88f, 1.99f);
        var["s3"]["s1"]["b"] = 711;
        var["s2"]["a"] = bool3(false, true, false);
        var["s2"]["s1"]["a"] = float2(0.55f, 8.31f);
        var["s2"]["s1"]["b"] = 431;
        var["s2"]["b"] = 1.65f;
        var["s2"]["c"] = uint2(7, 3);

        ctx.runProgram();

        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");

        EXPECT_EQ(result[0], asuint(1.1f));
        EXPECT_EQ(result[1], 17);
        EXPECT_EQ(result[2], 1);
        EXPECT_EQ(result[3], 1);
        EXPECT_EQ(result[4], 0);
        EXPECT_EQ(result[5], 1);
        EXPECT_EQ(result[6], asuint(9.3f));
        EXPECT_EQ(result[7], asuint(2.1f));
        EXPECT_EQ(result[8], 23);
        EXPECT_EQ(result[9], asuint(0.99f));
        EXPECT_EQ(result[10], 4);
        EXPECT_EQ(result[11], 8);
        EXPECT_EQ(result[12], asuint(0.1f));
        EXPECT_EQ(result[13], asuint(0.2f));
        EXPECT_EQ(result[14], asuint(0.3f));
        EXPECT_EQ(result[15], asuint(1.88f));
        EXPECT_EQ(result[16], asuint(1.99f));
        EXPECT_EQ(result[17], 711);
        EXPECT_EQ(result[18], 0);
        EXPECT_EQ(result[19], 1);
        EXPECT_EQ(result[20], 0);
        EXPECT_EQ(result[21], asuint(0.55f));
        EXPECT_EQ(result[22], asuint(8.31f));
        EXPECT_EQ(result[23], 431);
        EXPECT_EQ(result[24], asuint(1.65f));
        EXPECT_EQ(result[25], 7);
        EXPECT_EQ(result[26], 3);

        ctx.unmapBuffer("result");
    }
}
