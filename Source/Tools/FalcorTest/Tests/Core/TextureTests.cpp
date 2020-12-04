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
    /** GPU test for reading/writing a 3D texture.
    */
    GPU_TEST(RWTexture3D)
    {
        auto pTex = Texture::create3D(16, 16, 16, ResourceFormat::R32Uint, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        EXPECT(pTex);

        // Step 1: Write to 3D texture bound as UAV.
        ctx.createProgram("Tests/Core/TextureTests.cs.slang", "testTexture3DWrite");
        ctx["tex3D_uav"] = pTex;
        ctx.runProgram(16, 16, 16);

        // Step 2: Read from 3D texture bound as SRV.
        ctx.createProgram("Tests/Core/TextureTests.cs.slang", "testTexture3DRead");
        ctx.allocateStructuredBuffer("result", 4096);
        ctx["tex3D_srv"] = pTex;
        ctx.runProgram(16, 16, 16);

        // Verify result.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        size_t i = 0;
        for (uint32_t z = 0; z < 16; z++)
        {
            for (uint32_t y = 0; y < 16; y++)
            {
                for (uint32_t x = 0; x < 16; x++)
                {
                    EXPECT_EQ(result[i], x * y * z + 577) << "i = " << i++;
                }
            }
        }
        ctx.unmapBuffer("result");
    }
}
