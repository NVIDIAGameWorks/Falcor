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
#include "Utils/HostDeviceShared.slangh"

namespace Falcor
{
    /** GPU test for user-allocated constant buffers.
    */
    GPU_TEST(UserConstantBuffer)
    {
        ctx.createProgram("Tests/Core/UserConstantBufferTests.cs.slang", "test", Program::DefineList(), Shader::CompilerFlags::None);
        ctx.allocateStructuredBuffer("result", 3);

        // Set some values into the builtin constant buffer that is automatically allocated.
        // We will bind our user-allocated parameter block below which should override these.
        ctx["params"]["a"] = 1;
        ctx["params"]["b"] = 3;
        ctx["params"]["c"] = 5.5f;

        // Creating and binding a constant buffer manually is expected to fail. It should give a hard error.
        // TODO: We should add a test for this error condition if possible (i.e. catch exception).
        //auto pBuf = Buffer::create(12, Resource::BindFlags::Constant, Buffer::CpuAccess::None, initData.data());
        //ctx["params"] = pBuf;

        // Create a parameter block instead to replace the automatically allocated block.
        //auto vars = ctx["params"]; // TODO: Can we get a ParameterBlockReflector from ShaderVar?
        auto reflector = ctx.getProgram()->getReflector()->getParameterBlock("params");
        auto pCB = ParameterBlock::create(reflector);
        ctx["params"] = pCB;

        pCB["a"] = 2;
        pCB["b"] = 11;
        pCB["c"] = 13.4f;

        // Bind the parameter block and run the program.
        // This is just to make sure the vars are applied and the underlying constant buffer created.
        // TODO: Add separate mechanism to do that.
        ctx.runProgram(1, 1, 1);

        // Test that the shader saw the correct values in the CB.
        {
            const float* result = ctx.mapBuffer<const float>("result");
            EXPECT_EQ(result[0], 2);
            EXPECT_EQ(result[1], 11);
            EXPECT_EQ(result[2], 13.4f);
            ctx.unmapBuffer("result");
        }

        // Now, get the underlying buffer and populate it with some data from the outside.
        auto pBuf = pCB->getUnderlyingConstantBuffer();

        // First, inspect the current contents of the CB.
        // We'd expect to find the constants above in their native formats.
        {
            const uint32_t* data = static_cast<const uint32_t*>(pBuf->map(Buffer::MapType::Read));
            EXPECT_EQ(data[0], 2);
            EXPECT_EQ(data[1], 11);
            EXPECT_EQ(data[2], asuint(13.4f));
            pBuf->unmap();
        }

        // Now update the CB memory manually.
        // We flush before/after for safety's sake, but a real application should use GPU fences.
        {
            // Note that setBlob() maps the buffer causing it to be re-allocated,
            // which would screw up the book-keeping in ParameterBlock.
            // pBuf->setBlob(initData.data(), 0, 12);

            ctx.getRenderContext()->flush(true);

            std::vector<uint32_t> initData(3);
            initData[0] = 4;
            initData[1] = 17;
            initData[2] = asuint(19.1f);

            uint32_t* pDst = static_cast<uint32_t*>(pBuf->map(Buffer::MapType::Write));
            memcpy(pDst, initData.data(), 12);
            pBuf->unmap();

            ctx.getRenderContext()->flush(true);
        }

        // Inspect that the buffer was properly updated.
        {
            const uint32_t* data = static_cast<const uint32_t*>(pBuf->map(Buffer::MapType::Read));
            uint32_t v0 = data[0];
            uint32_t v1 = data[1];
            uint32_t v2 = data[2];

            EXPECT_EQ(data[0], 4);
            EXPECT_EQ(data[1], 17);
            EXPECT_EQ(data[2], asuint(19.1f));
            pBuf->unmap();
        }

        // Now the ParameterBlock's buffer should hold our constants.
        // But note that its CPU copy of the data will be out-of-date.
        // Just running the program should apply the vars, but should cause any updates.
        ctx.runProgram(1, 1, 1);

        // Test if we got what we expected.
        {
            const float* result = ctx.mapBuffer<const float>("result");
            EXPECT_EQ(result[0], 4);
            EXPECT_EQ(result[1], 17);
            EXPECT_EQ(result[2], 19.1f);
            ctx.unmapBuffer("result");
        }
    }
}
