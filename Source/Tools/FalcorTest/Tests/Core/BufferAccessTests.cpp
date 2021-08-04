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
        const uint32_t elems = 256;

        /** Create buffer with the given CPU access and elements initialized to 0,1,2,...
        */
        Buffer::SharedPtr createTestBuffer(Buffer::CpuAccess cpuAccess, bool initialize = true)
        {
            std::vector<uint32_t> initData(elems);
            for (uint32_t i = 0; i < elems; i++) initData[i] = i;
            return Buffer::create(elems * sizeof(uint32_t), Resource::BindFlags::ShaderResource, cpuAccess, initialize ? initData.data() : nullptr);
        }

        /** Tests readback from buffer created with the given CPU access flag work.
            The test binds the buffer to a compute program which reads back the data.
        */
        void testBufferReadback(GPUUnitTestContext& ctx, Buffer::CpuAccess cpuAccess)
        {
            auto pBuf = createTestBuffer(cpuAccess);

            // Run program that copies the buffer elements into result buffer.
            ctx.createProgram("Tests/Core/BufferAccessTests.cs.slang", "readback", Program::DefineList(), Shader::CompilerFlags::None);
            ctx.allocateStructuredBuffer("result", elems);
            ctx["buffer"] = pBuf;
            ctx.runProgram(elems, 1, 1);

            const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
            for (uint32_t i = 0; i < elems; i++)
            {
                EXPECT_EQ(result[i], i) << "i = " << i;
            }
            ctx.unmapBuffer("result");
        }
    }

    /** Test that initialization of buffer with CPU write access works.
        The test copies the data to a staging buffer on the GPU.
    */
    GPU_TEST(CopyBufferCpuAccessWrite)
    {
        auto pBuf = createTestBuffer(Buffer::CpuAccess::Write);

        // Copy buffer to staging buffer on the GPU.
        // Note we have to use copyBufferRegion() as our buffer is allocated within a page on the upload heap.
        auto pStaging = Buffer::create(elems * sizeof(uint32_t), Resource::BindFlags::None, Buffer::CpuAccess::Read);
        ctx.getRenderContext()->copyBufferRegion(pStaging.get(), 0, pBuf.get(), 0, elems * sizeof(uint32_t));

        // We have to flush here so that the copy is guaranteed to have finished by the time we map.
        // In user code, we normally want to use a GpuFence to signal this instead of a full flush.
        ctx.getRenderContext()->flush(true);

        const uint32_t* result = static_cast<const uint32_t*>(pStaging->map(Buffer::MapType::Read));
        for (uint32_t i = 0; i < elems; i++)
        {
            EXPECT_EQ(result[i], i) << "i = " << i;
        }
        pStaging->unmap();
    }

    // This test is disabled due to bug in creation of SRV/UAVs for buffers on the upload heap.

    /** Test setBlob() into buffer with CPU write access.
    */
    GPU_TEST(SetBlobBufferCpuAccessWrite, "Disabled due to issue with SRV/UAVs for resources on the upload heap (#638)")
    {
        auto pBuf = createTestBuffer(Buffer::CpuAccess::Write, false);

        // Set data into buffer using its setBlob() function.
        std::vector<uint32_t> initData(elems);
        for (uint32_t i = 0; i < elems; i++) initData[i] = i;
        pBuf->setBlob(initData.data(), 0, elems * sizeof(uint32_t));

        // Run program that copies the buffer elements into result buffer.
        ctx.createProgram("Tests/Core/BufferAccessTests.cs.slang", "readback", Program::DefineList(), Shader::CompilerFlags::None);
        ctx.allocateStructuredBuffer("result", elems);
        ctx["buffer"] = pBuf;
        ctx.runProgram(elems, 1, 1);

        const uint32_t* result = ctx.mapBuffer<const uint32_t>("result");
        for (uint32_t i = 0; i < elems; i++)
        {
            EXPECT_EQ(result[i], i) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }

    /** Test that GPU reads from buffer created without CPU access works.
    */
    GPU_TEST(BufferCpuAccessNone)
    {
        testBufferReadback(ctx, Buffer::CpuAccess::None);
    }

    /** Test that GPU reads from buffer created with CPU read access works.
    */
    GPU_TEST(BufferCpuAccessRead)
    {
        testBufferReadback(ctx, Buffer::CpuAccess::Read);
    }

    // This test is disabled due to bug in creation of SRV/UAVs for buffers on the upload heap.

    /** Test that GPU reads from buffer created with CPU write access works.
    */
    GPU_TEST(BufferCpuAccessWrite, "Disabled due to issue with SRV/UAVs for resources on the upload heap (#638)")
    {
        testBufferReadback(ctx, Buffer::CpuAccess::Write);
    }
}
