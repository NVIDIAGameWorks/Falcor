/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
GPU_TEST(BufferAliasingRead)
{
    Device* pDevice = ctx.getDevice().get();

    const size_t N = 32;

    std::vector<float> initData(N);
    for (size_t i = 0; i < initData.size(); i++)
        initData[i] = (float)i;
    auto pBuffer = Buffer::create(
        pDevice, initData.size() * sizeof(float), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, initData.data()
    );

    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testRead", Program::DefineList(), Shader::CompilerFlags::None);
    ctx.allocateStructuredBuffer("result", N * 2);

    // Bind buffer to two separate vars to test resource aliasing.
    ctx["bufA1"] = pBuffer;
    ctx["bufA2"] = pBuffer;

    ctx.runProgram(N, 1, 1);

    const float* result = ctx.mapBuffer<const float>("result");
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(result[i], (float)i) << "i = " << i;
        EXPECT_EQ(result[i + N], (float)i) << "i = " << i;
    }
    ctx.unmapBuffer("result");
}

GPU_TEST(BufferAliasingReadWrite)
{
    Device* pDevice = ctx.getDevice().get();

    const size_t N = 32;

    std::vector<float> initData(N * 2);
    for (size_t i = 0; i < initData.size(); i++)
        initData[i] = (float)i;
    auto pBuffer = Buffer::create(
        pDevice, initData.size() * sizeof(float), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
        Buffer::CpuAccess::None, initData.data()
    );

    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testReadWrite", Program::DefineList(), Shader::CompilerFlags::None);

    // Bind buffer to two separate vars to test resource aliasing.
    ctx["bufB1"] = pBuffer;
    ctx["bufB2"] = pBuffer;

    ctx.runProgram(N, 1, 1);

    const float* result = reinterpret_cast<const float*>(pBuffer->map(Buffer::MapType::Read));
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(result[i], (float)(N - i)) << "i = " << i;
        EXPECT_EQ(result[i + N], (float)(N - i)) << "i = " << i;
    }
    pBuffer->unmap();
}
} // namespace Falcor
