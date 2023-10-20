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
GPU_TEST(BufferAliasing_Read)
{
    ref<Device> pDevice = ctx.getDevice();

    const size_t N = 32;

    std::vector<float> initData(N);
    for (size_t i = 0; i < initData.size(); i++)
        initData[i] = (float)i;
    auto pBuffer =
        pDevice->createBuffer(initData.size() * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, initData.data());

    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testRead");
    ctx.allocateStructuredBuffer("result", N * 3);

    // Bind buffer to two separate vars to test resource aliasing.
    ctx["bufA1"] = pBuffer;
    ctx["bufA2"] = pBuffer;
    ctx["bufA3"] = pBuffer;

    ctx.runProgram(N, 1, 1);

    std::vector<float> result = ctx.readBuffer<float>("result");
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(result[i], (float)i) << "i = " << i;
        EXPECT_EQ(result[i + N], (float)i) << "i = " << i;
        EXPECT_EQ(result[i + 2 * N], (float)i) << "i = " << i;
    }
}

GPU_TEST(BufferAliasing_ReadWrite)
{
    ref<Device> pDevice = ctx.getDevice();

    const size_t N = 32;

    std::vector<float> initData(N * 3);
    for (size_t i = 0; i < initData.size(); i++)
        initData[i] = (float)i;
    auto pBuffer = pDevice->createBuffer(
        initData.size() * sizeof(float),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        initData.data()
    );

    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testReadWrite");

    // Bind buffer to two separate vars to test resource aliasing.
    ctx["bufB1"] = pBuffer;
    ctx["bufB2"] = pBuffer;
    ctx["bufB3"] = pBuffer;

    ctx.runProgram(N, 1, 1);

    std::vector<float> result = pBuffer->getElements<float>();
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(result[i], (float)(N - i)) << "i = " << i;
        EXPECT_EQ(result[i + N], (float)(N - i)) << "i = " << i;
        EXPECT_EQ(result[i + 2 * N], (float)(N - i)) << "i = " << i;
    }
}

GPU_TEST(BufferAliasing_StructRead, "Disabled because <uint> version fails")
{
    ref<Device> pDevice = ctx.getDevice();

    const size_t N = 32;

    std::vector<float> initData(N);
    for (size_t i = 0; i < initData.size(); i++)
        initData[i] = (float)i;
    auto pBuffer = pDevice->createStructuredBuffer(
        initData.size() * sizeof(float), 1, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, initData.data(), false
    );

    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testStructRead");
    ctx.allocateStructuredBuffer("result", N * 3);

    // Bind buffer to three separate vars to test resource aliasing.
    ctx["bufStruct1"] = pBuffer;
    ctx["bufStruct2"] = pBuffer;
    ctx["bufStruct3"] = pBuffer;

    ctx.runProgram(N, 1, 1);

    std::vector<float> result = ctx.readBuffer<float>("result");
    for (size_t i = 0; i < N; i++)
    {
        EXPECT_EQ(result[i], (float)i) << "i = " << i;
        EXPECT_EQ(result[i + N], (float)i) << "i = " << i;
        EXPECT_EQ(result[i + 2 * N], (float)i) << "i = " << i;
    }
}
} // namespace Falcor
