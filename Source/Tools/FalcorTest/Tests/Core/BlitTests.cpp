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

namespace Falcor
{
    namespace
    {
        template<class T>
        void testBlit(GPUUnitTestContext& ctx, const uint2 srcDim, const uint32_t scale)
        {
            FALCOR_ASSERT(scale == 1 || scale == 2);
            const uint2 dstDim = srcDim / uint2(scale);
            const uint32_t dstElemes = dstDim.x * dstDim.y * 4;

            // Setup test data. This is either float values in [0,1) or integers in [0,2^32) depending on format.
            std::mt19937 rng;
            auto dist = std::uniform_real_distribution<float>();
            auto r = [&]() { if constexpr (std::is_same_v<T, float>) return dist(rng); else return rng(); };

            std::vector<T> srcData(srcDim.x * srcDim.y * 4);
            for (auto& v : srcData) v = r();

            auto load = [&](uint32_t x, uint32_t y, uint32_t i) -> T { return srcData[(y * srcDim.x + x) * 4 + i]; };

            // Compute filtered reference.
            std::vector<T> dstData(dstElemes);
            uint32_t idx = 0;
            for (uint32_t y = 0; y < dstDim.y; y++)
            {
                for (uint32_t x = 0; x < dstDim.x; x++)
                {
                    for (uint32_t i = 0; i < 4; i++)
                    {
                        if (scale == 1)
                        {
                            dstData[idx] = srcData[idx];
                        }
                        else if (scale == 2)
                        {
                            T a = load(2 * x, 2 * y, i);
                            T b = load(2 * x + 1, 2 * y, i);
                            T c = load(2 * x, 2 * y + 1, i);
                            T d = load(2 * x + 1, 2 * y + 1, i);

                            T result;
                            if constexpr (std::is_same_v<T, float>) result = (float)(((double)a + (double)b + (double)c + (double)d) / 4.0);
                            else result = (uint32_t)(((uint64_t)a + (uint64_t)b + (uint64_t)c + (uint64_t)d) / 4);

                            dstData[idx] = result;
                        }
                        idx++;
                    }
                }
            }

            // Create textures and perform blit.
            ResourceFormat format = std::is_same_v<T, float> ? ResourceFormat::RGBA32Float : ResourceFormat::RGBA32Uint;
            auto pSrc = Texture::create2D(srcDim.x, srcDim.y, format, 1, 1, srcData.data(), ResourceBindFlags::ShaderResource);
            auto pDst = Texture::create2D(dstDim.x, dstDim.y, format, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

            ctx.getRenderContext()->blit(pSrc->getSRV(), pDst->getRTV());

            // Run program to copy resulting texels into readback buffer.
            Program::DefineList defines = { {"FLOAT_FORMAT", std::is_same_v<T, float> ? "1" : "0" } };
            ctx.createProgram("Tests/Core/BlitTests.cs.slang", "readback", defines, Shader::CompilerFlags::None);
            ctx.allocateStructuredBuffer("result", dstElemes);
            ctx["tex"] = pDst;
            ctx["CB"]["sz"] = dstDim;
            ctx.runProgram(dstDim.x, dstDim.y, 1);

            const T* result = ctx.mapBuffer<const T>("result");
            for (uint32_t i = 0; i < dstElemes; i++)
            {
                if constexpr (std::is_same_v<T, float>)
                {
                    EXPECT_LE(std::abs(result[i] - dstData[i]), 1e-6f) << "i = " << i;
                }
                else
                {
                    EXPECT_EQ(result[i], dstData[i]) << "i = " << i;
                }
            }
            ctx.unmapBuffer("result");
        }
    }

    GPU_TEST(BlitFloatNoFilter)
    {
        testBlit<float>(ctx, uint2(33, 63), 1);
    }

    GPU_TEST(BlitFloatFilter)
    {
        testBlit<float>(ctx, uint2(32, 64), 2);
    }

    GPU_TEST(BlitUintNoFilter)
    {
        testBlit<uint32_t>(ctx, uint2(33, 63), 1);
    }

    GPU_TEST(BlitUintFilter)
    {
        testBlit<uint32_t>(ctx, uint2(32, 64), 2);
    }
}
