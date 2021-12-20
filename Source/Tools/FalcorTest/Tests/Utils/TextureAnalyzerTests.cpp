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
#include "Utils/Image/TextureAnalyzer.h"

namespace Falcor
{
    namespace
    {
        const size_t kNumTests = 8;
        const size_t kNumPNGs = 6; // The rest are EXR
        const size_t kResultSize = TextureAnalyzer::getResultSize();

        const TextureAnalyzer::Result kExpectedResult[] =
        {
            // texture1.png
            {
                0x00011110, // Constant texture
                0, 0, 0,
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
            },
            // texture2.png
            {
                0x00011112, // G channel differs
                0, 0, 0,
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f,   0 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
            },
            // texture3.png
            {
                0x00011116, // BG channels differ
                0, 0, 0,
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f,   0 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f, 255 / 255.f, 255 / 255.f, 1.0f),
            },
            // texture4.png
            {
                0x00011117, // RGB channels differ
                0, 0, 0,
                float4(128 / 255.f, 255 / 255.f,  64 / 255.f, 1.0f),
                float4(128 / 255.f,   0 / 255.f,  64 / 255.f, 1.0f),
                float4(192 / 255.f, 255 / 255.f, 255 / 255.f, 1.0f),
            },
            // texture5.png
            {
                0x00011110, // Constant texture
                0, 0, 0,
                float4(81 / 255.f, 98 / 255.f, 201 / 255.f, 1.0f),
                float4(81 / 255.f, 98 / 255.f, 201 / 255.f, 1.0f),
                float4(81 / 255.f, 98 / 255.f, 201 / 255.f, 1.0f),
            },
            // texture6.png
            {
                0x00011118, // Alpha differs
                0, 0, 0,
                float4(163 / 255.f, 169 / 255.f, 218 / 255.f, 1.0f),
                float4(163 / 255.f, 169 / 255.f, 218 / 255.f, 0.0f),
                float4(163 / 255.f, 169 / 255.f, 218 / 255.f, 1.0f),
            },
            // texture7.exr
            {
                0x0001951e, // R constant, G inf, B NaN, A varying, all channels positive
                0, 0, 0,
                float4(1 / 8.f, 2 / 8.f, 3 / 8.f, 4 / 8.f),
                float4(1 / 8.f, 2 / 8.f, 3 / 8.f, 1 / 16.f),
                float4(1 / 8.f, INFINITY, 3 / 8.f, 4 / 8.f),
            },
            // texture8.exr
            {
                0x0003222d, // R varying neg, G constant neg, B varying neg, A varying pos/neg
                0, 0, 0,
                float4(-19.f, -17.f, -15.f, -13.f),
                float4(0.f, 0.f, 0.f, 0.f),
                float4(0.f, 0.f, 0.f, 1 / 256.f),
            },
        };
    }

    GPU_TEST(TextureAnalyzer)
    {
        TextureAnalyzer::SharedPtr pTextureAnalyzer = TextureAnalyzer::create();
        EXPECT(pTextureAnalyzer != nullptr);

        // Load test textures.
        std::vector<Texture::SharedPtr> textures(kNumTests);
        for (size_t i = 0; i < kNumTests; i++)
        {
            std::string fn = "texture" + std::to_string(i + 1) + (i < kNumPNGs ? ".png" : ".exr");
            textures[i] = Texture::createFromFile(fn, false, false);
            if (!textures[i]) throw RuntimeError("Failed to load {}", fn);
        }

        // Analyze textures.
        auto pResult = Buffer::create(kNumTests * kResultSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        EXPECT(pResult);
        ctx.getRenderContext()->clearUAV(pResult->getUAV().get(), uint4(0));

        for (size_t i = 0; i < kNumTests; i++)
        {
            pTextureAnalyzer->analyze(ctx.getRenderContext(), textures[i], 0, 0, pResult, i * kResultSize);
        }

        auto verify = [&ctx](Buffer::SharedPtr pResult)
        {
            // Verify results.
            const TextureAnalyzer::Result* result = static_cast<const TextureAnalyzer::Result*>(pResult->map(Buffer::MapType::Read));
            for (size_t i = 0; i < kNumTests; i++)
            {
                EXPECT_EQ(result[i].mask, kExpectedResult[i].mask) << "i = " << i;

                uint32_t rangeFlags = 0;
                for (int c = 0; c < 4; c++)
                {
                    bool isConstant = (kExpectedResult[i].mask & (1u << c)) == 0;
                    rangeFlags |= kExpectedResult[i].mask >> (4 + 4 * c);

                    EXPECT_EQ(result[i].isConstant(1u << c), isConstant) << " c = " << c;
                    EXPECT_EQ(result[i].minValue[c], kExpectedResult[i].minValue[c]) << "i = " << i << " c = " << c;
                    EXPECT_EQ(result[i].maxValue[c], kExpectedResult[i].maxValue[c]) << "i = " << i << " c = " << c;

                    if (isConstant)
                    {
                        EXPECT_EQ(result[i].value[c], kExpectedResult[i].value[c]) << "i = " << i << " c = " << c;
                    }
                }

                EXPECT_EQ(result[i].isPos(TextureChannelFlags::RGBA), (rangeFlags & (uint32_t)TextureAnalyzer::Result::RangeFlags::Pos) != 0) << "i = " << i;
                EXPECT_EQ(result[i].isNeg(TextureChannelFlags::RGBA), (rangeFlags & (uint32_t)TextureAnalyzer::Result::RangeFlags::Neg) != 0) << "i = " << i;
                EXPECT_EQ(result[i].isInf(TextureChannelFlags::RGBA), (rangeFlags & (uint32_t)TextureAnalyzer::Result::RangeFlags::Inf) != 0) << "i = " << i;
                EXPECT_EQ(result[i].isNaN(TextureChannelFlags::RGBA), (rangeFlags & (uint32_t)TextureAnalyzer::Result::RangeFlags::NaN) != 0) << "i = " << i;
            }
            pResult->unmap();
        };

        verify(pResult);

        // Test the array version of the interface.
        ctx.getRenderContext()->clearUAV(pResult->getUAV().get(), uint4(0xbabababa));
        pTextureAnalyzer->analyze(ctx.getRenderContext(), textures, pResult);

        verify(pResult);
    }
}
