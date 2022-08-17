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
#include "Utils/Image/ImageProcessing.h"
#include "Utils/Math/Float16.h"

namespace Falcor
{
    namespace
    {
        template<typename T>
        std::vector<T> generateTestData(size_t elems)
        {
            std::vector<T> data;
            for (size_t i = 0; i < elems; i++)
            {
                float c = i * 2.5f * (i % 2 ? -1.f : 1.f);
                data.push_back(T(c));
            }
            return data;
        }

        template<typename T>
        void testCopyColorChannel(GPUUnitTestContext& ctx, ImageProcessing::SharedPtr& pImageProcessing, uint32_t width, uint32_t height, ResourceFormat srcFormat, ResourceFormat dstFormat)
        {
            const auto srcChannels = getFormatChannelCount(srcFormat);
            const auto dstChannels = getFormatChannelCount(dstFormat);

            // Create test textures.
            auto data = generateTestData<T>(width * height * srcChannels);
            auto pSrc = Texture::create2D(width, height, srcFormat, 1, 1, data.data(), ResourceBindFlags::ShaderResource);
            auto pDst = Texture::create2D(width, height, dstFormat, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

            // Test copying from color channel i=0..3.
            std::vector<TextureChannelFlags> channelMask = { TextureChannelFlags::Red, TextureChannelFlags::Green, TextureChannelFlags::Blue, TextureChannelFlags::Alpha };
            for (uint32_t i = 0; i < 4; i++)
            {
                // Copy channel to destination texture.
                pImageProcessing->copyColorChannel(ctx.getRenderContext(), pSrc->getSRV(), pDst->getUAV(), channelMask[i]);

                // Validate result.
                auto rawData = ctx.getRenderContext()->readTextureSubresource(pDst.get(), 0);

                EXPECT_EQ(getFormatPixelsPerBlock(dstFormat), 1);
                EXPECT_EQ(rawData.size(), width * height * getFormatBytesPerBlock(dstFormat));

                const T* result = (const T*)rawData.data();

                for (size_t j = 0; j < (size_t)width * height; j++)
                {
                    T ref = data[j * srcChannels + i];
                    for (uint32_t k = 0; k < dstChannels; k++)
                    {
                        T value = result[j * dstChannels + k];
                        // TODO: Remove workaround when float16_t implements ostream operator.
                        if constexpr (std::is_same<float16_t, T>::value)
                        {
                            EXPECT_EQ((float)ref, (float)value) << "j=" << j << " k=" << k << " dstFormat=" + to_string(dstFormat);
                        }
                        else
                        {
                            EXPECT_EQ(ref, value) << "j=" << j << " k=" << k << " dstFormat=" + to_string(dstFormat);
                        }
                    }
                }
            }
        }
    }

    GPU_TEST(CopyColorChannel)
    {
        uint32_t w = 15, h = 3;
        auto pImageProcessing = ImageProcessing::create();
        testCopyColorChannel<float>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA32Float, ResourceFormat::RGBA32Float);
        testCopyColorChannel<float>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA32Float, ResourceFormat::RG32Float);
        testCopyColorChannel<float>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA32Float, ResourceFormat::R32Float);
        testCopyColorChannel<uint32_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA32Uint, ResourceFormat::RGBA32Uint);
        testCopyColorChannel<uint32_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA32Uint, ResourceFormat::RG32Uint);
        testCopyColorChannel<uint32_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA32Uint, ResourceFormat::R32Uint);
        testCopyColorChannel<float16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Float, ResourceFormat::RGBA16Float);
        testCopyColorChannel<float16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Float, ResourceFormat::RG16Float);
        testCopyColorChannel<float16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Float, ResourceFormat::R16Float);
        testCopyColorChannel<uint16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Unorm, ResourceFormat::RGBA16Unorm);
        testCopyColorChannel<uint16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Unorm, ResourceFormat::RG16Unorm);
        testCopyColorChannel<uint16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Unorm, ResourceFormat::R16Unorm);
        testCopyColorChannel<int16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Int, ResourceFormat::RGBA16Int);
        testCopyColorChannel<int16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Int, ResourceFormat::RG16Int);
        testCopyColorChannel<int16_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA16Int, ResourceFormat::R16Int);
        testCopyColorChannel<int8_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA8Snorm, ResourceFormat::RGBA8Snorm);
        testCopyColorChannel<int8_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA8Snorm, ResourceFormat::RG8Snorm);
        testCopyColorChannel<int8_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA8Snorm, ResourceFormat::R8Snorm);
        testCopyColorChannel<int8_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA8Int, ResourceFormat::RGBA8Int);
        testCopyColorChannel<int8_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA8Int, ResourceFormat::RG8Int);
        testCopyColorChannel<int8_t>(ctx, pImageProcessing, w, h, ResourceFormat::RGBA8Int, ResourceFormat::R8Int);
    }
}
