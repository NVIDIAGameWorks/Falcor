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
#include "Utils/Image/ImageIO.h"
#include "Utils/Image/TextureAnalyzer.h"

namespace Falcor
{
    std::ostream& operator<<(std::ostream& os, const ResourceFormat fmt)
    {
        os << to_string(fmt);
        return os;
    }

    namespace
    {
        void testDDS(GPUUnitTestContext& ctx, const std::string& testName, ResourceFormat fmt, bool expectLoadFailure)
        {
            // Read input DDS file
            std::string ddsFileName(testName + ".dds");
            std::filesystem::path ddsFullPath;
            findFileInDataDirectories(ddsFileName, ddsFullPath);
            EXPECT(!ddsFullPath.empty());
            if (ddsFullPath.empty())
            {
                return;
            }

            Texture::SharedPtr pDDSTex;
            // Note that we can always specify loadAsSrgb=false, even when fmt is sRGB, because
            // the flag is a no-op if the format encoded in the DDS file specifies a nonlinear format.
            try
            {
                pDDSTex = ImageIO::loadTextureFromDDS(ddsFullPath, false);
            }
            catch (...)
            {
                // We handle failures below, using the value of pDDSTex and the expectLoadFailure flag.
            }
            if (expectLoadFailure)
            {
                EXPECT(pDDSTex == nullptr);
            }
            else
            {
                EXPECT(pDDSTex != nullptr);
            }
            if (pDDSTex == nullptr)
            {
                return;
            }

            // Read reference image.  If no reference image exists, the test will fail, and a reference image will be output.
            std::string refFileName(testName + "-ref.png");
            std::filesystem::path refFullPath;
            findFileInDataDirectories(refFileName, refFullPath);
            EXPECT(!refFullPath.empty());

            Texture::SharedPtr pPngTex;
            if (!refFullPath.empty())
            {
                pPngTex = Texture::createFromFile(refFullPath, false, false);
                EXPECT(pPngTex);
            }

            EXPECT_EQ(pDDSTex->getFormat(), fmt);

            // Create uncompressed destination texture
            Texture::SharedPtr pSrcTex = pDDSTex;
            ResourceFormat destFormat = ResourceFormat::RGBA32Float;
            auto pDst = Texture::create2D(pDDSTex->getWidth(), pDDSTex->getHeight(), destFormat, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);

            // Blit the compressed DDS texture to decompressed destination
            ctx.getRenderContext()->blit(pDDSTex->getSRV(0, 1, 0, 1), pDst->getRTV());

            if (pPngTex)
            {
                // Create program to compare decompressed image with reference image
                ctx.createProgram("Tests/Core/DDSReadTests.cs.slang", "diff", Program::DefineList(), Shader::CompilerFlags::None);
                ctx.allocateStructuredBuffer("difference", 4 * sizeof(float) * pDst->getWidth() * pDst->getHeight());
                ctx["ref"] = pPngTex;       // Reference
            }
            else
            {
                // Create program to copy decompressed image so that we can save it as the reference
                ctx.createProgram("Tests/Core/DDSReadTests.cs.slang", "readback", Program::DefineList(), Shader::CompilerFlags::None);
                ctx.allocateStructuredBuffer("result", 4 * sizeof(uint32_t) * pDst->getWidth() * pDst->getHeight());
            }

            const uint2 dstDim(pDst->getWidth(), pDst->getHeight());
            ctx["tex"] = pDst;
            ctx["CB"]["sz"] = dstDim;
            ctx.runProgram(dstDim.x, dstDim.y, 1);

            if (pPngTex)
            {
                // Create texture from difference data
                const uint8_t* diff = ctx.mapBuffer<const uint8_t>("difference");
                EXPECT(diff != nullptr);
                Texture::SharedPtr pDiffTex(Texture::create2D(dstDim.x, dstDim.y, ResourceFormat::RGBA32Float, 1, 1, diff));
                ctx.unmapBuffer("difference");

                // Analyze difference texture
                TextureAnalyzer::SharedPtr analyzer(TextureAnalyzer::create());
                auto pResultBuffer = Buffer::create(TextureAnalyzer::getResultSize(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                analyzer->analyze(ctx.getRenderContext(), pDiffTex, 0, 0, pResultBuffer);
                const TextureAnalyzer::Result* result = static_cast<const TextureAnalyzer::Result*>(pResultBuffer->map(Buffer::MapType::Read));

                // Expect difference image to be uniform 0.
                EXPECT(result->isConstant(TextureChannelFlags::Red));
                EXPECT(result->isConstant(TextureChannelFlags::Green));
                EXPECT(result->isConstant(TextureChannelFlags::Blue));
                EXPECT(result->isConstant(TextureChannelFlags::Alpha));
                EXPECT_EQ(result->value.r, 0.f);
                EXPECT_EQ(result->value.g, 0.f);
                EXPECT_EQ(result->value.b, 0.f);
                EXPECT_EQ(result->value.a, 0.f);
                pResultBuffer->unmap();
            }
            else
            {
                // Save newly-created reference image
                const uint8_t* result = ctx.mapBuffer<const uint8_t>("result");
                EXPECT(result != nullptr);

                Bitmap::UniqueConstPtr resultBitmap(Bitmap::create(dstDim.x, dstDim.y, ResourceFormat::RGBA8Unorm, (const uint8_t*)result));
                std::filesystem::path outRefPath = getExecutableDirectory() / "Data" / refFileName;
                Bitmap::saveImage(outRefPath, dstDim.x, dstDim.y, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::Uncompressed | Bitmap::ExportFlags::ExportAlpha, ResourceFormat::RGBA8Unorm, false, (void*)result);
                ctx.unmapBuffer("result");
            }
        }
    }

#define DDS_TEST(x,f) GPU_TEST(x) { testDDS(ctx,std::string(#x),f,false); }

    DDS_TEST(BC1Unorm,          ResourceFormat::BC1Unorm);
    DDS_TEST(BC1UnormSrgb,      ResourceFormat::BC1UnormSrgb);
    DDS_TEST(BC2Unorm,          ResourceFormat::BC2Unorm);
    DDS_TEST(BC2UnormSrgb,      ResourceFormat::BC2UnormSrgb);
    DDS_TEST(BC2UnormSrgbTiny,  ResourceFormat::BC2UnormSrgb);
    DDS_TEST(BC3Unorm,          ResourceFormat::BC3Unorm);
    DDS_TEST(BC3UnormAlpha,     ResourceFormat::BC3Unorm);
    DDS_TEST(BC3UnormAlphaTiny, ResourceFormat::BC3Unorm);
    DDS_TEST(BC3UnormSrgb,      ResourceFormat::BC3UnormSrgb);
    DDS_TEST(BC3UnormSrgbOdd,   ResourceFormat::BC3UnormSrgb);
    DDS_TEST(BC3UnormSrgbTiny,  ResourceFormat::BC3UnormSrgb);
    DDS_TEST(BC4Unorm,          ResourceFormat::BC4Unorm);
    DDS_TEST(BC5Unorm,          ResourceFormat::BC5Unorm);
    DDS_TEST(BC5UnormTiny,      ResourceFormat::BC5Unorm);
    DDS_TEST(BC6HU16,           ResourceFormat::BC6HU16);
    DDS_TEST(BC7Unorm,          ResourceFormat::BC7Unorm);
    DDS_TEST(BC7UnormOdd,       ResourceFormat::BC7Unorm);
    DDS_TEST(BC7UnormSrgb,      ResourceFormat::BC7UnormSrgb);
    DDS_TEST(BC7UnormTiny,      ResourceFormat::BC7Unorm);

    GPU_TEST(BC7UnormBroken) { testDDS(ctx, std::string("BC7UnormBroken"), ResourceFormat::BC7Unorm, true); }
}

