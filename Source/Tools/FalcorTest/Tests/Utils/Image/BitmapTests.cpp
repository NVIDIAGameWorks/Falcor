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
#include "Utils/Image/Bitmap.h"

namespace Falcor
{
GPU_TEST(Bitmap_LinearRamp_PNG)
{
    const auto path = getRuntimeDirectory() / "test_linear_ramp.png";

    // Test saving a linear ramp as an 8-bit grayscale PNG.
    {
        uint8_t data[256];
        for (uint32_t i = 0; i < 256; i++)
            data[i] = (uint8_t)i;

        Bitmap::saveImage(
            path, 256, 1, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, ResourceFormat::R8Uint, true /* top-down */, data
        );
    }

    // Test loading the linear ramp.
    // Note that saving 8-bit grayscale data as PNG currently results in it being converted to BGRX in 8-bit unorm format.
    auto bmp = Bitmap::createFromFile(path, true /* top-down */);
    EXPECT(bmp != nullptr);

    if (bmp)
    {
        EXPECT_EQ(bmp->getWidth(), 256);
        EXPECT_EQ(bmp->getHeight(), 1);
        EXPECT_EQ((uint32_t)bmp->getFormat(), (uint32_t)ResourceFormat::BGRX8Unorm);

        auto data = bmp->getData();
        EXPECT(data != nullptr);
        EXPECT_EQ(bmp->getSize(), 1024);

        if (data && bmp->getSize() == 1024)
        {
            for (uint32_t i = 0; i < 256; i++)
            {
                EXPECT_EQ(data[4 * i + 0], (uint8_t)i); // B
                EXPECT_EQ(data[4 * i + 1], (uint8_t)i); // G
                EXPECT_EQ(data[4 * i + 2], (uint8_t)i); // R
            }
        }
    }

    // Delete the test file.
    std::filesystem::remove(path);
}
} // namespace Falcor
