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
#include "Utils/Image/TextureManager.h"

namespace Falcor
{
GPU_TEST(TextureManager_LoadMips)
{
    ref<Device> pDevice = ctx.getDevice();

    TextureManager textureManager(pDevice, 10);

    std::filesystem::path path = getRuntimeDirectory() / "data/tests/tiny_<MIP>.png";

    auto handle = textureManager.loadTexture(path, false, false, ResourceBindFlags::ShaderResource, false);
    EXPECT(handle.isValid());
    EXPECT(!handle.isUdim());

    auto tex = textureManager.getTexture(handle);
    ASSERT(tex != nullptr);

    EXPECT_EQ(tex->getWidth(), 4);
    EXPECT_EQ(tex->getHeight(), 4);
    EXPECT_EQ(tex->getDepth(), 1);
    EXPECT_EQ(tex->getMipCount(), 3);
    EXPECT_EQ(tex->getArraySize(), 1);
}
} // namespace Falcor
