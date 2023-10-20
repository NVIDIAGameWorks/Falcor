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
#include "Core/AssetResolver.h"
#include "Scene/Lights/EnvMap.h"
#include "Rendering/Lights/EnvMapSampler.h"

namespace Falcor
{
namespace
{
// TODO: This is not ideal, we should only access files in the runtime directory.
const std::filesystem::path kEnvMapPath = getProjectDirectory() / "media/test_scenes/envmaps/20050806-03_hd.hdr";
} // namespace

GPU_TEST(EnvMap)
{
    // Test loading a light probe.
    // This call runs setup code on the GPU to precompute the importance map.
    // If it succeeds, we at least know the code compiles and run.
    ref<EnvMap> pEnvMap = EnvMap::createFromFile(ctx.getDevice(), kEnvMapPath);
    EXPECT_NE(pEnvMap, nullptr);
    if (pEnvMap == nullptr)
        return;

    EnvMapSampler envMapSampler(ctx.getDevice(), pEnvMap);

    // Check that the importance map exists and is a square power-of-two
    // texture with a full mip map hierarchy.
    auto pImportanceMap = envMapSampler.getImportanceMap();
    EXPECT_NE(pImportanceMap, nullptr);
    if (pImportanceMap == nullptr)
        return;

    uint32_t w = pImportanceMap->getWidth();
    uint32_t h = pImportanceMap->getHeight();
    uint32_t mipCount = pImportanceMap->getMipCount();

    EXPECT(isPowerOf2(w) && w > 0);
    EXPECT_EQ(w, h);
    EXPECT_EQ(w, 1 << (mipCount - 1));
}
} // namespace Falcor
