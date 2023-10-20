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
#include <fstream>

namespace Falcor
{
static const std::filesystem::path kTestRoot = getRuntimeDirectory() / "asset_test_root";

static const std::vector<std::filesystem::path> kTestFiles = {
    "media1/asset1",
    "media2/asset1",
    "media2/asset2",
    "media3/asset1",
    "media3/asset2",
    "media3/asset3",
    "media4/textures/mip0.png",
    "media4/textures/mip1.png",
    "media4/textures/mip2.png",
    "media4/textures/mip3.png",
};

static void createTestFiles(UnitTestContext& ctx)
{
    std::error_code err;
    std::filesystem::create_directories(kTestRoot, err);
    ASSERT(!err);
    for (const auto& file : kTestFiles)
    {
        std::filesystem::path path = kTestRoot / file;
        std::filesystem::create_directories(path.parent_path(), err);
        ASSERT(!err);
        std::ofstream(path).close();
        ASSERT(std::filesystem::exists(path));
    }
}

static void removeTestFiles(UnitTestContext& ctx)
{
    std::filesystem::remove_all(kTestRoot);
    ASSERT(!std::filesystem::exists(kTestRoot));
}

CPU_TEST(AssetResolver)
{
    createTestFiles(ctx);

    const std::filesystem::path unresolved;

    // Test resolving absolute paths.
    {
        AssetResolver resolver;

        resolver.addSearchPath(kTestRoot / "media1");
        EXPECT_EQ(resolver.resolvePath(kTestRoot / "media2/asset1"), kTestRoot / "media2/asset1");

        auto resolved = resolver.resolvePathPattern(kTestRoot / "media4/textures", R"(mip[0-9]\.png)");
        EXPECT_EQ(resolved.size(), 4);
        std::sort(resolved.begin(), resolved.end());
        EXPECT_EQ(resolved[0], kTestRoot / "media4/textures/mip0.png");
        EXPECT_EQ(resolved[1], kTestRoot / "media4/textures/mip1.png");
        EXPECT_EQ(resolved[2], kTestRoot / "media4/textures/mip2.png");
        EXPECT_EQ(resolved[3], kTestRoot / "media4/textures/mip3.png");
    }

    // Test resolving relative paths to working directory.
    {
        AssetResolver resolver;

        resolver.addSearchPath(kTestRoot / "media1");
        EXPECT_EQ(
            resolver.resolvePath(std::filesystem::relative(kTestRoot / "media2/asset1", std::filesystem::current_path())),
            kTestRoot / "media2/asset1"
        );

        auto resolved = resolver.resolvePathPattern(
            resolver.resolvePath(std::filesystem::relative(kTestRoot / "media4/textures", std::filesystem::current_path())),
            R"(mip[0-9]\.png)"
        );
        EXPECT_EQ(resolved.size(), 4);
        std::sort(resolved.begin(), resolved.end());
        EXPECT_EQ(resolved[0], kTestRoot / "media4/textures/mip0.png");
        EXPECT_EQ(resolved[1], kTestRoot / "media4/textures/mip1.png");
        EXPECT_EQ(resolved[2], kTestRoot / "media4/textures/mip2.png");
        EXPECT_EQ(resolved[3], kTestRoot / "media4/textures/mip3.png");
    }

    // Test resolving with search paths.
    {
        AssetResolver resolver;

        resolver.addSearchPath(kTestRoot / "media1");
        EXPECT_EQ(resolver.resolvePath("asset1"), kTestRoot / "media1/asset1");
        EXPECT_EQ(resolver.resolvePath("asset2"), unresolved);
        EXPECT_EQ(resolver.resolvePath("asset3"), unresolved);
        EXPECT_EQ(resolver.resolvePath("asset4"), unresolved);

        resolver.addSearchPath(kTestRoot / "media2");
        EXPECT_EQ(resolver.resolvePath("asset1"), kTestRoot / "media1/asset1");
        EXPECT_EQ(resolver.resolvePath("asset2"), kTestRoot / "media2/asset2");
        EXPECT_EQ(resolver.resolvePath("asset3"), unresolved);
        EXPECT_EQ(resolver.resolvePath("asset4"), unresolved);

        resolver.addSearchPath(kTestRoot / "media3");
        EXPECT_EQ(resolver.resolvePath("asset1"), kTestRoot / "media1/asset1");
        EXPECT_EQ(resolver.resolvePath("asset2"), kTestRoot / "media2/asset2");
        EXPECT_EQ(resolver.resolvePath("asset3"), kTestRoot / "media3/asset3");
        EXPECT_EQ(resolver.resolvePath("asset4"), unresolved);
    }

    // Test resolving patterns with search paths.
    {
        AssetResolver resolver;

        resolver.addSearchPath(kTestRoot / "media4");
        auto resolved = resolver.resolvePathPattern("textures", R"(mip[0-9]\.png)");
        EXPECT_EQ(resolved.size(), 4);
        std::sort(resolved.begin(), resolved.end());
        EXPECT_EQ(resolved[0], kTestRoot / "media4/textures/mip0.png");
        EXPECT_EQ(resolved[1], kTestRoot / "media4/textures/mip1.png");
        EXPECT_EQ(resolved[2], kTestRoot / "media4/textures/mip2.png");
        EXPECT_EQ(resolved[3], kTestRoot / "media4/textures/mip3.png");

        resolved = resolver.resolvePathPattern("textures", R"(mip[0-9]\.png)", true);
        EXPECT_EQ(resolved.size(), 1);
        EXPECT(
            resolved[0] == kTestRoot / "media4/textures/mip0.png" || resolved[0] == kTestRoot / "media4/textures/mip1.png" ||
            resolved[0] == kTestRoot / "media4/textures/mip2.png" || resolved[0] == kTestRoot / "media4/textures/mip3.png"
        );
    }

    // Test asset categories.
    {
        AssetResolver resolver;

        resolver.addSearchPath(kTestRoot / "media3", SearchPathPriority::Last, AssetCategory::Any);
        resolver.addSearchPath(kTestRoot / "media2", SearchPathPriority::Last, AssetCategory::Scene);
        resolver.addSearchPath(kTestRoot / "media1", SearchPathPriority::Last, AssetCategory::Texture);

        EXPECT_EQ(resolver.resolvePath("asset1", AssetCategory::Any), kTestRoot / "media3/asset1");
        EXPECT_EQ(resolver.resolvePath("asset1", AssetCategory::Scene), kTestRoot / "media2/asset1");
        EXPECT_EQ(resolver.resolvePath("asset1", AssetCategory::Texture), kTestRoot / "media1/asset1");

        EXPECT_EQ(resolver.resolvePath("asset2", AssetCategory::Any), kTestRoot / "media3/asset2");
        EXPECT_EQ(resolver.resolvePath("asset2", AssetCategory::Scene), kTestRoot / "media2/asset2");
        EXPECT_EQ(resolver.resolvePath("asset2", AssetCategory::Texture), kTestRoot / "media3/asset2");

        EXPECT_EQ(resolver.resolvePath("asset3", AssetCategory::Any), kTestRoot / "media3/asset3");
        EXPECT_EQ(resolver.resolvePath("asset3", AssetCategory::Scene), kTestRoot / "media3/asset3");
        EXPECT_EQ(resolver.resolvePath("asset3", AssetCategory::Texture), kTestRoot / "media3/asset3");
    }

    // Test search path priorities.
    {
        AssetResolver resolver;

        resolver.addSearchPath(kTestRoot / "media1");
        EXPECT_EQ(resolver.resolvePath("asset1"), kTestRoot / "media1/asset1");
        resolver.addSearchPath(kTestRoot / "media2", SearchPathPriority::Last);
        EXPECT_EQ(resolver.resolvePath("asset1"), kTestRoot / "media1/asset1");
        resolver.addSearchPath(kTestRoot / "media3", SearchPathPriority::First);
        EXPECT_EQ(resolver.resolvePath("asset1"), kTestRoot / "media3/asset1");
    }

    removeTestFiles(ctx);
}

} // namespace Falcor
