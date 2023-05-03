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
#include "Utils/PathResolving.h"

#if FALCOR_WINDOWS
#define C_DRIVE "c:"
#define D_DRIVE "d:"
#define E_DRIVE "e:"
#else
#define C_DRIVE "/c"
#define D_DRIVE "/d"
#define E_DRIVE "/e"
#endif

namespace Falcor
{

CPU_TEST(PathResolving_ResolveEnvVar)
{
    auto proxyResolver = [](const std::string& varName) -> std::optional<std::string>
    {
        if (varName == "VAR1")
            return "Value1";
        if (varName == "VAR2")
            return "Value2";
        if (varName == "VAR3")
            return "Value3";
        return std::nullopt;
    };

    std::string test = "test1";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "test1");

    test = "${VAR1}";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "Value1");

    test = "_${VAR1}";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "_Value1");

    test = "_${VAR1}_";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "_Value1_");

    test = "${VAR1}${VAR2}";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "Value1Value2");

    test = "_${VAR1}${VAR2}";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "_Value1Value2");

    test = "_${VAR1}_${VAR2}";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "_Value1_Value2");

    test = "_${VAR1}_${VAR2}_";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "_Value1_Value2_");

    test = "${VAR1}_${VAR2}_";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "Value1_Value2_");

    test = "${VAR1}_${VAR2}";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "Value1_Value2");

    test = "_${VAR1}${VAR2}_";
    resolveEnvVariables(test, proxyResolver);
    EXPECT_EQ(test, "_Value1Value2_");
};

// We use comparison with weakly_canonical paths to resolve all possible permissible differences in paths,
// such as capital/lowercase letters or backwards/forwards slashes. Just straight path comparison does a string
// comparison, so C:\Media != c:\media, while std::filesystem::equivalent requires the target of the path to actually
// exist on the file system. weakly_canonical will resolve the existing part of the path to what exists on the filesystem,
// and the non-existing remainder of the path into a standardized path form.

CPU_TEST(PathResolving_Basic)
{
    const std::vector<std::filesystem::path> standard{C_DRIVE "/standard/path"};
    const std::vector<std::filesystem::path> current{C_DRIVE "/current/path"};
    std::vector<std::string> update;
    ResolvedPaths result;

    update = {C_DRIVE "/update/path/one", D_DRIVE "/update/path/two"};
    result = resolveSearchPaths(current, update, standard);
    ASSERT_EQ(result.invalid.size(), 0);
    ASSERT_EQ(result.resolved.size(), 2);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/update/path/one"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(D_DRIVE "/update/path/two"));

    update = {C_DRIVE "/update/path/one;" D_DRIVE "/update/path/two"};
    result = resolveSearchPaths(current, update, standard);
    ASSERT_EQ(result.invalid.size(), 0);
    ASSERT_EQ(result.resolved.size(), 2);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/update/path/one"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(D_DRIVE "/update/path/two"));

    update = {C_DRIVE "/update/path/one;&;" D_DRIVE "/update/path/two;@;"};
    result = resolveSearchPaths(current, update, standard);
    ASSERT_EQ(result.invalid.size(), 0);
    ASSERT_EQ(result.resolved.size(), 4);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/update/path/one"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(C_DRIVE "/current/path"));
    EXPECT_EQ(result.resolved[2], std::filesystem::weakly_canonical(D_DRIVE "/update/path/two"));
    EXPECT_EQ(result.resolved[3], std::filesystem::weakly_canonical(C_DRIVE "/standard/path"));

    update = {C_DRIVE "/update/path/one;&", D_DRIVE "/update/path/two;@;"};
    result = resolveSearchPaths(current, update, standard);
    ASSERT_EQ(result.invalid.size(), 0);
    ASSERT_EQ(result.resolved.size(), 4);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/update/path/one"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(C_DRIVE "/current/path"));
    EXPECT_EQ(result.resolved[2], std::filesystem::weakly_canonical(D_DRIVE "/update/path/two"));
    EXPECT_EQ(result.resolved[3], std::filesystem::weakly_canonical(C_DRIVE "/standard/path"));

    update = {"update/path/one;&;" D_DRIVE "/update/path/two;@;"};
    result = resolveSearchPaths(current, update, standard);
    ASSERT_EQ(result.invalid.size(), 1);
    EXPECT_EQ(result.invalid[0], "update/path/one");
    ASSERT_EQ(result.resolved.size(), 3);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/current/path"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(D_DRIVE "/update/path/two"));
    EXPECT_EQ(result.resolved[2], std::filesystem::weakly_canonical(C_DRIVE "/standard/path"));

    update = {"update/path/one;&;:/update/path/two;@;"};
    result = resolveSearchPaths(current, update, standard);
    ASSERT_EQ(result.invalid.size(), 2);
    EXPECT_EQ(result.invalid[0], "update/path/one");
    EXPECT_EQ(result.invalid[1], ":/update/path/two");
    ASSERT_EQ(result.resolved.size(), 2);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/current/path"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(C_DRIVE "/standard/path"));
}

CPU_TEST(PathResolving_EnvVar)
{
    auto proxyResolver = [](const std::string& varName) -> std::optional<std::string>
    {
        if (varName == "FALCOR_MEDIA_LIBRARY")
            return C_DRIVE "/Project/Media";
        if (varName == "USERNAME")
            return "jdoe";
        return std::nullopt;
    };

    const std::vector<std::filesystem::path> standard{C_DRIVE "/standard/path"};
    const std::vector<std::filesystem::path> current{C_DRIVE "/current/path"};
    std::vector<std::string> update;
    ResolvedPaths result;

    update = {"${FALCOR_MEDIA_LIBRARY}", C_DRIVE "/Users/${USERNAME}/.falcor/media"};
    result = resolveSearchPaths(current, update, standard, proxyResolver);
    ASSERT_EQ(result.invalid.size(), 0);
    ASSERT_EQ(result.resolved.size(), 2);
    EXPECT_EQ(result.resolved[0], std::filesystem::weakly_canonical(C_DRIVE "/Project/Media"));
    EXPECT_EQ(result.resolved[1], std::filesystem::weakly_canonical(C_DRIVE "/Users/jdoe/.falcor/media"));
}

CPU_TEST(PathResolving_resolvePath)
{
    auto fileChecker = [](const std::filesystem::path& path)
    {
        auto canonical = std::filesystem::weakly_canonical(path);
        if (canonical == std::filesystem::weakly_canonical(C_DRIVE "/Users/jdoe/settings.ini"))
            return true;
        if (canonical == std::filesystem::weakly_canonical(D_DRIVE "/Project/Media/cornellbox.obj"))
            return true;
        if (canonical == std::filesystem::weakly_canonical(D_DRIVE "/Project/Geometry/cornellbox.usd"))
            return true;
        if (canonical == std::filesystem::weakly_canonical(E_DRIVE "/Textures/checkers.exr"))
            return true;
        if (canonical == std::filesystem::weakly_canonical(D_DRIVE "/Project/Media/test.txt"))
            return true;
        if (canonical == std::filesystem::weakly_canonical(E_DRIVE "/Textures/test.txt"))
            return true;
        return false;
    };

    const std::vector<std::filesystem::path> searchPaths{D_DRIVE "/Project/Media/", E_DRIVE "/Textures/"};
    const std::filesystem::path cwd{D_DRIVE "/Project/Geometry/"};
    std::filesystem::path result;

    result = resolvePath(searchPaths, cwd, "../Media/cornellbox.obj", fileChecker);
    EXPECT_EQ(result, std::filesystem::weakly_canonical(D_DRIVE "/Project/Media/cornellbox.obj"));

    result = resolvePath(searchPaths, cwd, "cornellbox.obj", fileChecker);
    EXPECT_EQ(result, std::filesystem::weakly_canonical(D_DRIVE "/Project/Media/cornellbox.obj"));

    result = resolvePath(searchPaths, cwd, "test.txt", fileChecker);
    EXPECT_EQ(result, std::filesystem::weakly_canonical(D_DRIVE "/Project/Media/test.txt"));

    result = resolvePath(searchPaths, cwd, "checkers.exr", fileChecker);
    EXPECT_EQ(result, std::filesystem::weakly_canonical(E_DRIVE "/Textures/checkers.exr"));

    result = resolvePath(searchPaths, cwd, C_DRIVE "/Users/jdoe/settings.ini", fileChecker);
    EXPECT_EQ(result, std::filesystem::weakly_canonical(C_DRIVE "/Users/jdoe/settings.ini"));

    result = resolvePath(searchPaths, cwd, "./checkers.exr", fileChecker);
    EXPECT(result.empty());

    result = resolvePath(searchPaths, cwd, "./test.txt", fileChecker);
    EXPECT(result.empty());

    result = resolvePath(searchPaths, cwd, "./cornellbox.usd", fileChecker);
    EXPECT_EQ(result, std::filesystem::weakly_canonical(D_DRIVE "/Project/Geometry/cornellbox.usd"));

    result = resolvePath(searchPaths, cwd, "cornellbox.usd", fileChecker);
    EXPECT(result.empty());
}

} // namespace Falcor
