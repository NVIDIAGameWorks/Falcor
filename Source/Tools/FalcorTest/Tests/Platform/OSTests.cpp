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
#include "Core/Platform/OS.h"

namespace Falcor
{
    CPU_TEST(Junction)
    {
        std::filesystem::path cwd = std::filesystem::current_path();
        std::filesystem::path target = cwd / "junction_target";
        std::filesystem::path link = cwd / "junction_link";

        // Create junction_target/test
        std::filesystem::create_directories(target / "test");

        // Create junction from junction_link to junction_target
        EXPECT_EQ(createJunction(link, target), true);
        // Check that junction was successfully created by accessing junction_link/test
        EXPECT_EQ(std::filesystem::exists(link / "test"), true);
        // Delete junction
        EXPECT_EQ(deleteJunction(link), true);
        // Check that junction was deleted
        EXPECT_EQ(std::filesystem::exists(link), false);

        // Delete junction_target/test
        std::filesystem::remove_all(target);
    }

    CPU_TEST(HasExtension)
    {
        EXPECT_EQ(hasExtension("foo.exr", "exr"), true);
        EXPECT_EQ(hasExtension("foo.exr", ".exr"), true);
        EXPECT_EQ(hasExtension("foo.Exr", "exr"), true);
        EXPECT_EQ(hasExtension("foo.Exr", ".exr"), true);
        EXPECT_EQ(hasExtension("foo.Exr", "exR"), true);
        EXPECT_EQ(hasExtension("foo.Exr", ".exR"), true);
        EXPECT_EQ(hasExtension("foo.EXR", "exr"), true);
        EXPECT_EQ(hasExtension("foo.EXR", ".exr"), true);
        EXPECT_EQ(hasExtension("foo.xr", "exr"), false);
        EXPECT_EQ(hasExtension("/foo/png", ""), true);
        EXPECT_EQ(hasExtension("/foo/png", "exr"), false);
        EXPECT_EQ(hasExtension("/foo/.profile", ""), true);
    }

    CPU_TEST(GetExtensionFromPath)
    {
        EXPECT_EQ(getExtensionFromPath("foo.exr"), "exr");
        EXPECT_EQ(getExtensionFromPath("foo.Exr"), "exr");
        EXPECT_EQ(getExtensionFromPath("foo.EXR"), "exr");
        EXPECT_EQ(getExtensionFromPath("foo"), "");
        EXPECT_EQ(getExtensionFromPath("/foo/.profile"), "");
    }

}
