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
#include "Utils/Math/Vector.h"

#include <fmt/format.h>
#include <iostream>
#include <algorithm>
#include <random>

namespace Falcor
{

CPU_TEST(Vector_FloatFormatter)
{
    float2 test0(1.23456789f, 2.f);

    EXPECT_EQ(fmt::format("{}", test0), "{1.2345679, 2}");
    EXPECT_EQ(fmt::format("{:e}", test0), "{1.234568e+00, 2.000000e+00}");
    EXPECT_EQ(fmt::format("{:g}", test0), "{1.23457, 2}");
    EXPECT_EQ(fmt::format("{:.1}", test0), "{1, 2}");
    EXPECT_EQ(fmt::format("{:.3}", test0), "{1.23, 2}");
}

CPU_TEST(Vector_IntFormatter)
{
    int2 test0(12, 34);

    EXPECT_EQ(fmt::format("{}", test0), "{12, 34}");
    EXPECT_EQ(fmt::format("{:x}", test0), "{c, 22}");
    EXPECT_EQ(fmt::format("{:08x}", test0), "{0000000c, 00000022}");
    EXPECT_EQ(fmt::format("{:b}", test0), "{1100, 100010}");
    EXPECT_EQ(fmt::format("{:08b}", test0), "{00001100, 00100010}");
    EXPECT_EQ(fmt::format("{:08X}", test0), "{0000000C, 00000022}");
}

CPU_TEST(Vector_Comparison)
{
    std::vector<int2> vec{{-1, -1}, {-1, +1}, {+1, -1}, {+1, +1}, {-2, -2}, {-2, +2}, {+2, -2}, {+2, +2}};

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);
    std::sort(vec.begin(), vec.end(), std::less<int2>{});
    for (size_t i = 0; i < vec.size(); ++i)
    {
        for (size_t j = i + 1; j < vec.size(); ++j)
        {
            EXPECT(std::less<int2>{}(vec[i], vec[j]));
        }
    }
}

} // namespace Falcor
