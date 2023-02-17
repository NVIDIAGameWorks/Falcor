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
#include "Utils/Math/Rectangle.h"

namespace Falcor
{

CPU_TEST(Rectangle_Constructors)
{
    Rectangle tile0;
    EXPECT_FALSE(tile0.valid());

    Rectangle tile1(float2(0.5f));
    EXPECT_TRUE(tile1.valid());
    EXPECT_EQ(tile1.center(), float2(0.5f));
    EXPECT_EQ(tile1.extent(), float2(0.f));

    tile0.set(float2(0.5f));
    EXPECT(tile0 == tile1);
    EXPECT(tile0 == tile0.intersection(tile1));
}

CPU_TEST(Rectangle_Comparisons)
{
    Rectangle tile0;
    EXPECT_FALSE(tile0.valid());

    tile0.include(float2(-1.f));
    EXPECT_TRUE(tile0.valid());
    EXPECT_EQ(tile0.area(), 0.f);

    tile0.include(float2(1.f));
    EXPECT_TRUE(tile0.valid());
    EXPECT_EQ(tile0.area(), 4.f);
    EXPECT_EQ(tile0.extent(), float2(2.f));

    Rectangle tile1(float2(0.f), float2(2.f));

    Rectangle tile2 = tile0.intersection(tile1);
    Rectangle tile3 = tile1.intersection(tile2);

    EXPECT(tile2 == tile3);

    EXPECT_EQ(tile2.maxPoint, float2(1.f));
    EXPECT_EQ(tile2.minPoint, float2(0.f));
}

CPU_TEST(Rectangle_Contains)
{
    Rectangle invalid;
    ASSERT_FALSE(invalid.valid());

    Rectangle big(float2(-1.f), float2(1.f));
    Rectangle small0(float2(0.f), float2(1.f));
    Rectangle small1(float2(-1.f), float2(0.f));
    Rectangle small2(float2(-1.1f), float2(0.f));

    EXPECT(big.contains(big));
    EXPECT(big.contains(small0));
    EXPECT(big.contains(small1));
    EXPECT_FALSE(big.contains(small2));

    Rectangle invalid0;
    ASSERT_FALSE(invalid0.valid());
    EXPECT_FALSE(invalid0.overlaps(small0));
    EXPECT_FALSE(invalid0.overlaps(small1));
    EXPECT_FALSE(invalid0.overlaps(small2));
    EXPECT_FALSE(small0.overlaps(invalid0));
    EXPECT_FALSE(small1.overlaps(invalid0));
    EXPECT_FALSE(small2.overlaps(invalid0));

    Rectangle invalid1;
    ASSERT_FALSE(invalid1.valid());
    EXPECT_FALSE(invalid0.overlaps(invalid1));
}

CPU_TEST(Rectangle_Overlaps)
{
    Rectangle tile0(float2(-1.f), float2(1.f));
    Rectangle tile1(float2(0.f), float2(2.f));
    Rectangle tile2(float2(1.f), float2(2.f));

    EXPECT(tile0.overlaps(tile0));
    EXPECT(tile0.overlaps(tile1));
    EXPECT(tile1.overlaps(tile0));
    EXPECT_FALSE(tile0.overlaps(tile2));
    EXPECT_FALSE(tile2.overlaps(tile0));

    Rectangle invalid0;
    ASSERT_FALSE(invalid0.valid());
    EXPECT_FALSE(invalid0.overlaps(tile0));
    EXPECT_FALSE(invalid0.overlaps(tile1));
    EXPECT_FALSE(invalid0.overlaps(tile2));
    EXPECT_FALSE(tile0.overlaps(invalid0));
    EXPECT_FALSE(tile1.overlaps(invalid0));
    EXPECT_FALSE(tile2.overlaps(invalid0));

    Rectangle invalid1;
    ASSERT_FALSE(invalid1.valid());
    EXPECT_FALSE(invalid0.overlaps(invalid1));
}

} // namespace Falcor
