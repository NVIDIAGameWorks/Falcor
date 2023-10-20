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
#include "Core/Enum.h"

enum class TestEnum
{
    A,
    B,
    C,
};

FALCOR_ENUM_INFO(
    TestEnum,
    {
        {TestEnum::A, "A"},
        {TestEnum::B, "B"},
        {TestEnum::C, "C"},
    }
);
FALCOR_ENUM_REGISTER(TestEnum);

enum class TestFlags
{
    A = 1 << 0,
    B = 1 << 1,
    C = 1 << 2,
};
FALCOR_ENUM_CLASS_OPERATORS(TestFlags);

FALCOR_ENUM_INFO(
    TestFlags,
    {
        {TestFlags::A, "A"},
        {TestFlags::B, "B"},
        {TestFlags::C, "C"},
    }
);
FALCOR_ENUM_REGISTER(TestFlags);

namespace Falcor
{
struct TestStruct
{
    enum class TestEnum
    {
        X,
        Y,
    };

    FALCOR_ENUM_INFO(
        TestEnum,
        {
            {TestEnum::X, "X"},
            {TestEnum::Y, "Y"},
        }
    );
};

FALCOR_ENUM_REGISTER(TestStruct::TestEnum);
} // namespace Falcor

namespace Falcor
{
static_assert(has_enum_info<void>::value == false);
static_assert(has_enum_info_v<void> == false);
static_assert(has_enum_info<::TestEnum>::value == true);
static_assert(has_enum_info_v<::TestEnum> == true);
static_assert(has_enum_info<TestStruct::TestEnum>::value == true);
static_assert(has_enum_info_v<TestStruct::TestEnum> == true);

CPU_TEST(EnumInfo)
{
    EXPECT_TRUE(enumHasValue<TestEnum>("A"));
    EXPECT_TRUE(enumHasValue<TestEnum>("B"));
    EXPECT_TRUE(enumHasValue<TestEnum>("C"));
    EXPECT_FALSE(enumHasValue<TestEnum>("D"));

    EXPECT(stringToEnum<TestEnum>("A") == TestEnum::A);
    EXPECT(stringToEnum<TestEnum>("B") == TestEnum::B);
    EXPECT(stringToEnum<TestEnum>("C") == TestEnum::C);

    // Converting unregistered values throws.
    EXPECT_THROW(enumToString(TestEnum(-1)));

    // Converting unregistered strings throws.
    EXPECT_THROW(stringToEnum<TestEnum>("D"));

    EXPECT_TRUE(enumHasValue<TestStruct::TestEnum>("X"));
    EXPECT_TRUE(enumHasValue<TestStruct::TestEnum>("Y"));
    EXPECT_FALSE(enumHasValue<TestStruct::TestEnum>("Z"));

    // Test enum nested in namespace and struct.
    EXPECT(enumToString(TestStruct::TestEnum::X) == "X");
    EXPECT(enumToString(TestStruct::TestEnum::Y) == "Y");
    EXPECT(stringToEnum<TestStruct::TestEnum>("X") == TestStruct::TestEnum::X);
    EXPECT(stringToEnum<TestStruct::TestEnum>("Y") == TestStruct::TestEnum::Y);

    // Test flags.
    EXPECT(flagsToStringList(TestFlags{0}) == std::vector<std::string>({}));
    EXPECT(flagsToStringList(TestFlags::A) == std::vector<std::string>({"A"}));
    EXPECT(flagsToStringList(TestFlags::B) == std::vector<std::string>({"B"}));
    EXPECT(flagsToStringList(TestFlags::A | TestFlags::B) == std::vector<std::string>({"A", "B"}));
    EXPECT(flagsToStringList(TestFlags::A | TestFlags::B | TestFlags::C) == std::vector<std::string>({"A", "B", "C"}));

    // Converting unregistered values throws.
    EXPECT_THROW(flagsToStringList(TestFlags(-1)));

    EXPECT(stringListToFlags<TestFlags>({}) == TestFlags{0});
    EXPECT(stringListToFlags<TestFlags>({"A"}) == TestFlags::A);
    EXPECT(stringListToFlags<TestFlags>({"B"}) == TestFlags::B);
    EXPECT(stringListToFlags<TestFlags>({"A", "B"}) == (TestFlags::A | TestFlags::B));
    EXPECT(stringListToFlags<TestFlags>({"A", "B", "C"}) == (TestFlags::A | TestFlags::B | TestFlags::C));

    // Converting unregistered strings throws.
    EXPECT_THROW(stringListToFlags<TestFlags>({"D"}));
}
} // namespace Falcor
