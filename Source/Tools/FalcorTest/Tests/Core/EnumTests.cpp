
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
    try
    {
        enumToString(TestEnum(-1));
        EXPECT(false);
    }
    catch (const RuntimeError&)
    {
        EXPECT(true);
    }

    // Converting unregistered strings throws.
    try
    {
        stringToEnum<TestEnum>("D");
        EXPECT(false);
    }
    catch (const RuntimeError&)
    {
        EXPECT(true);
    }

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
    try
    {
        flagsToStringList(TestFlags(-1));
        EXPECT(false);
    }
    catch (const RuntimeError&)
    {
        EXPECT(true);
    }

    EXPECT(stringListToFlags<TestFlags>({}) == TestFlags{0});
    EXPECT(stringListToFlags<TestFlags>({"A"}) == TestFlags::A);
    EXPECT(stringListToFlags<TestFlags>({"B"}) == TestFlags::B);
    EXPECT(stringListToFlags<TestFlags>({"A", "B"}) == (TestFlags::A | TestFlags::B));
    EXPECT(stringListToFlags<TestFlags>({"A", "B", "C"}) == (TestFlags::A | TestFlags::B | TestFlags::C));

    // Converting unregistered strings throws.
    try
    {
        stringListToFlags<TestFlags>({"D"});
        EXPECT(false);
    }
    catch (const RuntimeError&)
    {
        EXPECT(true);
    }
}
} // namespace Falcor
