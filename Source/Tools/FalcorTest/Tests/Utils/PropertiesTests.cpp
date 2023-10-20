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
#include "Utils/Properties.h"
#include "Utils/Logger.h"
#include "Utils/StringFormatters.h"

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Falcor
{
namespace PropertiesTest
{
enum class TestEnum
{
    A,
    B,
    C
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

struct TestStruct
{
    int a = 1;
    float b = 2.f;
    std::string c = "3";

    bool operator==(const TestStruct& other) const { return a == other.a && b == other.b && c == other.c; }

    template<typename Archive>
    void serialize(Archive& ar)
    {
        ar("a", a);
        ar("b", b);
        ar("c", c);
    }
};

} // namespace PropertiesTest
} // namespace Falcor

template<>
struct fmt::formatter<Falcor::PropertiesTest::TestStruct> : formatter<std::string>
{
    template<typename FormatContext>
    auto format(const Falcor::PropertiesTest::TestStruct& t, FormatContext& ctx)
    {
        return format_to(ctx.out(), "TestStruct{{a={}, b={}, c={}}}", t.a, t.b, t.c);
    }
};

namespace Falcor
{

template<typename T>
void testPropertyType(CPUUnitTestContext& ctx, const T& checkValue, const T& differentValue)
{
    Properties props;
    props.set("value", checkValue);

    // Test Properties::has(name)
    EXPECT(props.has("value"));
    EXPECT(!props.has("value2"));

    // Test Properties::get<T>(name)
    EXPECT_EQ(props.get<T>("value"), checkValue);
    EXPECT_THROW(props.get<T>("value2"));

    // Test Properties::get<T>(name, def)
    EXPECT_EQ(props.get<T>("value", differentValue), checkValue);
    EXPECT_EQ(props.get<T>("value2", differentValue), differentValue);

    // Test Properties::getOpt<T>(name)
    {
        auto optional = props.getOpt<T>("value");
        ASSERT(optional.has_value());
        EXPECT_EQ(optional.value(), checkValue);
        EXPECT(props.getOpt<T>("value2") == std::nullopt);
    }

    // Test Properties::getTo<T>(name, value)
    {
        T holderValue{differentValue};
        EXPECT(props.getTo<T>("value", holderValue));
        EXPECT_EQ(holderValue, checkValue);
    }
    {
        T holderValue{differentValue};
        EXPECT(!props.getTo<T>("value2", holderValue));
        EXPECT_EQ(holderValue, differentValue);
    }
}

CPU_TEST(PropertiesBasicValues)
{
    testPropertyType<bool>(ctx, false, true);
    testPropertyType<bool>(ctx, true, false);

    testPropertyType<uint32_t>(ctx, std::numeric_limits<uint32_t>::lowest(), 1);
    testPropertyType<uint32_t>(ctx, std::numeric_limits<uint32_t>::max(), 1);

    testPropertyType<uint64_t>(ctx, std::numeric_limits<uint64_t>::lowest(), 1);
    testPropertyType<uint64_t>(ctx, std::numeric_limits<uint64_t>::max(), 1);

    testPropertyType<int32_t>(ctx, std::numeric_limits<int32_t>::lowest(), 1);
    testPropertyType<int32_t>(ctx, std::numeric_limits<int32_t>::max(), 1);

    testPropertyType<int64_t>(ctx, std::numeric_limits<int64_t>::lowest(), 1);
    testPropertyType<int64_t>(ctx, std::numeric_limits<int64_t>::max(), 1);

    testPropertyType<float>(ctx, std::numeric_limits<float>::lowest(), 1.f);
    testPropertyType<float>(ctx, std::numeric_limits<float>::max(), 1.f);

    testPropertyType<double>(ctx, std::numeric_limits<double>::lowest(), 1.0);
    testPropertyType<double>(ctx, std::numeric_limits<double>::max(), 1.0);

    testPropertyType<std::string>(ctx, "", " ");
    testPropertyType<std::string>(ctx, "test", "test2");

    {
        Properties emptyProps;
        Properties testProps;
        testProps.set("int", 123);
        testProps.set("str", "test");
        testPropertyType<Properties>(ctx, emptyProps, testProps);
        testPropertyType<Properties>(ctx, testProps, emptyProps);
    }

    testPropertyType<int2>(ctx, int2(-10, -11), int2(10, 11));
    testPropertyType<int3>(ctx, int3(-10, -11, -12), int3(10, 11, 12));
    testPropertyType<int4>(ctx, int4(-10, -11, -12, -13), int4(10, 11, 12, 13));

    testPropertyType<uint2>(ctx, uint2(10, 11), uint2(110, 111));
    testPropertyType<uint3>(ctx, uint3(10, 11, 12), uint3(110, 111, 112));
    testPropertyType<uint4>(ctx, uint4(10, 11, 12, 13), uint4(110, 111, 112, 113));

    testPropertyType<float2>(ctx, float2(-10.f, -11.f), float2(10.f, 11.f));
    testPropertyType<float3>(ctx, float3(-10.f, -11.f, -12.f), float3(10.f, 11.f, 12.f));
    testPropertyType<float4>(ctx, float4(-10.f, -11.f, -12.f, -13.f), float4(10.f, 11.f, 12.f, 13.f));

    testPropertyType<PropertiesTest::TestEnum>(ctx, PropertiesTest::TestEnum::B, PropertiesTest::TestEnum::C);

    testPropertyType<PropertiesTest::TestStruct>(ctx, PropertiesTest::TestStruct{}, PropertiesTest::TestStruct{2, 4.f, "6"});
}

CPU_TEST(PropertiesFromJson)
{
    Properties::json jnested = {
        {"str", "string"},
    };
    Properties::json j = {
        {"b", true},
        {"u32", std::numeric_limits<uint32_t>::max()},
        {"u64", std::numeric_limits<uint64_t>::max()},
        {"i32", std::numeric_limits<int32_t>::lowest()},
        {"i64", std::numeric_limits<int64_t>::lowest()},
        {"f32", std::numeric_limits<float>::max()},
        {"f64", std::numeric_limits<double>::max()},
        {"uint3", {1, 2, 3}},
        {"int3", {-1, 2, -3}},
        {"float3", {0.25f, 0.5f, 0.75f}},
        {"str", "string"},
        {"nested", jnested},
    };

    Properties props(j);
    EXPECT_EQ(props.get<bool>("b"), true);
    EXPECT_EQ(props.get<uint32_t>("u32"), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(props.get<uint64_t>("u64"), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(props.get<int32_t>("i32"), std::numeric_limits<int32_t>::lowest());
    EXPECT_EQ(props.get<int64_t>("i64"), std::numeric_limits<int64_t>::lowest());
    EXPECT_EQ(props.get<float>("f32"), std::numeric_limits<float>::max());
    EXPECT_EQ(props.get<double>("f64"), std::numeric_limits<double>::max());
    EXPECT_EQ(props.get<uint3>("uint3"), uint3(1, 2, 3));
    EXPECT_EQ(props.get<int3>("int3"), int3(-1, 2, -3));
    EXPECT_EQ(props.get<float3>("float3"), float3(0.25f, 0.5f, 0.75f));
    EXPECT_EQ(props.get<std::string>("str"), "string");
    EXPECT_EQ(props.get<Properties>("nested"), Properties(jnested));
}

CPU_TEST(PropertiesToJson)
{
    Properties::json jnested = {
        {"str", "string"},
    };
    Properties::json j = {
        {"b", true},
        {"u32", std::numeric_limits<uint32_t>::max()},
        {"u64", std::numeric_limits<uint64_t>::max()},
        {"i32", std::numeric_limits<int32_t>::lowest()},
        {"i64", std::numeric_limits<int64_t>::lowest()},
        {"f32", std::numeric_limits<float>::max()},
        {"f64", std::numeric_limits<double>::max()},
        {"uint3", {1, 2, 3}},
        {"int3", {-1, 2, -3}},
        {"float3", {0.25f, 0.5f, 0.75f}},
        {"str", "string"},
        {"nested", jnested},
    };

    Properties props;
    props.set("b", true);
    props.set("u32", std::numeric_limits<uint32_t>::max());
    props.set("u64", std::numeric_limits<uint64_t>::max());
    props.set("i32", std::numeric_limits<int32_t>::lowest());
    props.set("i64", std::numeric_limits<int64_t>::lowest());
    props.set("f32", std::numeric_limits<float>::max());
    props.set("f64", std::numeric_limits<double>::max());
    props.set("uint3", uint3(1, 2, 3));
    props.set("int3", int3(-1, 2, -3));
    props.set("float3", float3(0.25f, 0.5f, 0.75f));
    props.set("str", "string");
    props.set("nested", Properties(jnested));
    EXPECT_EQ(props.toJson(), j);
}

CPU_TEST(PropertiesFromPython)
{
    pybind11::dict pnested;
    pnested["str"] = "string";
    pybind11::dict p;
    p["b"] = true;
    p["u32"] = std::numeric_limits<uint32_t>::max();
    p["u64"] = std::numeric_limits<uint64_t>::max();
    p["i32"] = std::numeric_limits<int32_t>::lowest();
    p["i64"] = std::numeric_limits<int64_t>::lowest();
    p["f32"] = std::numeric_limits<float>::max();
    p["f64"] = std::numeric_limits<double>::max();
    p["uint3"] = std::vector<uint32_t>{1, 2, 3};
    p["int3"] = std::vector<int32_t>{-1, 2, -3};
    p["float3"] = std::vector<float>{0.25f, 0.5f, 0.75f};
    p["str"] = "string";
    p["nested"] = pnested;

    Properties props(p);
    logInfo("{}", props.dump());
    EXPECT_EQ(props.get<bool>("b"), true);
    EXPECT_EQ(props.get<uint32_t>("u32"), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(props.get<uint64_t>("u64"), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(props.get<int32_t>("i32"), std::numeric_limits<int32_t>::lowest());
    EXPECT_EQ(props.get<int64_t>("i64"), std::numeric_limits<int64_t>::lowest());
    EXPECT_EQ(props.get<float>("f32"), std::numeric_limits<float>::max());
    EXPECT_EQ(props.get<double>("f64"), std::numeric_limits<double>::max());
    EXPECT_EQ(props.get<uint3>("uint3"), uint3(1, 2, 3));
    EXPECT_EQ(props.get<int3>("int3"), int3(-1, 2, -3));
    EXPECT_EQ(props.get<float3>("float3"), float3(0.25f, 0.5f, 0.75f));
    EXPECT_EQ(props.get<std::string>("str"), "string");
    EXPECT_EQ(props.get<Properties>("nested"), Properties(pnested));
}

CPU_TEST(PropertiesToPython)
{
    pybind11::dict pnested;
    pnested["str"] = "string";
    pybind11::dict p;
    p["b"] = true;
    p["u32"] = std::numeric_limits<uint32_t>::max();
    p["u64"] = std::numeric_limits<uint64_t>::max();
    p["i32"] = std::numeric_limits<int32_t>::lowest();
    p["i64"] = std::numeric_limits<int64_t>::lowest();
    p["f32"] = std::numeric_limits<float>::max();
    p["f64"] = std::numeric_limits<double>::max();
    p["uint3"] = std::vector<uint32_t>{1, 2, 3};
    p["int3"] = std::vector<int32_t>{-1, 2, -3};
    p["float3"] = std::vector<float>{0.25f, 0.5f, 0.75f};
    p["str"] = "string";
    p["nested"] = pnested;

    Properties props;
    props.set("b", true);
    props.set("u32", std::numeric_limits<uint32_t>::max());
    props.set("u64", std::numeric_limits<uint64_t>::max());
    props.set("i32", std::numeric_limits<int32_t>::lowest());
    props.set("i64", std::numeric_limits<int64_t>::lowest());
    props.set("f32", std::numeric_limits<float>::max());
    props.set("f64", std::numeric_limits<double>::max());
    props.set("uint3", uint3(1, 2, 3));
    props.set("int3", int3(-1, 2, -3));
    props.set("float3", float3(0.25f, 0.5f, 0.75f));
    props.set("str", "string");
    props.set("nested", Properties(pnested));
    EXPECT(props.toPython().equal(p));
}

CPU_TEST(PropertiesAccessors)
{
    Properties props;
    props["a"] = 1;
    props["b"] = 2.f;
    props["c"] = "3";

    int a = props["a"];
    EXPECT_EQ(a, 1);
    float b = props["b"];
    EXPECT_EQ(b, 2.f);
    std::string c = props["c"];
    EXPECT_EQ(c, "3");
}

CPU_TEST(PropertiesIterators)
{
    {
        Properties props;
        props["a"] = 1;
        props["b"] = 2.f;
        props["c"] = "3";

        int index = 0;
        for (const auto& [key, value] : props)
        {
            switch (index)
            {
            case 0:
                EXPECT_EQ(key, "a");
                EXPECT_EQ(value.operator int(), 1);
                value = 2;
                EXPECT_EQ(value.operator int(), 2);
                break;
            case 1:
                EXPECT_EQ(key, "b");
                EXPECT_EQ(value.operator float(), 2.f);
                value = 4.f;
                EXPECT_EQ(value.operator float(), 4.f);
                break;
            case 2:
                EXPECT_EQ(key, "c");
                EXPECT_EQ(value.operator std::string(), "3");
                value = "6";
                EXPECT_EQ(value.operator std::string(), "6");
                break;
            }
            ++index;
        }
    }

    {
        const Properties props = []()
        {
            Properties props;
            props["a"] = 1;
            props["b"] = 2.f;
            props["c"] = "3";
            return props;
        }();

        int index = 0;
        for (const auto& [key, value] : props)
        {
            switch (index)
            {
            case 0:
                EXPECT_EQ(key, "a");
                EXPECT_EQ(value.operator int(), 1);
                break;
            case 1:
                EXPECT_EQ(key, "b");
                EXPECT_EQ(value.operator float(), 2.f);
                break;
            case 2:
                EXPECT_EQ(key, "c");
                EXPECT_EQ(value.operator std::string(), "3");
                break;
            }
            ++index;
        }
    }
}

struct NestedStruct
{
    int a = 11;
    float b = 22.f;
    std::string c = "33";

    template<typename Archive>
    void serialize(Archive& ar)
    {
        ar("a", a);
        ar("b", b);
        ar("c", c);
    }
};

struct TestStruct
{
    int a = 1;
    float b = 2.f;
    std::string c = "3";
    NestedStruct nested;

    template<typename Archive>
    void serialize(Archive& ar)
    {
        ar("a", a);
        ar("b", b);
        ar("c", c);
        ar("nested", nested);
    }
};

CPU_TEST(PropertiesSerialization)
{
    static_assert(detail::has_serialize_v<int> == false);
    static_assert(detail::has_serialize_v<TestStruct> == true);

    TestStruct ts;
    Properties props = PropertiesWriter::write(ts);
    EXPECT_EQ(props.get<int>("a"), 1);
    EXPECT_EQ(props.get<float>("b"), 2.f);
    EXPECT_EQ(props.get<std::string>("c"), "3");
    Properties nestedProps = props.get<Properties>("nested");
    EXPECT_EQ(nestedProps.get<int>("a"), 11);
    EXPECT_EQ(nestedProps.get<float>("b"), 22.f);
    EXPECT_EQ(nestedProps.get<std::string>("c"), "33");

    props.set("a", 2);
    props.set("b", 4.f);
    props.set("c", "6");
    nestedProps.set("a", 22);
    nestedProps.set("b", 44.f);
    nestedProps.set("c", "66");
    props.set("nested", nestedProps);
    ts = PropertiesReader::read<TestStruct>(props);

    EXPECT_EQ(ts.a, 2);
    EXPECT_EQ(ts.b, 4.f);
    EXPECT_EQ(ts.c, "6");
    EXPECT_EQ(ts.nested.a, 22);
    EXPECT_EQ(ts.nested.b, 44.f);
    EXPECT_EQ(ts.nested.c, "66");
}

} // namespace Falcor
