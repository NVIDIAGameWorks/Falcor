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
#include "Core/Plugin.h"

namespace Falcor
{

/// Base class for plugin type A.
class PluginBaseA
{
public:
    struct PluginInfo
    {
        std::string desc;
    };

    using PluginCreate = std::function<std::shared_ptr<PluginBaseA>(const std::string&)>;

    FALCOR_PLUGIN_BASE_CLASS(PluginBaseA);

    virtual ~PluginBaseA() {}
    virtual const std::string& getText() const = 0;
};

/// First plugin of type A.
class PluginA1 : public PluginBaseA
{
public:
    FALCOR_PLUGIN_CLASS(PluginA1, "PluginA1", "This is PluginA1");

    static std::shared_ptr<PluginA1> create(const std::string& text) { return std::make_shared<PluginA1>(text); }

    PluginA1(const std::string& text) : mText(text) {}

    const std::string& getText() const { return mText; }

private:
    std::string mText;
};

/// Second plugin of type A.
class PluginA2 : public PluginBaseA
{
public:
    FALCOR_PLUGIN_CLASS(PluginA2, "PluginA2", "This is PluginA2");

    static std::shared_ptr<PluginA2> create(const std::string& text) { return std::make_shared<PluginA2>(text); }

    PluginA2(const std::string& text) : mText(text + "\n" + text) {}

    const std::string& getText() const { return mText; }

private:
    std::string mText;
};

/// Base class for plugin type B.
/// Here we use a different create function that returns pointers.
class PluginBaseB
{
public:
    struct PluginInfo
    {
        std::string name;
        std::vector<int> sequence;
    };

    using PluginCreate = PluginBaseB* (*)();

    FALCOR_PLUGIN_BASE_CLASS(PluginBaseB);

    virtual ~PluginBaseB() {}
};

/// First plugin of type B.
class PluginB1 : public PluginBaseB
{
public:
    FALCOR_PLUGIN_CLASS(PluginB1, "PluginB1", PluginInfo({"This is PluginB1", {1, 2, 4, 8}}));

    static PluginBaseB* create() { return new PluginB1(); }
};

/// Second plugin of type B.
class PluginB2 : public PluginBaseB
{
public:
    FALCOR_PLUGIN_CLASS(PluginB2, "PluginB2", PluginInfo({"This is PluginB2", {2, 4, 8, 16}}));

    static PluginBaseB* create() { return new PluginB2(); }
};

CPU_TEST(Plugin)
{
    PluginManager pm;

    // No classes of first type are registered yet.
    EXPECT(!pm.hasClass<PluginBaseA>("PluginA1"));
    EXPECT(!pm.hasClass<PluginBaseA>("PluginA2"));

    // Register plugins of first type.
    // Note: This is typically done within the plugin library in an exported registerPlugin function.
    {
        PluginRegistry registry(pm, 0);
        registry.registerClass<PluginBaseA, PluginA1>();
        registry.registerClass<PluginBaseA, PluginA2>();
    }

    // Check for registered classes of first type.
    EXPECT(pm.hasClass<PluginBaseA>("PluginA1"));
    EXPECT(pm.hasClass<PluginBaseA>("PluginA2"));

    // Check infos of first type.
    {
        bool hasPluginA1{false};
        bool hasPluginA2{false};
        size_t count = 0;
        for (const auto& [name, info] : pm.getInfos<PluginBaseA>())
        {
            if (name == "PluginA1")
                hasPluginA1 = info.desc == "This is PluginA1";
            else if (name == "PluginA2")
                hasPluginA2 = info.desc == "This is PluginA2";
            count++;
        }
        EXPECT(hasPluginA1 && hasPluginA2 && count == 2);
    }

    // Create plugins of first type.
    {
        auto pluginA1 = pm.createClass<PluginBaseA>("PluginA1", "Hello world");
        auto pluginA2 = pm.createClass<PluginBaseA>("PluginA2", "Hello world again");
        auto pluginA3 = pm.createClass<PluginBaseA>("PluginA3", ""); // does not exist

        EXPECT(pluginA1 != nullptr);
        EXPECT(pluginA2 != nullptr);
        EXPECT(pluginA3 == nullptr);

        EXPECT_EQ(pluginA1->getPluginType(), "PluginA1");
        EXPECT_EQ(pluginA1->getPluginInfo().desc, "This is PluginA1");
        EXPECT_EQ(pluginA1->getText(), "Hello world");

        EXPECT_EQ(pluginA2->getPluginType(), "PluginA2");
        EXPECT_EQ(pluginA2->getPluginInfo().desc, "This is PluginA2");
        EXPECT_EQ(pluginA2->getText(), "Hello world again\nHello world again");
    }

    // No classes of second type are registered yet.
    EXPECT(!pm.hasClass<PluginBaseB>("PluginB1"));
    EXPECT(!pm.hasClass<PluginBaseB>("PluginB2"));

    // Register plugins of second type.
    // Note: This is typically done within the plugin library in an exported registerPlugin function.
    {
        PluginRegistry registry(pm, 0);
        registry.registerClass<PluginBaseB, PluginB1>();
        registry.registerClass<PluginBaseB, PluginB2>();
    }

    // Check for registered classes of second type.
    EXPECT(pm.hasClass<PluginBaseB>("PluginB1"));
    EXPECT(pm.hasClass<PluginBaseB>("PluginB2"));

    // Check infos of second type.
    {
        bool hasPluginB1{false};
        bool hasPluginB2{false};
        size_t count = 0;
        for (const auto& [name, info] : pm.getInfos<PluginBaseB>())
        {
            if (name == "PluginB1")
                hasPluginB1 = info.sequence == std::vector<int>{1, 2, 4, 8};
            else if (name == "PluginB2")
                hasPluginB2 = info.sequence == std::vector<int>{2, 4, 8, 16};
            count++;
        }
        EXPECT(hasPluginB1 && hasPluginB2 && count == 2);
    }

    // Create plugins of second type.
    {
        auto pluginB1 = pm.createClass<PluginBaseB>("PluginB1");
        auto pluginB2 = pm.createClass<PluginBaseB>("PluginB2");
        auto pluginB3 = pm.createClass<PluginBaseB>("PluginB3"); // does not exist

        EXPECT(pluginB1 != nullptr);
        EXPECT(pluginB2 != nullptr);
        EXPECT(pluginB3 == nullptr);

        EXPECT_EQ(pluginB1->getPluginType(), "PluginB1");
        EXPECT(pluginB1->getPluginInfo().sequence == std::vector<int>({1, 2, 4, 8}));

        EXPECT_EQ(pluginB2->getPluginType(), "PluginB2");
        EXPECT(pluginB2->getPluginInfo().sequence == std::vector<int>({2, 4, 8, 16}));
    }
}

} // namespace Falcor
