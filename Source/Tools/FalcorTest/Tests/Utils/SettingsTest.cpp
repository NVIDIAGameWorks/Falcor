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
#include "Utils/Settings.h"

#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace Falcor
{
    CPU_TEST(OptionsIntBool)
    {
        pybind11::dict pyDict;
        pyDict["TrueAsBool"] = true;
        pyDict["TrueAsInt"]  = 1;
        pyDict["FalseAsBool"] = false;
        pyDict["FalseAsInt"]  = 0;

        Settings settings;
        settings.addOptions(pyDict);
        Properties options = settings.getOptions();

        EXPECT_EQ(options.get("TrueAsBool", false), true);
        EXPECT_EQ(options.get("TrueAsBool", 0),     1);
        EXPECT_EQ(options.get("TrueAsInt", false), true);
        EXPECT_EQ(options.get("TrueAsInt", 0),     1);

        EXPECT_EQ(options.get("FalseAsBool", true), false);
        EXPECT_EQ(options.get("FalseAsBool", 1),     0);
        EXPECT_EQ(options.get("FalseAsInt", true), false);
        EXPECT_EQ(options.get("FalseAsInt", 1),     0);
    }

    CPU_TEST(OptionsNesting)
    {
        pybind11::dict pyDict;
        pyDict["mogwai"] = pybind11::dict();
        pyDict["mogwai"]["value"] = 17;

        Settings settings;
        settings.addOptions(pyDict);
        Properties options = settings.getOptions();

        {
            Properties mogwai = options.get("mogwai", Properties());
            EXPECT_EQ(mogwai.get("value", 0), 17);
        }

        EXPECT(options.get<Properties>("mogwai"));
        EXPECT(!options.get<Properties>("user"));

        if (auto local = options.get<Properties>("mogwai"))
        {
            EXPECT_EQ(local->get("value", 0), 17);
        }
        else
        {
            EXPECT(false); // never get here
        }

        EXPECT_EQ(options.get("mogwai:value", 0), 17);
    }

    CPU_TEST(OptionsTypes)
    {
        pybind11::dict pyDict;
        pyDict["string"] = std::string("string");
        pyDict["float"]  = 1.f;
        pyDict["int"]    = 2;
        pyDict["bool"]   = true;
        pyDict["int[2]"] = std::array<int,2>{1, 2};

        Settings settings;
        settings.addOptions(pyDict);
        Properties options = settings.getOptions();

        EXPECT_EQ(options.get("string", std::string()), "string");
        EXPECT_EQ(options.get("float", 0.f), 1.f);
        EXPECT_EQ(options.get("int", 0), 2);
        EXPECT_EQ(options.get("int", 0.f), 2.f); // check we convert to float
        EXPECT_EQ(options.get("bool", false), true);

        // Can't have comma in macro
        const std::array<int,2> invalidTuple({-1, -2});
        const std::array<int,2> validTuple({1, 2});
        const std::array<int,2> result = options.get("int[2]", invalidTuple);
        EXPECT_EQ(result[0], validTuple[0]);
        EXPECT_EQ(result[1], validTuple[1]);

        bool wrongTypeString = false;
        try
        {
            options.get("string", int(3));
        }
        catch(Falcor::Properties::TypeError err)
        {
            wrongTypeString = true;
        }
        EXPECT(wrongTypeString);

        bool wrongTypeInt = false;
        try
        {
            options.get("int", std::string("test"));
        }
        catch(Falcor::Properties::TypeError err)
        {
            wrongTypeInt = true;
        }
        EXPECT(wrongTypeInt);

        bool wrongTypeArray = false;
        try
        {
            options.get("int[2]", float(0.f));
        }
        catch(Falcor::Properties::TypeError err)
        {
            wrongTypeArray = true;
        }
        EXPECT(wrongTypeArray);
    }

    CPU_TEST(OptionsOverride)
    {
        pybind11::dict pyDict;
        pyDict["mogwai"] = pybind11::dict();
        pyDict["mogwai"]["value"] = 17;

        Settings settings;
        settings.addOptions(pyDict);
        Properties options = settings.getOptions();

        EXPECT_EQ(options.get("mogwai:value", 0), 17);

        {
            pybind11::dict pyDict;
            pyDict["mogwai"] = pybind11::dict();
            pyDict["mogwai"]["string"] = "test";
            settings.addOptions(pyDict);
        }
        // reget to make sure we get the update
        options = settings.getOptions();

        // Check the original value
        EXPECT_EQ(options.get("mogwai:value", 0), 17);
        // Check the added value
        EXPECT_EQ(options.get("mogwai:string", std::string("foo")), "test");

        {
            pybind11::dict pyDict;
            pyDict["mogwai"] = pybind11::dict();
            pyDict["mogwai"]["string"] = "test2";
            settings.addOptions(pyDict);
        }

        options = settings.getOptions();

        EXPECT_EQ(options.get("mogwai:value", 0), 17);
        EXPECT_EQ(options.get("mogwai:string", std::string("foo")), "test2");

        {
            pybind11::dict pyDict;
            pyDict["mogwai"] = pybind11::dict();
            pyDict["mogwai"]["string"] = 14;
            settings.addOptions(pyDict);
        }
        options = settings.getOptions();

        EXPECT_EQ(options.get("mogwai:value", 0), 17);
        EXPECT_EQ(options.get("mogwai:string", 0), 14);

        {
            pybind11::dict pyDict;
            pyDict["mogwai"] = 2;
            settings.addOptions(pyDict);
        }
        options = settings.getOptions();

        EXPECT(!options.has("mogwai:value"));
        EXPECT(!options.has("mogwai:string"));

        EXPECT_EQ(options.get("mogwai", 0), 2);
    }

    pybind11::list makeList(const std::string_view& expression, bool negateRegex = false)
    {
        pybind11::list result;
        result.append(expression);
        if (negateRegex)
            result.append(negateRegex);
        return result;
    }

    struct Shape
    {
        std::string name;
        bool        motionEnabled { true };
        bool        deduplicateVerts { false };
        int         verbosity { 0 };
        float       multiplyEmission { 1 };
    };

    CPU_TEST(AttributeAssign)
    {
        pybind11::dict pyDict;
        pyDict["usdImporter"] = pybind11::dict();

        // No filter
        pyDict["usdImporter"]["verbosity"] = 5;
        // Filter is: [expression, bool]
        pyDict["usdImporter"]["motionEnable"] = false;
        pyDict["usdImporter"]["motionEnable.filter"] = makeList("/World/Tiger.*", true);

        // Filter is: expression
        pyDict["usdImporter"]["deduplicateVerts"] = true;
        pyDict["usdImporter"]["deduplicateVerts.filter"] = "/World/Tiger_Fur.*";

        // Filter is: [expression]
        pyDict["usdImporter"]["multiplyEmission"] = 3;
        pyDict["usdImporter"]["multiplyEmission.filter"] = makeList("lights/.*");

        Settings settings;
        settings.addFilteredAttributes(pyDict);

        std::vector<Shape> shapes;
        shapes.emplace_back(Shape{"/World/Tiger/Body"});
        shapes.emplace_back(Shape{"/World/Tiger_Fur/back"});
        shapes.emplace_back(Shape{"/World/Ground"});
        shapes.emplace_back(Shape{"lights/dome"});

        for(Shape& shape : shapes)
        {
            shape.motionEnabled    = settings.getAttribute(shape.name, "usdImporter:motionEnable", shape.motionEnabled);
            shape.deduplicateVerts = settings.getAttribute(shape.name, "usdImporter:deduplicateVerts", shape.deduplicateVerts);
            shape.verbosity        = settings.getAttribute(shape.name, "usdImporter:verbosity", shape.verbosity);
            shape.multiplyEmission = settings.getAttribute(shape.name, "usdImporter:multiplyEmission", shape.multiplyEmission);
        }

        EXPECT_EQ(shapes[0].name,             "/World/Tiger/Body");
        EXPECT_EQ(shapes[0].motionEnabled,    true);
        EXPECT_EQ(shapes[0].deduplicateVerts, false);
        EXPECT_EQ(shapes[0].verbosity,        5);
        EXPECT_EQ(shapes[0].multiplyEmission, 1);

        EXPECT_EQ(shapes[1].name,             "/World/Tiger_Fur/back");
        EXPECT_EQ(shapes[1].motionEnabled,    true);
        EXPECT_EQ(shapes[1].deduplicateVerts, true);
        EXPECT_EQ(shapes[1].verbosity,        5);
        EXPECT_EQ(shapes[1].multiplyEmission, 1);

        EXPECT_EQ(shapes[2].name,             "/World/Ground");
        EXPECT_EQ(shapes[2].motionEnabled,    false);
        EXPECT_EQ(shapes[2].deduplicateVerts, false);
        EXPECT_EQ(shapes[2].verbosity,        5);
        EXPECT_EQ(shapes[2].multiplyEmission, 1);

        EXPECT_EQ(shapes[3].name,             "lights/dome");
        EXPECT_EQ(shapes[3].motionEnabled,    false);
        EXPECT_EQ(shapes[3].deduplicateVerts, false);
        EXPECT_EQ(shapes[3].verbosity,        5);
        EXPECT_EQ(shapes[3].multiplyEmission, 3);
    }
}
