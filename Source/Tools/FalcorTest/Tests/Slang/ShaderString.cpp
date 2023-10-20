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
#include <random>

namespace Falcor
{
namespace
{
const char kShaderModuleA[] =
    "struct A\n"
    "{\n"
    "    ByteAddressBuffer buf;\n"
    "    uint c;\n"
    "    uint f(uint i)\n"
    "    {\n"
    "        return c * buf.Load(i * 4);\n"
    "    }\n"
    "}\n";

const char kShaderModuleB[] =
    "import ShaderStringUtil;\n"
    "uint f(uint i)\n"
    "{\n"
    "    return test(i);\n"
    "}\n";

const char kShaderModuleC[] =
    "import Tests.Slang.ShaderStringUtil;\n"
    "uint f(uint i)\n"
    "{\n"
    "    return test(i);\n"
    "}\n";

const char kShaderModuleD[] =
    "uint f(uint i)\n"
    "{\n"
    "    return i * 997;\n"
    "}\n";

const uint32_t kSize = 32;
} // namespace

GPU_TEST(ShaderStringInline)
{
    ref<Device> pDevice = ctx.getDevice();

    // Create program with generated code placed inline in the same translation
    // unit as the entry point.
    ProgramDesc desc;
    desc.addShaderModule().addFile("Tests/Slang/ShaderStringInline.cs.slang").addString(kShaderModuleA);
    desc.csEntry("main");

    ctx.createProgram(desc, DefineList());
    ctx.allocateStructuredBuffer("result", kSize);

    // Create and bind test data.
    std::mt19937 r;
    std::vector<uint32_t> values(kSize);
    for (auto& v : values)
        v = r();

    auto buf =
        pDevice->createBuffer(values.size() * sizeof(uint32_t), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, values.data());
    auto var = ctx.vars().getRootVar();
    var["gTest"]["moduleA"]["buf"] = buf;
    var["gTest"]["moduleA"]["c"] = 991u;

    // Run program and validate results.
    ctx.runProgram(kSize, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kSize; i++)
    {
        EXPECT_EQ(result[i], values[i] * 991);
    }
}

GPU_TEST(ShaderStringModule)
{
    // Create program with generated code placed in another translation unit.
    // The generated code is imported as a module using a relative path.
    ProgramDesc desc;
    desc.addShaderModule("GeneratedModule").addString(kShaderModuleD, "Tests/Slang/GeneratedModule.slang");
    desc.addShaderModule().addFile("Tests/Slang/ShaderStringModule.cs.slang");
    desc.csEntry("main");

    ctx.createProgram(desc, DefineList());
    ctx.allocateStructuredBuffer("result", kSize);

    // Run program and validate results.
    ctx.runProgram(kSize, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kSize; i++)
    {
        EXPECT_EQ(result[i], i * 997);
    }
}

GPU_TEST(ShaderStringImport)
{
    // Create program with generated code placed inline in the same translation
    // unit as the entry point. The generated code imports another module using an absolute path.
    ProgramDesc desc;
    desc.addShaderModule().addFile("Tests/Slang/ShaderStringImport.cs.slang").addString(kShaderModuleC);
    desc.csEntry("main");

    ctx.createProgram(desc, DefineList());
    ctx.allocateStructuredBuffer("result", kSize);

    // Run program and validate results.
    ctx.runProgram(kSize, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kSize; i++)
    {
        EXPECT_EQ(result[i], i * 993);
    }
}

GPU_TEST(ShaderStringImportDuplicate, "Duplicate import not working")
{
    // Create program with generated code placed inline in the same translation
    // unit as the entry point. The generated code imports another module using an absolute path.
    // The main translation unit imports the same module. This currently does not work.
    ProgramDesc desc;
    desc.addShaderModule().addFile("Tests/Slang/ShaderStringImport.cs.slang").addString(kShaderModuleC);
    desc.csEntry("main");

    ctx.createProgram(desc, {{"IMPORT_FROM_MAIN", "1"}});
    ctx.allocateStructuredBuffer("result", kSize);

    // Run program and validate results.
    ctx.runProgram(kSize, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kSize; i++)
    {
        EXPECT_EQ(result[i], i * 993);
    }
}

GPU_TEST(ShaderStringImported)
{
    // Create program with generated code placed in a new translation unit.
    // The program imports a module that imports the generated module.
    ProgramDesc desc;
    desc.addShaderModule("GeneratedModule").addString(kShaderModuleD, "Tests/Slang/GeneratedModule.slang");
    desc.addShaderModule().addFile("Tests/Slang/ShaderStringImported.cs.slang");
    desc.csEntry("main");

    ctx.createProgram(desc, DefineList());
    ctx.allocateStructuredBuffer("result", kSize);

    // Run program and validate results.
    ctx.runProgram(kSize, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kSize; i++)
    {
        EXPECT_EQ(result[i], i * 997);
    }
}

GPU_TEST(ShaderStringDynamicObject)
{
    const uint32_t typeID = 55;

    // Create program with generated code placed in a new translation unit.
    // The program imports a module that imports the generated module.
    // The generated code is called from a dynamically created object.
    ProgramDesc desc;
    desc.addShaderModule("GeneratedModule").addString(kShaderModuleD);
    desc.addShaderModule().addFile("Tests/Slang/ShaderStringDynamic.cs.slang");
    desc.csEntry("main");

    TypeConformanceList typeConformances = TypeConformanceList{{{"DynamicType", "IDynamicType"}, typeID}};
    desc.addTypeConformances(typeConformances);

    ctx.createProgram(desc, DefineList());
    ctx.allocateStructuredBuffer("result", kSize);

    auto var = ctx.vars().getRootVar();
    var["CB"]["type"] = typeID;

    // Run program and validate results.
    ctx.runProgram(kSize, 1, 1);

    std::vector<uint32_t> result = ctx.readBuffer<uint32_t>("result");
    for (uint32_t i = 0; i < kSize; i++)
    {
        EXPECT_EQ(result[i], i * 997);
    }
}
} // namespace Falcor
