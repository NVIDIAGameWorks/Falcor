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
#include "Core/Error.h"
#include "Utils/StringUtils.h"
#include "Testing/UnitTest.h"

#include <args.hxx>

#include <cstdio>
#include <string>
#include <vector>

using namespace Falcor;

FALCOR_EXPORT_D3D12_AGILITY_SDK

int runMain(int argc, char** argv)
{
    args::ArgumentParser parser("Falcor unit tests.");
    parser.helpParams.programName = "FalcorTest";
    args::HelpFlag helpFlag(parser, "help", "Display this help menu.", {'h', "help"});
    args::ValueFlag<uint32_t> parallelFlag(parser, "N", "EXPERIMENTAL: Number of worker threads (default: 1).", {'p', "parallel"});
    args::ValueFlag<std::string> deviceTypeFlag(parser, "d3d12|vulkan", "Graphics device type.", {'d', "device-type"});
    args::Flag listGPUsFlag(parser, "", "List available GPUs", {"list-gpus"});
    args::ValueFlag<uint32_t> gpuFlag(parser, "index", "Select specific GPU to use", {"gpu"});
    args::Flag listTestSuites(parser, "", "List test suites", {"list-test-suites"});
    args::Flag listTestCases(parser, "", "List test cases", {"list-test-cases"});
    args::Flag listTags(parser, "", "List tags", {"list-tags"});
    args::ValueFlag<std::string> testSuiteFilterFlag(parser, "regex", "Filter test suites to run.", {'s', "test-suite"});
    args::ValueFlag<std::string> testCaseFilterFlag(parser, "regex", "Filter test cases to run.", {'f', "test-case"});
    args::ValueFlag<std::string> tagFilterFlag(parser, "tags", "Filter test cases by tags.", {'t', "tags"});
    args::ValueFlag<std::string> xmlReportFlag(parser, "path", "XML report output file.", {'x', "xml-report"});
    args::ValueFlag<uint32_t> repeatFlag(parser, "N", "Number of times to repeat the test.", {'r', "repeat"});
    args::Flag enableDebugLayerFlag(parser, "", "Enable debug layer (enabled by default in Debug build).", {"enable-debug-layer"});
    args::Flag enableAftermathFlag(parser, "", "Enable Aftermath GPU crash dump.", {"enable-aftermath"});

    args::CompletionFlag completionFlag(parser, {"complete"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (const args::RequiredError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    unittest::RunOptions options;

    if (deviceTypeFlag)
    {
        if (args::get(deviceTypeFlag) == "d3d12")
            options.deviceDesc.type = Device::Type::D3D12;
        else if (args::get(deviceTypeFlag) == "vulkan")
            options.deviceDesc.type = Device::Type::Vulkan;
        else
        {
            std::cerr << "Invalid device type, use 'd3d12' or 'vulkan'" << std::endl;
            return 1;
        }
    }
    if (listGPUsFlag)
    {
        const auto gpus = Device::getGPUs(options.deviceDesc.type);
        for (size_t i = 0; i < gpus.size(); ++i)
            fmt::print("GPU {}: {}\n", i, gpus[i].name);
        return 0;
    }

    if (gpuFlag)
        options.deviceDesc.gpu = args::get(gpuFlag);
    if (enableDebugLayerFlag)
        options.deviceDesc.enableDebugLayer = true;
    if (enableAftermathFlag)
        options.deviceDesc.enableAftermath = true;

    if (testSuiteFilterFlag)
        options.testSuiteFilter = args::get(testSuiteFilterFlag);
    if (testCaseFilterFlag)
        options.testCaseFilter = args::get(testCaseFilterFlag);
    if (tagFilterFlag)
        options.tagFilter = args::get(tagFilterFlag);
    if (xmlReportFlag)
        options.xmlReportPath = args::get(xmlReportFlag);
    if (parallelFlag)
        options.parallel = args::get(parallelFlag);
    if (repeatFlag)
        options.repeat = args::get(repeatFlag);

    if (listTestSuites || listTestCases || listTags)
    {
        std::vector<unittest::Test> tests = unittest::enumerateTests();
        tests = unittest::filterTests(tests, options.testSuiteFilter, options.testCaseFilter, options.tagFilter, options.deviceDesc.type);

        if (listTestSuites)
        {
            std::set<std::string> suites;
            for (const auto& test : tests)
                suites.insert(test.suiteName);
            for (const auto& suite : suites)
                fmt::print("{}\n", suite);
            return 0;
        }
        if (listTestCases)
        {
            for (const auto& test : tests)
                fmt::print("{}:{}\n", test.suiteName, test.name);
            return 0;
        }
        if (listTags)
        {
            std::set<std::string> tags;
            for (const auto& test : tests)
                for (const auto& tag : test.tags)
                    tags.insert(tag);
            for (const auto& tag : tags)
                fmt::print("{}\n", tag);
            return 0;
        }
    }

    // Setup error diagnostics to not break on exceptions.
    // We might have unit tests that check for exceptions, so we want to throw
    // them without breaking into the debugger in order to let tests run
    // uninterrupted with the debugger attached. The test framework will
    // break into the debugger when a test conditions is not met.
    setErrorDiagnosticFlags(getErrorDiagnosticFlags() & ~ErrorDiagnosticFlags::BreakOnThrow);

    return unittest::runTests(options);
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
