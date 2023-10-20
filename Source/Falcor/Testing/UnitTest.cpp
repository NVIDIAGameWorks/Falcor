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
#include "UnitTest.h"
#include "Core/Version.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/Program/ProgramManager.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Threading.h"
#include "Utils/StringUtils.h"
#include "Utils/TermColor.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Vector.h"

#include <fmt/format.h>
#include <fmt/color.h>
#include <pugixml.hpp>
#include <BS_thread_pool_light.hpp>

#include <algorithm>
#include <chrono>
#include <regex>
#include <cstdint>

namespace Falcor
{
namespace unittest
{

struct TestDesc
{
    std::filesystem::path path;
    std::string name;
    unittest::Options options;
    CPUTestFunc cpuFunc;
    GPUTestFunc gpuFunc;
};

struct TestResult
{
    enum class Status
    {
        Passed,
        Failed,
        Skipped
    };

    Status status;
    std::vector<std::string> messages;
    std::string extraMessage;
    uint64_t elapsedMS = 0;
};

static std::vector<TestDesc>& getTestRegistry()
{
    static std::vector<TestDesc> registry;
    return registry;
}

class DevicePool
{
public:
    DevicePool(Device::Desc defaultDesc) : mDefaultDesc(defaultDesc) {}

    ref<Device> acquireDevice(Device::Type deviceType)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto& devices = mDevices[deviceType];
        if (devices.empty())
        {
            Device::Desc desc = mDefaultDesc;
            desc.type = deviceType;
            return make_ref<Device>(desc);
        }
        auto device = devices.back();
        devices.pop_back();
        return device;
    }

    void releaseDevice(ref<Device>&& device)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mDevices[device->getType()].push_back(device);
    }

private:
    Device::Desc mDefaultDesc;
    std::mutex mMutex;
    std::map<Device::Type, std::vector<ref<Device>>> mDevices;
};

inline const char* plural(size_t count, const char* suffix)
{
    if (count == 1)
        return "";
    return suffix;
}

void registerCPUTest(std::filesystem::path path, std::string name, unittest::Options options, CPUTestFunc func)
{
    TestDesc desc;
    desc.path = std::move(path);
    desc.name = std::move(name);
    desc.options = std::move(options);
    desc.cpuFunc = std::move(func);
    getTestRegistry().push_back(desc);
}

void registerGPUTest(std::filesystem::path path, std::string name, unittest::Options options, GPUTestFunc func)
{
    TestDesc desc;
    desc.path = std::move(path);
    desc.name = std::move(name);
    desc.options = std::move(options);
    desc.gpuFunc = std::move(func);
    getTestRegistry().push_back(desc);
}

/// Prints the UnitTest report line, making sure it is always printed to the console once.
template<typename... Args>
void reportLine(const std::string_view format, Args&&... args)
{
    std::string report = fmt::vformat(format, fmt::make_format_args(std::forward<Args>(args)...));
    bool willLogPrint = is_set(Logger::getOutputs(), Logger::OutputFlags::Console) ||
                        (is_set(Logger::getOutputs(), Logger::OutputFlags::DebugWindow) && isDebuggerPresent());

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (!willLogPrint)
    {
        std::cout << report << std::endl;
        std::flush(std::cout);
    }
    logInfo(report);
}

/**
 * Write a test report in JUnit's XML format.
 * @param[in] path File path.
 * @param[in] report List of tests/results.
 */
inline void writeXmlReport(const std::filesystem::path& path, const std::vector<std::pair<Test, TestResult>>& report)
{
    pugi::xml_document doc;

    pugi::xml_node testsuitesNode = doc.append_child("testsuites");
    testsuitesNode.append_attribute("name").set_value("Unit Tests");

    // Split reports into suites.
    std::map<std::string, std::vector<std::pair<Test, TestResult>>> reportBySuite;
    for (const auto& item : report)
        reportBySuite[item.first.suiteName].push_back(item);

    for (const auto& [suiteName, suiteReport] : reportBySuite)
    {
        pugi::xml_node testsuiteNode = testsuitesNode.append_child("testsuite");
        testsuiteNode.append_attribute("name").set_value(suiteName.c_str());

        for (const auto& [test, result] : suiteReport)
        {
            pugi::xml_node testcaseNode = testsuiteNode.append_child("testcase");
            testcaseNode.append_attribute("name").set_value(test.name.c_str());
            testcaseNode.append_attribute("time").set_value(result.elapsedMS / 1000.0);

            switch (result.status)
            {
            case TestResult::Status::Passed:
                break;
            case TestResult::Status::Skipped:
                testcaseNode.append_child("skipped");
                break;
            case TestResult::Status::Failed:
            default:
            {
                std::string message = joinStrings(result.messages, "\n");
                testcaseNode.append_child("failure").append_attribute("message").set_value(message.c_str());
            }
            break;
            }
        }
    }

    doc.save_file(path.native().c_str());
}

inline TestResult runTest(const Test& test, DevicePool& devicePool)
{
    if (!test.skipMessage.empty())
        return {TestResult::Status::Skipped, {test.skipMessage}};

    TestResult result{TestResult::Status::Passed};

    ref<Device> pDevice;
    if (test.gpuFunc)
        pDevice = devicePool.acquireDevice(test.deviceType);

    CPUUnitTestContext cpuCtx;
    GPUUnitTestContext gpuCtx(pDevice);

    auto startTime = std::chrono::steady_clock::now();

    try
    {
        if (test.cpuFunc)
            test.cpuFunc(cpuCtx);
        else
            test.gpuFunc(gpuCtx);
    }
    catch (const SkippingTestException& e)
    {
        result.status = TestResult::Status::Skipped;
        result.extraMessage = e.what();
    }
    catch (const ErrorRunningTestException& e)
    {
        result.status = TestResult::Status::Failed;
        result.extraMessage = e.what();
    }
    catch (const AssertingTestException& e)
    {
        result.status = TestResult::Status::Failed;
        result.extraMessage = e.what();
    }
    catch (const TooManyFailedTestsException&)
    {
        result.status = TestResult::Status::Failed;
        result.extraMessage = "Gave up after " + std::to_string(kMaxTestFailures) + " failures.";
    }
    catch (const std::exception& e)
    {
        result.status = TestResult::Status::Failed;
        result.extraMessage = e.what();
    }

    result.messages = test.cpuFunc ? cpuCtx.getFailureMessages() : gpuCtx.getFailureMessages();

    if (!result.messages.empty())
        result.status = TestResult::Status::Failed;

    if (!result.extraMessage.empty())
        result.messages.push_back(result.extraMessage);

    auto endTime = std::chrono::steady_clock::now();
    result.elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // Release GPU resources.
    if (pDevice)
    {
        pDevice->endFrame();
        pDevice->wait();
        devicePool.releaseDevice(std::move(pDevice));
    }

    return result;
}

inline int32_t runTestsParallel(const RunOptions& options)
{
    // Abort on Ctrl-C.
    std::atomic<bool> abort{false};
    setKeyboardInterruptHandler(
        [&abort]()
        {
            reportLine("\nDetected Ctrl-C, aborting ...\n");
            abort = true;
        }
    );

    auto startTime = std::chrono::steady_clock::now();

    DevicePool devicePool(options.deviceDesc);

    // Gather tests.
    std::vector<Test> tests = enumerateTests();
    tests = filterTests(tests, options.testSuiteFilter, options.testCaseFilter, options.tagFilter, options.deviceDesc.type);

    std::vector<TestResult> results(tests.size());

    BS::thread_pool_light threadPool(options.parallel);

    reportLine("[==========] Running {} test{}.", tests.size(), plural(tests.size(), "s"));

    for (size_t testIndex = 0; testIndex < tests.size(); ++testIndex)
    {
        threadPool.push_task(
            [&abort, &tests, &results, &devicePool, testIndex]()
            {
                if (abort)
                    return;

                const Test& test = tests[testIndex];
                TestResult& result = results[testIndex];
                std::string repeats;

                reportLine("[ RUN      ] {}:{}{}", test.suiteName, test.name, repeats);

                result = runTest(test, devicePool);

                std::string statusTag;
                switch (result.status)
                {
                case TestResult::Status::Passed:
                    statusTag = "[       OK ]";
                    break;
                case TestResult::Status::Failed:
                    statusTag = "[  FAILED  ]";
                    break;
                case TestResult::Status::Skipped:
                    statusTag = "[  SKIPPED ]";
                    break;
                }
                if (!result.extraMessage.empty())
                    reportLine("{}", result.extraMessage);
                reportLine("{} {}:{}{} ({} ms)", statusTag, test.suiteName, test.name, repeats, result.elapsedMS);
            }
        );
    }

    threadPool.wait_for_tasks();

    if (abort)
    {
        reportLine("[ ABORTED  ]");
        return 1;
    }

    auto endTime = std::chrono::steady_clock::now();
    uint64_t totalMS = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    int32_t failureCount = 0;
    for (const auto& result : results)
        failureCount += result.status == TestResult::Status::Failed ? 1 : 0;

    reportLine("[==========] {} test{} ran. ({} ms total)", tests.size(), plural(tests.size(), "s"), totalMS);
    reportLine("[  PASSED  ] {} test{}.", tests.size() - failureCount, plural(tests.size() - failureCount, "s"));
    if (failureCount > 0)
    {
        reportLine("[  FAILED  ] {} test{}, listed below.", failureCount, plural(failureCount, "s"));
        for (size_t i = 0; i < tests.size(); ++i)
            if (results[i].status == TestResult::Status::Failed)
                reportLine("[  FAILED  ] {}:{}", tests[i].suiteName, tests[i].name);
        reportLine("");
        reportLine("{} FAILED TEST{}", failureCount, plural(failureCount, "S"));
    }

    return failureCount;
}

inline int32_t runTestsSerial(const RunOptions& options)
{
    // Abort on Ctrl-C.
    std::atomic<bool> abort{false};
    setKeyboardInterruptHandler(
        [&abort]()
        {
            reportLine("\nDetected Ctrl-C, aborting ...\n");
            abort = true;
        }
    );

    auto startTime = std::chrono::steady_clock::now();

    DevicePool devicePool(options.deviceDesc);

    // Gather tests.
    std::vector<Test> tests = enumerateTests();
    tests = filterTests(tests, options.testSuiteFilter, options.testCaseFilter, options.tagFilter, options.deviceDesc.type);

    // Split tests into suites.
    std::map<std::string, std::vector<Test>> suites;
    for (const auto& test : tests)
        suites[test.suiteName].push_back(test);

    std::map<std::string, std::vector<Test>> failedTests;
    std::vector<std::pair<Test, TestResult>> report;

    size_t suiteCount = suites.size();
    size_t testCount = tests.size();
    int32_t failureCount = 0;
    reportLine(
        "[==========] Running {} test{} from {} test suite{}.", testCount, plural(testCount, "s"), suiteCount, plural(suiteCount, "s")
    );
    for (const auto& [suiteName, suiteTests] : suites)
    {
        if (abort)
            break;

        reportLine("[----------] {} test{} from {}", suiteTests.size(), plural(suiteTests.size(), "s"), suiteName);
        uint64_t suiteMS = 0;
        for (const auto& test : suiteTests)
        {
            if (abort)
                break;

            bool success = true;
            for (uint32_t repeatIndex = 0; repeatIndex < options.repeat; ++repeatIndex)
            {
                if (abort)
                    break;

                std::string repeats;
                if (options.repeat > 1)
                    repeats = fmt::format("[{}/{}]", repeatIndex + 1, options.repeat);
                reportLine("[ RUN      ] {}:{}{}", suiteName, test.name, repeats);
                TestResult result = runTest(test, devicePool);
                report.emplace_back(test, result);

                std::string statusTag;
                switch (result.status)
                {
                case TestResult::Status::Passed:
                    statusTag = "[       OK ]";
                    break;
                case TestResult::Status::Failed:
                    statusTag = "[  FAILED  ]";
                    break;
                case TestResult::Status::Skipped:
                    statusTag = "[  SKIPPED ]";
                    break;
                }
                if (!result.extraMessage.empty())
                    reportLine("{}", result.extraMessage);
                reportLine("{} {}:{}{} ({} ms)", statusTag, suiteName, test.name, repeats, result.elapsedMS);
                suiteMS += result.elapsedMS;
                if (success && result.status == TestResult::Status::Failed)
                {
                    failedTests[suiteName].push_back(test);
                    ++failureCount;
                    success = false;
                }
            }
        }
        reportLine("[----------] {} test{} from {} ({} ms total)", suiteTests.size(), plural(suiteTests.size(), "s"), suiteName, suiteMS);
        reportLine("");
    }

    if (abort)
    {
        reportLine("[ ABORTED  ]");
        return 1;
    }

    auto endTime = std::chrono::steady_clock::now();
    uint64_t totalMS = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    if (!options.xmlReportPath.empty())
        writeXmlReport(options.xmlReportPath, report);

    reportLine(
        "[==========] {} test{} from {} test suite{} ran. ({} ms total)",
        testCount,
        plural(testCount, "s"),
        suiteCount,
        plural(suiteCount, "s"),
        totalMS
    );
    reportLine("[  PASSED  ] {} test{}.", testCount - failureCount, plural(testCount - failureCount, "s"));
    if (failureCount > 0)
    {
        reportLine("[  FAILED  ] {} test{}, listed below.", failureCount, plural(failureCount, "s"));
        for (auto& suite : failedTests)
            for (auto& test : suite.second)
                reportLine("[  FAILED  ] {}:{}", suite.first, test.name);
        reportLine("");
        reportLine("{} FAILED TEST{}", failureCount, plural(failureCount, "S"));
    }

    return failureCount;
}

int32_t runTests(const RunOptions& options)
{
    // Disable logging to console, we don't want to clutter the test runner output with log messages.
    Logger::setOutputs(Logger::OutputFlags::File | Logger::OutputFlags::DebugWindow);

    logInfo("Falcor {}", getLongVersionString());

    OSServices::start();
    Threading::start();
    Scripting::start();

    int32_t failureCount = options.parallel > 1 ? runTestsParallel(options) : runTestsSerial(options);

    Scripting::shutdown();
    Threading::shutdown();
    OSServices::stop();

    return failureCount;
}

std::vector<Test> enumerateTests()
{
    std::vector<Test> tests;

    for (auto& desc : getTestRegistry())
    {
        Test test;

        test.suiteName = desc.path.filename().string();
        test.name = desc.name;
        test.tags = desc.options.tags;
        test.skipMessage = desc.options.skipMessage;
        test.deviceType = Device::Type::Default;
        test.cpuFunc = desc.cpuFunc;
        test.gpuFunc = desc.gpuFunc;

        if (test.cpuFunc)
        {
            tests.push_back(test);
        }
        else if (test.gpuFunc)
        {
#if FALCOR_HAS_D3D12
            if (desc.options.deviceTypes.empty() || desc.options.deviceTypes.count(Device::Type::D3D12))
            {
                test.deviceType = Device::Type::D3D12;
                test.name = fmt::format("{} (D3D12)", desc.name);
                tests.push_back(test);
            }
#endif
#if FALCOR_HAS_VULKAN
            if (desc.options.deviceTypes.empty() || desc.options.deviceTypes.count(Device::Type::Vulkan))
            {
                test.deviceType = Device::Type::Vulkan;
                test.name = fmt::format("{} (Vulkan)", desc.name);
                tests.push_back(test);
            }
#endif
        }
    }

    // Sort by suite name first, followed by test name.
    std::sort(
        tests.begin(),
        tests.end(),
        [](const Test& a, const Test& b)
        {
            if (a.suiteName == b.suiteName)
                return a.name < b.name;
            return a.suiteName < b.suiteName;
        }
    );

    return tests;
}

std::vector<Test> filterTests(
    std::vector<Test> tests,
    std::string testSuiteFilter,
    std::string testCaseFilter,
    std::string tagFilter,
    Device::Type deviceType
)
{
    std::vector<Test> filtered;

    std::regex suiteFilterRegex(testSuiteFilter, std::regex::icase | std::regex::basic);
    std::regex testFilterRegex(testCaseFilter, std::regex::icase | std::regex::basic);

    std::set<std::string> includeTags;
    std::set<std::string> excludeTags;
    for (const auto& token : splitString(tagFilter, ","))
    {
        if (token.empty())
            continue;
        if (token[0] == '-' || token[0] == '!' || token[0] == '~')
            excludeTags.insert(token.substr(1));
        else if (token[0] == '+')
            includeTags.insert(token.substr(1));
        else
            includeTags.insert(token);
    }

    auto matchTags =
        [](const std::set<std::string>& tags, const std::set<std::string>& includeTags, const std::set<std::string>& excludeTags)
    {
        bool include = includeTags.empty();
        bool exclude = false;

        for (const auto& tag : tags)
        {
            include |= includeTags.count(tag) == 1;
            exclude |= excludeTags.count(tag) == 1;
        }

        return include && !exclude;
    };

    for (auto&& test : tests)
    {
        if (!testSuiteFilter.empty() && !std::regex_search(test.suiteName, suiteFilterRegex))
            continue;
        if (!testCaseFilter.empty() && !std::regex_search(test.name, testFilterRegex))
            continue;
        if (!matchTags(test.tags, includeTags, excludeTags))
            continue;
        if (deviceType != Device::Type::Default && test.deviceType != deviceType)
            continue;
        filtered.push_back(test);
    }

    return filtered;
}

///////////////////////////////////////////////////////////////////////////

void UnitTestContext::reportFailure(const std::string& message)
{
    if (message.empty())
        return;
    reportLine("{}", message);
    mFailureMessages.push_back(message);

    if (isDebuggerPresent())
        debugBreak();
}

///////////////////////////////////////////////////////////////////////////

void GPUUnitTestContext::createProgram(
    const std::filesystem::path& path,
    const std::string& entry,
    const DefineList& programDefines,
    SlangCompilerFlags flags,
    ShaderModel shaderModel,
    bool createShaderVars
)
{
    // Create program.
    mpProgram = Program::createCompute(mpDevice, path, entry, programDefines, flags, shaderModel);
    mpState = ComputeState::create(mpDevice);
    mpState->setProgram(mpProgram);

    // Create vars unless it should be deferred.
    if (createShaderVars)
        createVars();
}

void GPUUnitTestContext::createProgram(const ProgramDesc& desc, const DefineList& programDefines, bool createShaderVars)
{
    // Create program.
    mpProgram = Program::create(mpDevice, desc, programDefines);
    mpState = ComputeState::create(mpDevice);
    mpState->setProgram(mpProgram);

    // Create vars unless it should be deferred.
    if (createShaderVars)
        createVars();
}

void GPUUnitTestContext::createVars()
{
    // Create shader variables.
    ref<const ProgramReflection> pReflection = mpProgram->getReflector();
    mpVars = ProgramVars::create(mpDevice, pReflection);
    FALCOR_ASSERT(mpVars);

    // Try to use shader reflection to query thread group size.
    // ((1,1,1) is assumed if it's not specified.)
    mThreadGroupSize = pReflection->getThreadGroupSize();
    FALCOR_ASSERT(mThreadGroupSize.x >= 1 && mThreadGroupSize.y >= 1 && mThreadGroupSize.z >= 1);
}

void GPUUnitTestContext::allocateStructuredBuffer(const std::string& name, uint32_t nElements, const void* pInitData, size_t initDataSize)
{
    FALCOR_CHECK(mpVars != nullptr, "Program vars not created");
    mStructuredBuffers[name] = mpDevice->createStructuredBuffer(mpVars->getRootVar()[name], nElements);
    if (pInitData)
    {
        ref<Buffer> buffer = mStructuredBuffers[name];
        size_t expectedDataSize = buffer->getStructSize() * buffer->getElementCount();
        if (initDataSize == 0)
            initDataSize = expectedDataSize;
        else if (initDataSize != expectedDataSize)
            throw ErrorRunningTestException("StructuredBuffer '" + name + "' initial data size mismatch");
        buffer->setBlob(pInitData, 0, initDataSize);
    }
}

void GPUUnitTestContext::runProgram(const uint3& dimensions)
{
    FALCOR_CHECK(mpVars != nullptr, "Program vars not created");
    for (const auto& buffer : mStructuredBuffers)
    {
        mpVars->setBuffer(buffer.first, buffer.second);
    }

    uint3 groups = div_round_up(dimensions, mThreadGroupSize);

    // // Check dispatch dimensions.
    if (any(groups > mpDevice->getLimits().maxComputeDispatchThreadGroups))
    {
        throw ErrorRunningTestException("GPUUnitTestContext::runProgram() - Dispatch dimension exceeds maximum.");
    }

    mpDevice->getRenderContext()->dispatch(mpState.get(), mpVars.get(), groups);
}

} // namespace unittest

/**
 * Simple tests of the testing framework. How meta.
 */
CPU_TEST(TestCPUTest)
{
    EXPECT_EQ(1, 1);
    EXPECT(1 == 1);
    EXPECT_NE(1, 2);
    EXPECT_LT(1, 2);
    EXPECT_GT(2, 1);
    EXPECT_LE(2, 2);
    EXPECT_GE(3, 2);

    // Commented out to not have the debugger break if break-on-exception is enabled.
    // EXPECT_THROW({ throw std::runtime_error("Test"); });
    // EXPECT_THROW_AS({ throw std::runtime_error("Test"); }, std::runtime_error);
}

CPU_TEST(TestSingleEval)
{
    // Make sure that arguments to test macros are only evaluated once.
    int i = 0;
    EXPECT_EQ(++i, 1);
    EXPECT_EQ(i, 1);
    EXPECT_NE(++i, 3);
    EXPECT_EQ(i, 2);
    EXPECT_LT(++i, 4);
    EXPECT_EQ(i, 3);
    EXPECT_LE(++i, 4);
    EXPECT_EQ(i, 4);
    EXPECT_GT(++i, 4);
    EXPECT_EQ(i, 5);
    EXPECT_GE(++i, 6);
    EXPECT_EQ(i, 6);
    EXPECT(++i == 7);
    EXPECT_EQ(i, 7);
}

GPU_TEST(TestGPUTest)
{
    ctx.createProgram("Testing/UnitTest.cs.slang");
    ctx.allocateStructuredBuffer("result", 10);
    ctx["TestCB"]["nValues"] = 10;
    ctx["TestCB"]["scale"] = 2.f;
    ctx.runProgram();

    std::vector<float> s = ctx.readBuffer<float>("result");
    // s[i] == 2*i
    EXPECT(s[1] == 2);
    EXPECT_EQ(s[1], 2);
    EXPECT_NE(s[2], 3);
    EXPECT_LT(s[3], 10);
    EXPECT_LE(s[4], 8);
    EXPECT_GT(s[5], 5);
    EXPECT_GE(s[6], 11);
}

CPU_TEST(TestSkip1, "skipped")
{
    EXPECT(false);
}

CPU_TEST(TestSkip2, SKIP("skipped"))
{
    EXPECT(false);
}

CPU_TEST(TestTags, TAGS("tag1", "tag2", "tag3"))
{
    EXPECT(true);
}

} // namespace Falcor
