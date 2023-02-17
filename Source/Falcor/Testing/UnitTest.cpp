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
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/StringUtils.h"
#include "Utils/TermColor.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Vector.h"
#include <fmt/format.h>
#include <fmt/color.h>
#include <algorithm>
#include <chrono>
#include <regex>
#include <inttypes.h>

#include <pugixml.hpp>

namespace Falcor
{
namespace
{
struct Test
{
    std::string getTitle() const
    {
        std::string tag;
        if (cpuFunc)
        {
            tag = "CPU";
        }
        else
        {
            tag = "GPU";
            switch (deviceType)
            {
            case Device::Type::D3D12:
                tag += " D3D12";
                break;
            case Device::Type::Vulkan:
                tag += " Vulkan";
                break;
            }
        }

        return fmt::format("{}/{} ({})", path.filename(), name, tag);
    }

    std::filesystem::path path;
    std::string name;
    std::string skipMessage;
    CPUTestFunc cpuFunc;
    GPUTestFunc gpuFunc;
    UnitTestDeviceFlags supportedDevices;
    Device::Type deviceType;
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

static std::vector<Test>& getTestRegistry()
{
    static std::vector<Test> registry;
    return registry;
}

} // end anonymous namespace

const char* plural(size_t count, const char* suffix)
{
    if (count == 1)
        return "";
    return suffix;
}

void registerCPUTest(const std::filesystem::path& path, const std::string& name, const std::string& skipMessage, CPUTestFunc func)
{
    Test test;
    test.path = path;
    test.name = name;
    test.skipMessage = skipMessage;
    test.cpuFunc = std::move(func);
    getTestRegistry().push_back(test);
}

void registerGPUTest(
    const std::filesystem::path& path,
    const std::string& name,
    const std::string& skipMessage,
    GPUTestFunc func,
    UnitTestDeviceFlags supportedDevices
)
{
    Test test;
    test.path = path;
    test.name = name;
    test.skipMessage = skipMessage;
    test.gpuFunc = std::move(func);
    test.supportedDevices = supportedDevices;
    getTestRegistry().push_back(test);
}

/// Prints the UnitTest report line, making sure it is always printed to the console once.
template<typename... Args>
void reportLine(const std::string_view format, Args&&... args)
{
    std::string report = fmt::vformat(format, fmt::make_format_args(std::forward<Args>(args)...));
    bool willLogPrint = is_set(Logger::getOutputs(), Logger::OutputFlags::Console) ||
                        (is_set(Logger::getOutputs(), Logger::OutputFlags::DebugWindow) && isDebuggerPresent());

    if (!willLogPrint)
        std::cout << report << std::endl;
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

    pugi::xml_node testsuiteNode = testsuitesNode.append_child("testsuite");
    testsuiteNode.append_attribute("name").set_value("Unit Tests");

    for (const auto& [test, result] : report)
    {
        pugi::xml_node testcaseNode = testsuiteNode.append_child("testcase");
        testcaseNode.append_attribute("name").set_value(test.getTitle().c_str());
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

    doc.save_file(path.native().c_str());
}

inline TestResult runTest(const Test& test, std::shared_ptr<Device> pDevice, Fbo* pTargetFbo)
{
    if (!test.skipMessage.empty())
        return {TestResult::Status::Skipped, {test.skipMessage}};

    if (test.gpuFunc)
    {
        if (pDevice->getType() == Device::Type::D3D12 && !is_set(test.supportedDevices, UnitTestDeviceFlags::D3D12))
            return {TestResult::Status::Skipped, {"Not supported on D3D12."}};
        if (pDevice->getType() == Device::Type::Vulkan && !is_set(test.supportedDevices, UnitTestDeviceFlags::Vulkan))
            return {TestResult::Status::Skipped, {"Not supported on Vulkan."}};
    }

    TestResult result{TestResult::Status::Passed};

    auto startTime = std::chrono::steady_clock::now();

    CPUUnitTestContext cpuCtx;
    GPUUnitTestContext gpuCtx(pDevice, pTargetFbo);

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
    if (test.gpuFunc)
        pDevice->flushAndSync();

    return result;
}

int32_t runTests(
    std::shared_ptr<Device> pDevice,
    Fbo* pTargetFbo,
    UnitTestCategoryFlags categoryFlags,
    const std::string& testFilter,
    const std::filesystem::path& xmlReportPath,
    uint32_t repeatCount
)
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

    size_t totalTestCount = 0;
    std::map<std::string, std::vector<Test>> tests;
    std::vector<std::pair<Test, TestResult>> report;
    std::map<std::string, std::vector<Test>> failedTests;

    // Filter tests.
    std::regex testFilterRegex(testFilter, std::regex::icase | std::regex::basic);
    for (auto& it : getTestRegistry())
    {
        if (it.cpuFunc && !is_set(categoryFlags, UnitTestCategoryFlags::CPU))
            continue;
        if (it.gpuFunc && !is_set(categoryFlags, UnitTestCategoryFlags::GPU))
            continue;

        it.deviceType = pDevice->getType();

        if (std::regex_search(it.getTitle(), testFilterRegex))
        {
            totalTestCount++;
            tests[it.path.filename().string()].push_back(it);
        }
    }

    // Sort tests by name.
    for (auto& it : tests)
    {
        std::sort(it.second.begin(), it.second.end(), [](const Test& a, const Test& b) { return a.name < b.name; });
    }

    int32_t failureCount = 0;
    uint64_t totalMS = 0;
    reportLine(
        "[==========] Running {} test{} from {} test suite{}.", totalTestCount, plural(totalTestCount, "s"), tests.size(),
        plural(tests.size(), "s")
    );
    for (auto suiteIt = tests.begin(); suiteIt != tests.end(); ++suiteIt)
    {
        if (abort)
            break;

        const std::string& suiteName = suiteIt->first;
        reportLine("[----------] {} test{} from {}", suiteIt->second.size(), plural(suiteIt->second.size(), "s"), suiteName);
        uint64_t suiteMS = 0;
        for (auto& test : suiteIt->second)
        {
            if (abort)
                break;

            bool success = true;
            for (uint32_t repeatIndex = 0; repeatIndex < repeatCount; ++repeatIndex)
            {
                if (abort)
                    break;

                std::string repeats;
                if (repeatCount > 1)
                    repeats = fmt::format("[{}/{}]", repeatIndex + 1, repeatCount);
                reportLine("[ RUN      ] {}:{}{}", suiteName, test.name, repeats);
                TestResult result = runTest(test, pDevice, pTargetFbo);
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
        reportLine(
            "[----------] {} test{} from {} ({} ms total)", suiteIt->second.size(), plural(suiteIt->second.size(), "s"), suiteIt->first,
            suiteMS
        );
        reportLine("");
        totalMS += suiteMS;
    }

    if (abort)
    {
        reportLine("[ ABORTED  ]");
        return 1;
    }

    if (!xmlReportPath.empty())
        writeXmlReport(xmlReportPath, report);

    reportLine(
        "[==========] {} test{} from {} test suite{} ran. ({} ms total)", totalTestCount, plural(totalTestCount, "s"), tests.size(),
        plural(tests.size(), "s"), totalMS
    );
    reportLine("[  PASSED  ] {} test{}.", totalTestCount - failureCount, plural(totalTestCount - failureCount, "s"));
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

///////////////////////////////////////////////////////////////////////////

void UnitTestContext::reportFailure(const std::string& message)
{
    if (message.empty())
        return;
    reportLine("{}", message);
    mFailureMessages.push_back(message);
}

///////////////////////////////////////////////////////////////////////////

void GPUUnitTestContext::createProgram(
    const std::filesystem::path& path,
    const std::string& entry,
    const Program::DefineList& programDefines,
    Shader::CompilerFlags flags,
    const std::string& shaderModel,
    bool createShaderVars
)
{
    // Create program.
    mpProgram = ComputeProgram::createFromFile(mpDevice, path, entry, programDefines, flags, shaderModel);
    mpState = ComputeState::create(mpDevice);
    mpState->setProgram(mpProgram);

    // Create vars unless it should be deferred.
    if (createShaderVars)
        createVars();
}

void GPUUnitTestContext::createProgram(const Program::Desc& desc, const Program::DefineList& programDefines, bool createShaderVars)
{
    // Create program.
    mpProgram = ComputeProgram::create(mpDevice, desc, programDefines);
    mpState = ComputeState::create(mpDevice);
    mpState->setProgram(mpProgram);

    // Create vars unless it should be deferred.
    if (createShaderVars)
        createVars();
}

void GPUUnitTestContext::createVars()
{
    // Create shader variables.
    ProgramReflection::SharedConstPtr pReflection = mpProgram->getReflector();
    mpVars = ComputeVars::create(mpDevice, pReflection);
    FALCOR_ASSERT(mpVars);

    // Try to use shader reflection to query thread group size.
    // ((1,1,1) is assumed if it's not specified.)
    mThreadGroupSize = pReflection->getThreadGroupSize();
    FALCOR_ASSERT(mThreadGroupSize.x >= 1 && mThreadGroupSize.y >= 1 && mThreadGroupSize.z >= 1);
}

void GPUUnitTestContext::allocateStructuredBuffer(const std::string& name, uint32_t nElements, const void* pInitData, size_t initDataSize)
{
    checkInvariant(mpVars != nullptr, "Program vars not created");
    mStructuredBuffers[name].pBuffer = Buffer::createStructured(mpDevice.get(), mpProgram.get(), name, nElements);
    FALCOR_ASSERT(mStructuredBuffers[name].pBuffer);
    if (pInitData)
    {
        size_t expectedDataSize = mStructuredBuffers[name].pBuffer->getStructSize() * mStructuredBuffers[name].pBuffer->getElementCount();
        if (initDataSize == 0)
            initDataSize = expectedDataSize;
        else if (initDataSize != expectedDataSize)
            throw ErrorRunningTestException("StructuredBuffer '" + name + "' initial data size mismatch");
        mStructuredBuffers[name].pBuffer->setBlob(pInitData, 0, initDataSize);
    }
}

void GPUUnitTestContext::runProgram(const uint3& dimensions)
{
    checkInvariant(mpVars != nullptr, "Program vars not created");
    for (const auto& buffer : mStructuredBuffers)
    {
        mpVars->setBuffer(buffer.first, buffer.second.pBuffer);
    }

    uint3 groups = div_round_up(dimensions, mThreadGroupSize);

    // // Check dispatch dimensions.
    if (glm::any(glm::greaterThan(groups, mpDevice->getLimits().maxComputeDispatchThreadGroups)))
    {
        throw ErrorRunningTestException("GPUUnitTestContext::runProgram() - Dispatch dimension exceeds maximum.");
    }

    mpDevice->getRenderContext()->dispatch(mpState.get(), mpVars.get(), groups);
}

void GPUUnitTestContext::unmapBuffer(const char* bufferName)
{
    FALCOR_ASSERT(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
    if (!mStructuredBuffers[bufferName].mapped)
        throw ErrorRunningTestException(std::string(bufferName) + ": buffer not mapped");
    mStructuredBuffers[bufferName].pBuffer->unmap();
    mStructuredBuffers[bufferName].mapped = false;
}

const void* GPUUnitTestContext::mapRawRead(const char* bufferName)
{
    FALCOR_ASSERT(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
    if (mStructuredBuffers.find(bufferName) == mStructuredBuffers.end())
    {
        throw ErrorRunningTestException(std::string(bufferName) + ": couldn't find buffer to map");
    }
    if (mStructuredBuffers[bufferName].mapped)
        throw ErrorRunningTestException(std::string(bufferName) + ": buffer already mapped");
    mStructuredBuffers[bufferName].mapped = true;
    return mStructuredBuffers[bufferName].pBuffer->map(Buffer::MapType::Read);
}

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

    const float* s = ctx.mapBuffer<const float>("result");
    // s[i] == 2*i
    EXPECT(s[1] == 2);
    EXPECT_EQ(s[1], 2);
    EXPECT_NE(s[2], 3);
    EXPECT_LT(s[3], 10);
    EXPECT_LE(s[4], 8);
    EXPECT_GT(s[5], 5);
    EXPECT_GE(s[6], 11);

    ctx.unmapBuffer("result");
}
} // namespace Falcor
