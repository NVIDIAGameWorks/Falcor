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
#include "UnitTest.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/StringUtils.h"
#include "Utils/TermColor.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
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
                return path.filename().string() + "/" + name + " (" + (cpuFunc ? "CPU" : "GPU") + ")";
            }

            std::filesystem::path path;
            std::string name;
            std::string skipMessage;
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
            uint64_t elapsedMS = 0;
        };

        /** testRegistry is declared as pointer so that we can ensure it can be explicitly
             allocated when register[CG]PUTest() is called.  (The C++ static object
             initialization fiasco.)
         */
        std::vector<Test>* testRegistry;

    }   // end anonymous namespace

    void registerCPUTest(const std::filesystem::path& path, const std::string& name,
                         const std::string& skipMessage, CPUTestFunc func)
    {
        if (!testRegistry) testRegistry = new std::vector<Test>;
        testRegistry->push_back({ path, name, skipMessage, std::move(func), {} });
    }

    void registerGPUTest(const std::filesystem::path& path, const std::string& name,
                         const std::string& skipMessage, GPUTestFunc func)
    {
        if (!testRegistry) testRegistry = new std::vector<Test>;
        testRegistry->push_back({ path, name, skipMessage, {}, std::move(func) });
    }

    /** Write a test report in JUnit's XML format.
        \param[in] path File path.
        \param[in] report List of tests/results.
    */
    inline void writeXmlReport(const std::filesystem::path& path, const std::vector<std::pair<Test, TestResult>>& report)
    {
        pugi::xml_document doc;

        pugi::xml_node testsuitesNode = doc.append_child("testsuites");

        pugi::xml_node testsuiteNode = testsuitesNode.append_child("testsuite");
        testsuiteNode.append_attribute("name").set_value("Falcor");

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

    inline TestResult runTest(const Test& test, RenderContext* pRenderContext)
    {
        if (!test.skipMessage.empty()) return { TestResult::Status::Skipped, { test.skipMessage } };

        TestResult result { TestResult::Status::Passed };

        auto startTime = std::chrono::steady_clock::now();

        CPUUnitTestContext cpuCtx;
        GPUUnitTestContext gpuCtx(pRenderContext);

        std::string extraMessage;

        try
        {
            if (test.cpuFunc) test.cpuFunc(cpuCtx);
            else test.gpuFunc(gpuCtx);
        }
        catch (const SkippingTestException& e)
        {
            result.status = TestResult::Status::Skipped;
            extraMessage = e.what();
        }
        catch (const ErrorRunningTestException& e)
        {
            result.status = TestResult::Status::Failed;
            extraMessage = e.what();
        }
        catch (const TooManyFailedTestsException&)
        {
            result.status = TestResult::Status::Failed;
            extraMessage = "Gave up after " + std::to_string(kMaxTestFailures) + " failures.";
        }
        catch (const std::exception& e)
        {
            result.status = TestResult::Status::Failed;
            extraMessage = e.what();
        }

        result.messages = test.cpuFunc ? cpuCtx.getFailureMessages() : gpuCtx.getFailureMessages();

        if (!result.messages.empty()) result.status = TestResult::Status::Failed;

        if (!extraMessage.empty()) result.messages.push_back(extraMessage);

        auto endTime = std::chrono::steady_clock::now();
        result.elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        // Release GPU resources.
        if (test.gpuFunc) gpDevice->flushAndSync();

        return result;
    }

    int32_t runTests(std::ostream& stream, RenderContext* pRenderContext, const std::string &testFilter, const std::filesystem::path& xmlReportPath, uint32_t repeatCount)
    {
        if (testRegistry == nullptr) return 0;

        std::vector<Test> tests;
        std::vector<std::pair<Test, TestResult>> report;

        // Filter tests.
        std::regex testFilterRegex(testFilter, std::regex::icase | std::regex::basic);
        std::copy_if(testRegistry->begin(), testRegistry->end(), std::back_inserter(tests),
            [&testFilterRegex] (const Test& test)
        {
            return std::regex_search(test.getTitle(), testFilterRegex);
        });

        // Sort tests by name.
        std::sort(tests.begin(), tests.end(),
            [](const Test &a, const Test &b)
        {
            return (a.path / a.name).string() < (b.path / b.name).string();
        });

        stream << fmt::format("Running {} tests:\n", tests.size());
        logInfo("Running {} tests.", tests.size());

        int32_t failureCount = 0;

        for (const auto& test : tests)
        {
            for (uint32_t repeatIndex = 0; repeatIndex < repeatCount; ++repeatIndex)
            {
                stream << fmt::format("  {:80}: ", test.getTitle());
                if (repeatCount > 1) stream << fmt::format("[{}/{}] ", repeatIndex + 1, repeatCount);
                stream << std::flush;
                logInfo("Running test '{}'.", test.getTitle());

                TestResult result = runTest(test, pRenderContext);
                report.emplace_back(test, result);

                std::string statusTag;
                TermColor statusColor;

                switch (result.status)
                {
                case TestResult::Status::Passed:
                    statusTag = "PASSED";
                    statusColor = TermColor::Green;
                    break;
                case TestResult::Status::Failed:
                    statusTag = "FAILED";
                    statusColor = TermColor::Red;
                    break;
                case TestResult::Status::Skipped:
                    statusTag = "SKIPPED";
                    statusColor = TermColor::Yellow;
                    break;
                }

                stream << colored(statusTag, statusColor, stream);
                stream << fmt::format(" ({} ms)\n", result.elapsedMS);
                for (const auto& m : result.messages) stream << fmt::format("    {}\n", m);

                logInfo("Finished test '{}' in {} ms with status {}.", test.getTitle(), result.elapsedMS, statusTag);
                for (const auto& m : result.messages) logInfo("    {}", m);

                if (result.status == TestResult::Status::Failed) ++failureCount;
            }
        }

        if (!xmlReportPath.empty()) writeXmlReport(xmlReportPath, report);

        return failureCount;
    }

    ///////////////////////////////////////////////////////////////////////////

    void GPUUnitTestContext::createProgram(const std::filesystem::path& path,
                                           const std::string& entry,
                                           const Program::DefineList& programDefines,
                                           Shader::CompilerFlags flags,
                                           const std::string& shaderModel,
                                           bool createShaderVars)
    {
        // Create program.
        mpProgram = ComputeProgram::createFromFile(path, entry, programDefines, flags, shaderModel);
        mpState = ComputeState::create();
        mpState->setProgram(mpProgram);

        // Create vars unless it should be deferred.
        if (createShaderVars) createVars();
    }

    void GPUUnitTestContext::createProgram(const Program::Desc& desc,
                                           const Program::DefineList& programDefines,
                                           bool createShaderVars)
    {
        // Create program.
        mpProgram = ComputeProgram::create(desc, programDefines);
        mpState = ComputeState::create();
        mpState->setProgram(mpProgram);

        // Create vars unless it should be deferred.
        if (createShaderVars) createVars();
    }

    void GPUUnitTestContext::createVars()
    {
        // Create shader variables.
        ProgramReflection::SharedConstPtr pReflection = mpProgram->getReflector();
        mpVars = ComputeVars::create(pReflection);
        FALCOR_ASSERT(mpVars);

        // Try to use shader reflection to query thread group size.
        // ((1,1,1) is assumed if it's not specified.)
        mThreadGroupSize = pReflection->getThreadGroupSize();
        FALCOR_ASSERT(mThreadGroupSize.x >= 1 && mThreadGroupSize.y >= 1 && mThreadGroupSize.z >= 1);
    }

    void GPUUnitTestContext::allocateStructuredBuffer(const std::string& name, uint32_t nElements, const void* pInitData, size_t initDataSize)
    {
        checkInvariant(mpVars != nullptr, "Program vars not created");
        mStructuredBuffers[name].pBuffer = Buffer::createStructured(mpProgram.get(), name, nElements);
        FALCOR_ASSERT(mStructuredBuffers[name].pBuffer);
        if (pInitData)
        {
            size_t expectedDataSize = mStructuredBuffers[name].pBuffer->getStructSize() * mStructuredBuffers[name].pBuffer->getElementCount();
            if (initDataSize == 0) initDataSize = expectedDataSize;
            else if (initDataSize != expectedDataSize) throw ErrorRunningTestException("StructuredBuffer '" + name + "' initial data size mismatch");
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

#ifdef FALCOR_D3D12
        // Check dispatch dimensions. TODO: Should be moved into Falcor.
        if (groups.x > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            groups.y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            groups.z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
        {
            throw ErrorRunningTestException("GPUUnitTestContext::runProgram() - Dispatch dimension exceeds maximum.");
        }
#endif  // FALCOR_D3D12

        mpContext->dispatch(mpState.get(), mpVars.get(), groups);
    }

    void GPUUnitTestContext::unmapBuffer(const char* bufferName)
    {
        FALCOR_ASSERT(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
        if (!mStructuredBuffers[bufferName].mapped) throw ErrorRunningTestException(std::string(bufferName) + ": buffer not mapped");
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
        if (mStructuredBuffers[bufferName].mapped) throw ErrorRunningTestException(std::string(bufferName) + ": buffer already mapped");
        mStructuredBuffers[bufferName].mapped = true;
        return mStructuredBuffers[bufferName].pBuffer->map(Buffer::MapType::Read);
    }

    /** Simple tests of the testing framework. How meta.
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

}  // namespace Falcor
