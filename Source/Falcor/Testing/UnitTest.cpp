/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "UnitTest.h"
#include <algorithm>
#include <chrono>
#include <regex>
#include <inttypes.h>

namespace Falcor
{
    namespace
    {
        struct Test
        {
            std::string getTitle() const
            {
                return getFilenameFromPath(filename) + "/" + name + " (" + (cpuFunc ? "CPU" : "GPU") + ")";
            }

            std::string filename;
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

    void registerCPUTest(const std::string& filename, const std::string& name,
                         const std::string& skipMessage, CPUTestFunc func)
    {
        if (!testRegistry) testRegistry = new std::vector<Test>;
        testRegistry->push_back({ filename, name, skipMessage, std::move(func), {} });
    }

    void registerGPUTest(const std::string& filename, const std::string& name,
                         const std::string& skipMessage, GPUTestFunc func)
    {
        if (!testRegistry) testRegistry = new std::vector<Test>;
        testRegistry->push_back({ filename, name, skipMessage, {}, std::move(func) });
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

        return result;
    }

    int32_t runTests(std::ostream& stream, RenderContext* pRenderContext, const std::string &testFilter)
    {
        if (testRegistry == nullptr) return 0;

        std::vector<Test> tests;

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
            return (a.filename + "/" + a.name) < (b.filename + "/" + b.name);
        });

        stream << "Running " << std::to_string(tests.size()) << " tests" << std::endl;

        int32_t failureCount = 0;

        for (const auto& test : tests)
        {
            stream << "  " << padStringToLength(test.getTitle(), 60) << ": " << std::flush;

            TestResult result = runTest(test, pRenderContext);

            switch (result.status)
            {
            case TestResult::Status::Passed: stream << colored("PASSED", TermColor::Green, stream); break;
            case TestResult::Status::Failed: stream << colored("FAILED", TermColor::Red, stream); break;
            case TestResult::Status::Skipped: stream << colored("SKIPPED", TermColor::Yellow, stream); break;
            }

            stream << " (" << std::to_string(result.elapsedMS) << " ms)" << std::endl;
            for (const auto& m : result.messages) stream << "    "  << m << std::endl;

            if (result.status == TestResult::Status::Failed) ++failureCount;
        }

        return failureCount;
    }

    ///////////////////////////////////////////////////////////////////////////

    void GPUUnitTestContext::createProgram(const std::string& path,
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

    void GPUUnitTestContext::createVars()
    {
        // Create shader variables.
        ProgramReflection::SharedConstPtr pReflection = mpProgram->getReflector();
        mpVars = ComputeVars::create(pReflection);
        assert(mpVars);

        // Try to use shader reflection to query thread group size.
        // ((1,1,1) is assumed if it's not specified.)
        mThreadGroupSize = pReflection->getThreadGroupSize();
        assert(mThreadGroupSize.x >= 1 && mThreadGroupSize.y >= 1 && mThreadGroupSize.z >= 1);
    }

    void GPUUnitTestContext::allocateStructuredBuffer(const std::string& name, uint32_t nElements, const void* pInitData, size_t initDataSize)
    {
        assert(mpVars);
        mStructuredBuffers[name].pBuffer = Buffer::createStructured(mpProgram.get(), name, nElements);
        assert(mStructuredBuffers[name].pBuffer);
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
        assert(mpVars);
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
        assert(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
        if (!mStructuredBuffers[bufferName].mapped) throw ErrorRunningTestException(std::string(bufferName) + ": buffer not mapped");
        mStructuredBuffers[bufferName].pBuffer->unmap();
        mStructuredBuffers[bufferName].mapped = false;
    }

    const void* GPUUnitTestContext::mapRawRead(const char* bufferName)
    {
        assert(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
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
