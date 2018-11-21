/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "UnitTest.h"

#include "Utils/Platform/OS.h"

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

            std::string filename, name;
            CPUTestFunc cpuFunc;
            GPUTestFunc gpuFunc;
        };

        /** tests is declared as pointer so that we can ensure it can be explicitly
             allocated when register[CG]PUTest() is called.  (The C++ static object
             initialization fiasco.)
         */
        std::vector<Test>* tests;

    }   // end anonymous namespace

    void registerCPUTest(const std::string& filename, const std::string& name,
                         CPUTestFunc func)
    {
        if (!tests) tests = new std::vector<Test>;
        tests->push_back({ filename, name, std::move(func), {} });
    }

    void registerGPUTest(const std::string& filename, const std::string& name,
                         GPUTestFunc func)
    {
        if (!tests) tests = new std::vector<Test>;
        tests->push_back({ filename, name, {}, std::move(func) });
    }

    int32_t runTests(FILE *file, RenderContext *pRenderContext, const std::string &testFilter)
    {
        int32_t nFailures = 0;

        if (tests == nullptr) return 0;

        std::regex testFilterRegex(testFilter, std::regex::icase | std::regex::basic);
        size_t nTests = std::count_if(tests->begin(), tests->end(),
            [&testFilterRegex](const Test &test)
        {
            return std::regex_search(test.getTitle(), testFilterRegex);
        });

        fprintf(file, "Running %d tests\n", int32_t(nTests));

        std::sort(tests->begin(), tests->end(),
            [](const Test &a, const Test &b)
        {
            return (a.filename + "/" + a.name) < (b.filename + "/" + b.name);
        });

        for (const auto& t : *tests)
        {
            if (!testFilter.empty() && !std::regex_search(t.getTitle(), testFilterRegex)) continue;

            auto startTime = std::chrono::steady_clock::now();
            CPUUnitTestContext cpuCtx;
            GPUUnitTestContext gpuCtx(pRenderContext);

            std::string status, failureDetails;
            try
            {
                if (t.cpuFunc) t.cpuFunc(cpuCtx);
                else t.gpuFunc(gpuCtx);
            }
            catch (ErrorRunningTestException e)
            {
                status = "SKIPPED";
                failureDetails = "    ";
                failureDetails += e.what();
                failureDetails += "\n";
            }
            catch (TooManyFailedTestsException e)
            {
                status = "ABORTED";
                failureDetails = "    Gave up after " + std::to_string(kMaxTestFailures) + " failures.\n";
            }

            std::string failureMessage;
            if (t.cpuFunc) failureMessage = cpuCtx.getFailureMessage();
            else failureMessage = gpuCtx.getFailureMessage();

            if (status.empty()) status = failureMessage.empty() ? "PASSED" : "FAILED";

            auto endTime = std::chrono::steady_clock::now();
            int64_t elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

            fprintf(file, "  %-60s: %s (%" PRId64 " ms)\n", t.getTitle().c_str(), status.c_str(), elapsedMS);
            if (!failureMessage.empty())
            {
                ++nFailures;
                fprintf(file, "%s", failureMessage.c_str());
                if (!failureDetails.empty()) fprintf(file, "%s", failureDetails.c_str());
            }
        }

        return nFailures;
    }

    ///////////////////////////////////////////////////////////////////////////

    void GPUUnitTestContext::createProgram(const std::string& path,
                                           const std::string& entry,
                                           const Program::DefineList& programDefines,
                                           Shader::CompilerFlags flags,
                                           const std::string& shaderModel)
    {
        mpProgram = ComputeProgram::createFromFile(path, entry, programDefines, flags, shaderModel);

        mpState = ComputeState::create();
        mpState->setProgram(mpProgram);

        // Create shader variables.
        ProgramReflection::SharedConstPtr pReflection = mpProgram->getReflector();
        mpVars = ComputeVars::create(pReflection);

        if (!mpVars) throw ErrorRunningTestException("couldn't create vars");

        // Try to use shader reflection to query thread group size.  ((1,1,1)
        // is assumed if it's not specified.)
        mThreadGroupSize = pReflection->getThreadGroupSize();
        assert(mThreadGroupSize.x >= 1 && mThreadGroupSize.y >= 1 && mThreadGroupSize.z >= 1);
    }

    void GPUUnitTestContext::allocateStructuredBuffer(const std::string& name, size_t nElements)
    {
        mStructuredBuffers[name].pBuffer = StructuredBuffer::create(mpProgram, name, nElements);
        if (!mStructuredBuffers[name].pBuffer) throw ErrorRunningTestException(name + ": couldn't create structured buffer");
    }

    template <typename T> static T div_round_up(T a, T b) { return (a + b - (T)1) / b; }

    void GPUUnitTestContext::runProgram(int32_t width, int32_t height, int32_t depth)
    {
        for (const auto& buffer : mStructuredBuffers)
        {
            mpVars->setStructuredBuffer(buffer.first, buffer.second.pBuffer);
        }

        mpContext->pushComputeState(mpState);
        mpContext->pushComputeVars(mpVars);

        uvec3 groups = div_round_up(glm::uvec3(width, height, depth), mThreadGroupSize);

#ifdef FALCOR_D3D12
        // Check dispatch dimensions. TODO: Should be moved into Falcor.
        if (groups.x > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            groups.y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            groups.z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
        {
            throw ErrorRunningTestException("GPUUnitTestContext::runProgram() - Dispatch dimension exceeds maximum.");
        }
#endif  // FALCOR_D3D12

        mpContext->dispatch(groups.x, groups.y, groups.z);

        mpContext->popComputeVars();
        mpContext->popComputeState();
    }

    void GPUUnitTestContext::unmapBuffer(const char* bufferName)
    {
        assert(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
        if (!mStructuredBuffers[bufferName].mapped) throw ErrorRunningTestException(std::string(bufferName) + ": buffer not mapped");
        mStructuredBuffers[bufferName].pBuffer->unmap();
        mStructuredBuffers[bufferName].mapped = false;
    }

    const void *GPUUnitTestContext::mapRawRead(const char* bufferName)
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

    /* Simple tests of the testing framework. How meta. */
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

    GPU_TEST(TestGPUTest)
    {
        ctx.createProgram("UnitTest.cs.hlsl");
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
