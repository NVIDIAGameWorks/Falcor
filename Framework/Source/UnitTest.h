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
#pragma once
#include "Falcor.h"

#include <exception>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

/**

This file defines both the user-visible API for the unit testing framework
as well as the various classes that implement it.  For documentation about
how to use it, please see https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/wikis/testing/unit-testing.

 */

namespace Falcor
{
    static constexpr int kMaxTestFailures = 25;

    class CPUUnitTestContext;
    class GPUUnitTestContext;

    struct TooManyFailedTestsException : public std::exception { };

    class ErrorRunningTestException : public std::exception
    {
    public:
        ErrorRunningTestException(const std::string& what) : mWhat(what) { }

        const char* what() const noexcept override { return mWhat.c_str(); }

    private:
        std::string mWhat;
    };

    using CPUTestFunc = std::function<void(CPUUnitTestContext& ctx)>;
    void registerCPUTest(const std::string& filename, const std::string& name,
                         CPUTestFunc func);

    using GPUTestFunc = std::function<void(GPUUnitTestContext& ctx)>;
    void registerGPUTest(const std::string& filename, const std::string& name,
                         GPUTestFunc func);

    int32_t runTests(FILE *file, RenderContext* pRenderContext, const std::string& testFilterRegexp);

    class UnitTestContext
    {
    public:
        /** reportFailure is called with an error message to report a failing
            test.  Normally it's only used by the EXPECT_EQ (etc.) macros,
            though it's fine for a user to call it with different failures.
        */
        void reportFailure(const std::string& message)
        {
            if (message.empty()) return;
            mFailureMessage += std::string("    ") + message + '\n';
        }

        std::string getFailureMessage() const { return mFailureMessage; }

        int mNumFailures = 0;

    private:
        std::string mFailureMessage;
    };

    class CPUUnitTestContext : public UnitTestContext
    {
    };

    class GPUUnitTestContext : public UnitTestContext
    {
    public:
        GPUUnitTestContext(RenderContext* pContext) : mpContext(pContext) { }

        /** createProgram creates a compute program from the source code at the
            given path.  The entrypoint is assumed to be |main()| unless
            otherwise specified with the |csEntry| parameter.  Preprocessor
            defines and compiler flags can also be optionally provided.
         */
        void createProgram(const std::string& path,
                           const std::string& csEntry = "main",
                           const Program::DefineList& programDefines = Program::DefineList(),
                           Shader::CompilerFlags flags = Shader::CompilerFlags::None,
                           const std::string& shaderModel = "");

        /** vars returns the ComputeVars for the program for use in binding
            textures, etc.
         */
        ComputeVars& vars() { return *mpVars; }

        /** operator[] returns the |ConstantBuffer| with the given name (or
            nullptr if no such constant buffer exists).
         */
        ConstantBuffer::SharedPtr operator[](const std::string& cbName)
        {
            return mpVars->getDefaultBlock()->getConstantBuffer(cbName);
        }

        /** allocateStructuredBuffer is a helper method that allocates a
            structured buffer of the given name with the given number of
            elements.  Note: the given structured buffer must be declared at
            global scope.

            TODO: support structured buffers in parameter blocks?
            TODO: add support for other buffer allocation types?
        */
        void allocateStructuredBuffer(const std::string& name, size_t nElements);

        /* runProgram runs the compute program that was specified in
           |createProgram|, where the total number of threads that runs is
           given by the product of the three provided dimensions.
         */
        void runProgram(int32_t width = 1, int32_t height = 1, int32_t depth = 1);

        /** mapBuffer returns a pointer to the named structured buffer.
            Returns nullptr if no such buffer exists.  SFINAE is used to
            require that a the requested pointer is const.
        */
        template <typename T> T* mapBuffer(const char* bufferName,
            typename std::enable_if<std::is_const<T>::value>::type* = 0)
        {
            return reinterpret_cast<T*>(mapRawRead(bufferName));
        }

        /** unmapBuffer unmaps a buffer after it's been used after a call to
            |mapBuffer()|.
        */
        void unmapBuffer(const char* bufferName);

    private:
        RenderContext * mpContext;
        ComputeState::SharedPtr mpState;
        ComputeProgram::SharedPtr mpProgram;
        ComputeVars::SharedPtr mpVars;
        glm::uvec3 mThreadGroupSize = { 0, 0, 0 };
        struct ParameterBuffer
        {
            StructuredBuffer::SharedPtr pBuffer;
            bool mapped = false;
        };
        std::map<std::string, ParameterBuffer> mStructuredBuffers;

        const void* mapRawRead(const char* bufferName);
    };

    /** StreamSink is a utility class used by the testing framework that either
        captures values printed via C++'s operator<< (as with regular
        std::ostreams) or discards them.  (If a test has failed, then
        StreamSink does the former, and if it has passed, it does the latter.)
        In the event of a test failure, passes along the failure message to the
        provided GPUUnitTestContext's |reportFailure| method.
     */
    class StreamSink
    {
    public:
        /* If a non-nullptr UnitTestContext is provided, the values printed
           will be accumulated and passed to the context's reportFailure()
           method when the StreamSink destructor runs. */
        StreamSink(UnitTestContext* ctx) : mpCtx(ctx) {}

        ~StreamSink()
        {
            if (mpCtx) mpCtx->reportFailure(mSs.str());
        }

        template <typename T>
        StreamSink& operator<<(T&&s)
        {
            if (mpCtx) mSs << std::move(s);
            return *this;
        }

    private:
        std::stringstream mSs;
        UnitTestContext* mpCtx;
    };

    ///////////////////////////////////////////////////////////////////////////

    /** Start of user-facing API */

#define CPU_TEST(Name) \
    static void CPUUnitTest##Name(CPUUnitTestContext& ctx);           \
    struct CPUUnitTestRegisterer##Name {                              \
        CPUUnitTestRegisterer##Name()                                 \
        {                                                             \
            registerCPUTest(__FILE__, #Name, CPUUnitTest##Name);      \
        }                                                             \
    } RegisterCPUTest##Name;                                          \
    static void CPUUnitTest##Name(CPUUnitTestContext& ctx) /* over to the user for the braces */

/** Macro to define a GPU unit test.  It defines an instance of the
    |GPUUnitTestRegisterer| class, which in turn registers the test with
    the test framework when its constructor executes at program startup
    time.  Next, it starts the definition of the testing function, up to
    the point at which the user should supply an open brace and start
    writing code.
*/
#define GPU_TEST(Name) \
    static void GPUUnitTest##Name(GPUUnitTestContext& ctx);           \
    struct GPUUnitTestRegisterer##Name {                              \
        GPUUnitTestRegisterer##Name()                                 \
        {                                                             \
            registerGPUTest(__FILE__, #Name, GPUUnitTest##Name);      \
        }                                                             \
    } RegisterGPUTest##Name;                                          \
    static void GPUUnitTest##Name(GPUUnitTestContext& ctx) /* over to the user for the braces */

/** Macro definitions for the GPU unit testing framework. Note that they
    are all a single statement (including any additional << printed
    values).  Thus, it's perfectly fine to write code like:

    if (foo)  // look, no braces
        EXPECT_EQ(x, y);

    It is, however, a little tricky to make that work: we'd like to all
    EXPECT*s be single statements, but that's tricky since we'd also like
    to throw an exception and abort the test if too many EXPECT*s fail.
    One might think to do that from the StreamSink destructor, since that's
    a natural place to check how many failures we've seen, but... it's
    illegal to throw exceptions from destructors.

    Therefore, we take advantage of the comma operator for the throw in
    that case.  In the event of a failed test, we see if there have been
    too many errors and throw if appropriate.  But... there's one more
    thing: we can't directly use it in C++'s ternary operator since not
    only is throw a statement, but the types of the true and false sides
    have to be the same.  However, function call is an expression, and
    it's legit to have two void function calls in a ternary expression.

    Thus, we have two helper functions that throw or not, call the
    appropriate one based on the EXPECT* test and the number of failed
    tests, then use the comma operator to move on to the declaration of a
    StreamSink variable to take care of the test message output.

    For maximum C++ grossness, I suppose these could have been lambdas.

*/
    inline void throwTooManyFailures() { throw TooManyFailedTestsException(); }
    inline void dontThrow() { }

    /** Note: the tests are written as they are so that NaNs are handled correctly.
     */
#define EXPECT_EQ(x, y)                                                 \
  (((!((x) == (y)) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!((x) == (y)) ? &ctx : nullptr)) <<                       \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " == " #y " (" <<   \
    x << " vs. " << y << ") "

#define EXPECT_GE(x, y)                                                 \
  (((!((x) >= (y)) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!((x) >= (y)) ? &ctx : nullptr)) <<                       \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " >= " #y " (" <<   \
    x << " vs. " << y << ") "

#define EXPECT_GT(x, y)                                                 \
  (((!((x) > (y)) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!((x) > (y)) ? &ctx : nullptr)) <<                        \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " > " #y " (" <<    \
    x << " vs. " << y << ") "

#define EXPECT_LE(x, y)                                                 \
  (((!((x) <= (y)) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!((x) <= (y)) ? &ctx : nullptr)) <<                       \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " <= " #y " (" <<   \
    x << " vs. " << y << ") "

#define EXPECT_LT(x, y)                                                 \
  (((!((x) < (y)) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!((x) < (y)) ? &ctx : nullptr)) <<                        \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " < " #y " (" <<    \
    x << " vs. " << y << ") "

#define EXPECT_NE(x, y)                                                 \
  (((!((x) != (y)) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!((x) != (y)) ? &ctx : nullptr)) <<                       \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " != " #y " (" <<   \
    x << " vs. " << y << ") "

#define EXPECT(x)                                                       \
  (((!(x) && ++ctx.mNumFailures == kMaxTestFailures) ? throwTooManyFailures() : dontThrow()), \
   StreamSink(!(x) ? &ctx : nullptr)) <<                                \
    __FILE__ ":" << __LINE__ << " Test failed: " #x " "

} // namespace Falcor
