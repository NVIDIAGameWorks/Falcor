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
#pragma once
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Math/Vector.h"

#include <glm/gtx/io.hpp>

#include <filesystem>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// @skallweit: This is temporary to allow FalcorTest be compiled unmodified. Needs to be removed.
#include "Core/API/RenderContext.h"

/** This file defines both the user-visible API for the unit testing framework as well as the various classes that implement it
*/

namespace Falcor
{
    class RenderContext;

    static constexpr int kMaxTestFailures = 25;

    class CPUUnitTestContext;
    class GPUUnitTestContext;

    struct TooManyFailedTestsException : public Exception { };

    class ErrorRunningTestException : public Exception
    {
    public:
        ErrorRunningTestException(const std::string& what) : Exception(what) { }
    };

    class SkippingTestException : public Exception
    {
    public:
        SkippingTestException(const std::string& what) : Exception(what) { }
    };

    using CPUTestFunc = std::function<void(CPUUnitTestContext& ctx)>;
    using GPUTestFunc = std::function<void(GPUUnitTestContext& ctx)>;

    FALCOR_API void registerCPUTest(const std::filesystem::path& path, const std::string& name, const std::string& skipMessage, CPUTestFunc func);
    FALCOR_API void registerGPUTest(const std::filesystem::path& path, const std::string& name, const std::string& skipMessage, GPUTestFunc func);
    FALCOR_API int32_t runTests(std::ostream& stream, RenderContext* pRenderContext, const std::string& testFilterRegexp, const std::filesystem::path& xmlReportPath, uint32_t repeatCount = 1);

    class FALCOR_API UnitTestContext
    {
    public:
        /** reportFailure is called with an error message to report a failing
            test.  Normally it's only used by the EXPECT_EQ (etc.) macros,
            though it's fine for a user to call it with different failures.
        */
        void reportFailure(const std::string& message)
        {
            if (message.empty()) return;
            mFailureMessages.push_back(message);
        }

        std::vector<std::string> getFailureMessages() const { return mFailureMessages; }

        int mNumFailures = 0;

    private:
        std::vector<std::string> mFailureMessages;
    };

    class FALCOR_API CPUUnitTestContext : public UnitTestContext
    {
    };

    class FALCOR_API GPUUnitTestContext : public UnitTestContext
    {
    public:
        GPUUnitTestContext(RenderContext* pContext) : mpContext(pContext) { }

        /** createProgram creates a compute program from the source code at the
            given path.  The entrypoint is assumed to be |main()| unless
            otherwise specified with the |csEntry| parameter.  Preprocessor
            defines and compiler flags can also be optionally provided.
        */
        void createProgram(const std::filesystem::path& path,
                           const std::string& csEntry = "main",
                           const Program::DefineList& programDefines = Program::DefineList(),
                           Shader::CompilerFlags flags = Shader::CompilerFlags::None,
                           const std::string& shaderModel = "",
                           bool createShaderVars = true);

        /** Create compute program based on program desc and defines.
        */
        void createProgram(const Program::Desc& desc,
                           const Program::DefineList& programDefines = Program::DefineList(),
                           bool createShaderVars = true);

        /** (Re-)create the shader variables. Call this if vars were not
            created in createProgram() (if createVars = false), or after
            the shader variables have changed through specialization.
        */
        void createVars();

        /** vars returns the ComputeVars for the program for use in binding
            textures, etc.
        */
        ComputeVars& vars()
        {
            FALCOR_ASSERT(mpVars);
            return *mpVars;
        }

        /** Get a shader variable that points at the field with the given `name`.
            This is an alias for `vars().getRootVar()[name]`.
        */
        ShaderVar operator[](const std::string& name)
        {
            return vars().getRootVar()[name];
        }

        /** allocateStructuredBuffer is a helper method that allocates a
            structured buffer of the given name with the given number of
            elements.  Note: the given structured buffer must be declared at
            global scope.

            TODO: support structured buffers in parameter blocks?
            TODO: add support for other buffer allocation types?

            \param[in] name Name of the buffer in the shader.
            \param[in] nElements Number of elements to allocate.
            \param[in] pInitData Optional parameter. Initial buffer data.
            \param[in] initDataSize Optional parameter. Size of the pointed initial data for validation (if 0 the buffer is assumed to be of the right size).
        */
        void allocateStructuredBuffer(const std::string& name, uint32_t nElements, const void* pInitData = nullptr, size_t initDataSize = 0);

        /** runProgram runs the compute program that was specified in
            |createProgram|, where the total number of threads that runs is
            given by the product of the three provided dimensions.
            \param[in] dimensions Number of threads to dispatch in each dimension.
        */
        void runProgram(const uint3& dimensions);

        /** runProgram runs the compute program that was specified in
            |createProgram|, where the total number of threads that runs is
            given by the product of the three provided dimensions.
        */
        void runProgram(uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1) { runProgram(uint3(width, height, depth)); }

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

        /** Returns the current Falcor render context.
        */
        RenderContext* getRenderContext() const { return mpContext; }

        /** Returns the program.
        */
        ComputeProgram* getProgram() const { return mpProgram.get(); }

    private:
        const void* mapRawRead(const char* bufferName);

        // Internal state
        RenderContext* mpContext;
        ComputeState::SharedPtr mpState;
        ComputeProgram::SharedPtr mpProgram;
        ComputeVars::SharedPtr mpVars;
        uint3 mThreadGroupSize = { 0, 0, 0 };

        struct ParameterBuffer
        {
            Buffer::SharedPtr pBuffer;
            bool mapped = false;
        };
        std::map<std::string, ParameterBuffer> mStructuredBuffers;
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
        /** We need to declare this constructor in order to return
            StreamSinks as rvalues from functions because we've declared a
            StreamSink destructor below.
         */
        StreamSink(StreamSink &&) = default;

        /** Construct a StreamSink for a test context.
            If a non-nullptr UnitTestContext is provided, the values printed
            will be accumulated and passed to the context's reportFailure()
            method when the StreamSink destructor runs.
        */
        StreamSink(UnitTestContext* ctx) : mpCtx(ctx) {}

        ~StreamSink()
        {
            if (mpCtx) mpCtx->reportFailure(mSs.str());
        }

        template <typename T>
        StreamSink& operator<<(T&&s)
        {
            if (mpCtx) mSs << s;
            return *this;
        }

    private:
        std::stringstream mSs;
        UnitTestContext* mpCtx = nullptr;
    };

    template <typename T, typename U>
    inline StreamSink expectEqInternal(T x, const char* xString, U y, const char* yString,
                                       UnitTestContext& ctx, const char* file, int line) {
        if (x == y) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " == " <<
            yString << " (" << x << " vs. " << y << ") ";
        return ss;
    }

    template <typename T, typename U>
    inline StreamSink expectNeInternal(T x, const char* xString, U y, const char* yString,
                                       UnitTestContext& ctx, const char* file, int line) {
        if (x != y) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " != " <<
            yString << " (" << x << " vs. " << y << ") ";
        return ss;
    }

    template <typename T, typename U>
    inline StreamSink expectGeInternal(T x, const char* xString, U y, const char* yString,
                                       UnitTestContext& ctx, const char* file, int line) {
        if (x >= y) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " >= " <<
            yString << " (" << x << " vs. " << y << ") ";
        return ss;
    }

    template <typename T, typename U>
    inline StreamSink expectGtInternal(T x, const char* xString, U y, const char* yString,
                                       UnitTestContext& ctx, const char* file, int line) {
        if (x > y) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " > " <<
            yString << " (" << x << " vs. " << y << ") ";
        return ss;
    }

    template <typename T, typename U>
    inline StreamSink expectLeInternal(T x, const char* xString, U y, const char* yString,
                                       UnitTestContext& ctx, const char* file, int line) {
        if (x <= y) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " <= " <<
            yString << " (" << x << " vs. " << y << ") ";
        return ss;
    }

    template <typename T, typename U>
    inline StreamSink expectLtInternal(T x, const char* xString, U y, const char* yString,
                                       UnitTestContext& ctx, const char* file, int line) {
        if (x < y) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " < " <<
            yString << " (" << x << " vs. " << y << ") ";
        return ss;
    }

    template <typename T>
    inline StreamSink expectInternal(T x, const char* xString, UnitTestContext& ctx,
                                     const char* file, int line) {
        if (x) return StreamSink(nullptr);

        if (++ctx.mNumFailures == kMaxTestFailures) throw TooManyFailedTestsException();

        StreamSink ss(&ctx);
        ss << file << ":" << line << " Test failed: " << xString << " ";
        return ss;
    }

    ///////////////////////////////////////////////////////////////////////////

    /** Start of user-facing API */

/** Macro to define a CPU unit test. The optional skip message will
    disable the test from running without leading to a failure.
    The macro defines an instance of the |CPUUnitTestRegisterer| class,
    which in turn registers the test with the test framework when its
    constructor executes at program startup time. Next, it starts the
    definition of the testing function, up to the point at which
    the user should supply an open brace and start writing code.
*/
#define CPU_TEST(Name, ...)                                                     \
    static void CPUUnitTest##Name(CPUUnitTestContext& ctx);                     \
    struct CPUUnitTestRegisterer##Name {                                        \
        CPUUnitTestRegisterer##Name()                                           \
        {                                                                       \
            std::filesystem::path path = __FILE__;                              \
            const char* skipMessage = "" __VA_ARGS__;                           \
            registerCPUTest(path, #Name, skipMessage, CPUUnitTest##Name);       \
        }                                                                       \
    } RegisterCPUTest##Name;                                                    \
    static void CPUUnitTest##Name(CPUUnitTestContext& ctx) /* over to the user for the braces */

/** Macro to define a GPU unit test. The optional skip message will
    disable the test from running without leading to a failure.
    The macro works in the same ways as CPU_TEST().
*/
#define GPU_TEST(Name, ...)                                                     \
    static void GPUUnitTest##Name(GPUUnitTestContext& ctx);                     \
    struct GPUUnitTestRegisterer##Name {                                        \
        GPUUnitTestRegisterer##Name()                                           \
        {                                                                       \
            std::filesystem::path path = __FILE__;                              \
            const char* skipMessage = "" __VA_ARGS__;                           \
            registerGPUTest(path, #Name, skipMessage, GPUUnitTest##Name);       \
        }                                                                       \
    } RegisterGPUTest##Name;                                                    \
    static void GPUUnitTest##Name(GPUUnitTestContext& ctx) /* over to the user for the braces */

/** Define GPU_TEST_D3D12 macro that defines a GPU unit test only supported on D3D12.
*/
#if FALCOR_HAS_D3D12
#define GPU_TEST_D3D12(Name, ...) GPU_TEST(Name, __VA_ARGS__)
#else
#define GPU_TEST_D3D12(Name, ...) GPU_TEST(Name, "Not supported on Vulkan.")
#endif

/** Define GPU_TEST_VK macro that defines a GPU unit test only supported on Vulkan.
*/
#if FALCOR_HAS_VULKAN
#define GPU_TEST_VK(Name, ...) GPU_TEST(Name, __VA_ARGS__)
#else
#define GPU_TEST_VK(Name, ...) GPU_TEST(Name, "Not supported on D3D12.")
#endif

/** Macro definitions for the GPU unit testing framework. Note that they
    are all a single statement (including any additional << printed
    values).  Thus, it's perfectly fine to write code like:

    if (foo)  // look, no braces
        EXPECT_EQ(x, y);

    The work of the test and accounting for failures is taken care of by various
    expect*Internal() functions; this ensures that the macro operands are only
    evaluated once and that we can do non-trivial work in a function without
    getting into contortions.
*/

#define EXPECT_EQ(x, y) expectEqInternal((x), #x, (y), #y, ctx, __FILE__, __LINE__)
#define EXPECT_NE(x, y) expectNeInternal((x), #x, (y), #y, ctx, __FILE__, __LINE__)
#define EXPECT_GE(x, y) expectGeInternal((x), #x, (y), #y, ctx, __FILE__, __LINE__)
#define EXPECT_GT(x, y) expectGtInternal((x), #x, (y), #y, ctx, __FILE__, __LINE__)
#define EXPECT_LE(x, y) expectLeInternal((x), #x, (y), #y, ctx, __FILE__, __LINE__)
#define EXPECT_LT(x, y) expectLtInternal((x), #x, (y), #y, ctx, __FILE__, __LINE__)
#define EXPECT(x)       expectInternal((x), #x, ctx, __FILE__, __LINE__)

} // namespace Falcor
