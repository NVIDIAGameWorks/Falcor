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
#pragma once
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Math/Vector.h"
#include "Utils/StringFormatters.h"

#include <glm/gtx/io.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <filesystem>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// @skallweit: This is temporary to allow FalcorTest be compiled unmodified. Needs to be removed.
#include "Core/API/RenderContext.h"

/**
 * This file defines both the user-visible API for the unit testing framework as well as the various classes that implement it
 */

namespace Falcor
{

class RenderContext;

class CPUUnitTestContext;
class GPUUnitTestContext;

static constexpr int kMaxTestFailures = 25;

struct TooManyFailedTestsException : public Exception
{};

class ErrorRunningTestException : public Exception
{
public:
    ErrorRunningTestException(const std::string& what) : Exception(what) {}
};

/// Intentionally not using Falcor::Exception to avoid printing the stack trace
class AssertingTestException : public std::runtime_error
{
public:
    AssertingTestException(const std::string& what) : std::runtime_error(what.c_str()) {}
};

/// Intentionally not using Falcor::Exception to avoid printing the stack trace
class SkippingTestException : public std::runtime_error
{
public:
    SkippingTestException(const std::string& what) : std::runtime_error(what.c_str()) {}
};

using CPUTestFunc = std::function<void(CPUUnitTestContext& ctx)>;
using GPUTestFunc = std::function<void(GPUUnitTestContext& ctx)>;

enum class UnitTestCategoryFlags
{
    None = 0x0,
    CPU = 0x1,
    GPU = 0x2,
    All = CPU | GPU,
};

FALCOR_ENUM_CLASS_OPERATORS(UnitTestCategoryFlags);

enum class UnitTestDeviceFlags
{
    D3D12 = 0x1,
    Vulkan = 0x2,
    All = D3D12 | Vulkan,
};

FALCOR_ENUM_CLASS_OPERATORS(UnitTestDeviceFlags);

FALCOR_API void registerCPUTest(
    const std::filesystem::path& path,
    const std::string& name,
    const std::string& skipMessage,
    CPUTestFunc func
);
FALCOR_API void registerGPUTest(
    const std::filesystem::path& path,
    const std::string& name,
    const std::string& skipMessage,
    GPUTestFunc func,
    UnitTestDeviceFlags supportedDevices
);
FALCOR_API int32_t runTests(
    std::shared_ptr<Device> pDevice,
    Fbo* pTargetFbo,
    UnitTestCategoryFlags categoryFlags,
    const std::string& testFilterRegexp,
    const std::filesystem::path& xmlReportPath,
    uint32_t repeatCount = 1
);

class FALCOR_API UnitTestContext
{
public:
    /**
     * reportFailure is called with an error message to report a failing
     * test.  Normally it's only used by the EXPECT_EQ (etc.) macros,
     * though it's fine for a user to call it with different failures.
     */
    void reportFailure(const std::string& message);

    std::vector<std::string> getFailureMessages() const { return mFailureMessages; }

    int mNumFailures = 0;

private:
    std::vector<std::string> mFailureMessages;
};

class FALCOR_API CPUUnitTestContext : public UnitTestContext
{};

class FALCOR_API GPUUnitTestContext : public UnitTestContext
{
public:
    GPUUnitTestContext(std::shared_ptr<Device> pDevice, Fbo* pTargetFbo) : mpDevice(std::move(pDevice)), mpTargetFbo(pTargetFbo) {}

    /**
     * createProgram creates a compute program from the source code at the
     * given path.  The entrypoint is assumed to be |main()| unless
     * otherwise specified with the |csEntry| parameter.  Preprocessor
     * defines and compiler flags can also be optionally provided.
     */
    void createProgram(
        const std::filesystem::path& path,
        const std::string& csEntry = "main",
        const Program::DefineList& programDefines = Program::DefineList(),
        Shader::CompilerFlags flags = Shader::CompilerFlags::None,
        const std::string& shaderModel = "",
        bool createShaderVars = true
    );

    /**
     * Create compute program based on program desc and defines.
     */
    void createProgram(
        const Program::Desc& desc,
        const Program::DefineList& programDefines = Program::DefineList(),
        bool createShaderVars = true
    );

    /**
     * (Re-)create the shader variables. Call this if vars were not
     * created in createProgram() (if createVars = false), or after
     * the shader variables have changed through specialization.
     */
    void createVars();

    /**
     * vars returns the ComputeVars for the program for use in binding
     * textures, etc.
     */
    ComputeVars& vars()
    {
        FALCOR_ASSERT(mpVars);
        return *mpVars;
    }

    /**
     * Get a shader variable that points at the field with the given `name`.
     * This is an alias for `vars().getRootVar()[name]`.
     */
    ShaderVar operator[](const std::string& name) { return vars().getRootVar()[name]; }

    /**
     * allocateStructuredBuffer is a helper method that allocates a
     * structured buffer of the given name with the given number of
     * elements.  Note: the given structured buffer must be declared at
     * global scope.
     *
     * TODO: support structured buffers in parameter blocks?
     * TODO: add support for other buffer allocation types?
     *
     * @param[in] name Name of the buffer in the shader.
     * @param[in] nElements Number of elements to allocate.
     * @param[in] pInitData Optional parameter. Initial buffer data.
     * @param[in] initDataSize Optional parameter. Size of the pointed initial data for validation (if 0 the buffer is assumed to be of the
     * right size).
     */
    void allocateStructuredBuffer(const std::string& name, uint32_t nElements, const void* pInitData = nullptr, size_t initDataSize = 0);

    /**
     * runProgram runs the compute program that was specified in
     * |createProgram|, where the total number of threads that runs is
     * given by the product of the three provided dimensions.
     * @param[in] dimensions Number of threads to dispatch in each dimension.
     */
    void runProgram(const uint3& dimensions);

    /**
     * runProgram runs the compute program that was specified in
     * |createProgram|, where the total number of threads that runs is
     * given by the product of the three provided dimensions.
     */
    void runProgram(uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1) { runProgram(uint3(width, height, depth)); }

    /**
     * mapBuffer returns a pointer to the named structured buffer.
     * Returns nullptr if no such buffer exists.  SFINAE is used to
     * require that a the requested pointer is const.
     */
    template<typename T>
    T* mapBuffer(const char* bufferName, typename std::enable_if<std::is_const<T>::value>::type* = 0)
    {
        return reinterpret_cast<T*>(mapRawRead(bufferName));
    }

    /**
     * unmapBuffer unmaps a buffer after it's been used after a call to
     * |mapBuffer()|.
     */
    void unmapBuffer(const char* bufferName);

    /**
     * Returns the current Falcor render device.
     */
    const std::shared_ptr<Device>& getDevice() const { return mpDevice; }

    /**
     * Returns the current Falcor render context.
     */
    RenderContext* getRenderContext() const { return mpDevice->getRenderContext(); }

    /**
     * Returns the current FBO.
     */
    Fbo* getTargetFbo() const { return mpTargetFbo; }

    /**
     * Returns the program.
     */
    ComputeProgram* getProgram() const { return mpProgram.get(); }

private:
    const void* mapRawRead(const char* bufferName);

    // Internal state
    std::shared_ptr<Device> mpDevice;
    Fbo* mpTargetFbo;
    ComputeState::SharedPtr mpState;
    ComputeProgram::SharedPtr mpProgram;
    ComputeVars::SharedPtr mpVars;
    uint3 mThreadGroupSize = {0, 0, 0};

    struct ParameterBuffer
    {
        Buffer::SharedPtr pBuffer;
        bool mapped = false;
    };
    std::map<std::string, ParameterBuffer> mStructuredBuffers;
};

namespace unittest
{
/**
 * StreamSink is a utility class used by the testing framework that either
 * captures values printed via C++'s operator<< (as with regular
 * std::ostreams) or discards them.  (If a test has failed, then
 * StreamSink does the former, and if it has passed, it does the latter.)
 * In the event of a test failure, passes along the failure message to the
 * provided GPUUnitTestContext's |reportFailure| method.
 */
class StreamSink
{
public:
    /**
     * We need to declare this constructor in order to return
     * StreamSinks as rvalues from functions because we've declared a
     * StreamSink destructor below.
     */
    StreamSink(StreamSink&&) = default;

    /**
     * Construct a StreamSink for a test context.
     * If a non-nullptr UnitTestContext is provided, the values printed
     * will be accumulated and passed to the context's reportFailure()
     * method when the StreamSink destructor runs.
     */
    StreamSink(UnitTestContext* ctx) : mpCtx(ctx) {}

    ~StreamSink()
    {
        if (mpCtx)
            mpCtx->reportFailure(mSs.str());
    }

    void setInsertNewLine() { mInsertNewLine = true; }

    template<typename T>
    StreamSink& operator<<(T&& s)
    {
        if (mpCtx)
        {
            if (mInsertNewLine)
            {
                mSs << "\n";
                mInsertNewLine = false;
            }
            mSs << s;
        }
        return *this;
    }

private:
    std::stringstream mSs;
    UnitTestContext* mpCtx = nullptr;
    bool mInsertNewLine = false;
};

/// A test has failed when there is a failureMsg, if the value is not set, the test has passed
inline StreamSink createFullMessage(
    UnitTestContext& ctx,
    std::optional<std::string> failureMsg,
    std::string_view userMsg,
    bool isAssert,
    std::string_view file,
    int line
)
{
    if (!failureMsg)
        return StreamSink(nullptr);

    if (++ctx.mNumFailures == kMaxTestFailures)
        throw TooManyFailedTestsException();

    StreamSink ss(&ctx);
    ss << fmt::format("{}:{}:\nerror: {}", file, line, *failureMsg);
    if (!userMsg.empty())
        ss << "\n" << userMsg;
    if (isAssert)
        throw AssertingTestException("Test asserted, couldn't continue.");
    ss.setInsertNewLine();
    return ss;
}

#define FTEST_ASSERT_RESULT true
#define FTEST_EXPECT_RESULT false
#define FTEST_MESSAGE(failureMsg, userFailMsg, isAssert) \
    ::Falcor::unittest::createFullMessage(ctx, failureMsg, userFailMsg, isAssert, __FILE__, __LINE__)

/// If the comparison is failure, it will return a fail message, if it is not a failure, there will be no message
inline std::optional<std::string> createBoolMessage(std::string_view valueStr, bool value, bool expected)
{
    // don't bother creating message when it won't be used
    if (value == expected)
        return std::optional<std::string>{};
    return fmt::format(
        "Expected: ({}) == {}, actual:\n  {} = {}", valueStr, expected ? "true" : "false", valueStr, value ? "true" : "false"
    );
}

#define FTEST_TEST_BOOLEAN(valueStr, value, expected, userFailMsg, isAssert) \
    FTEST_MESSAGE(::Falcor::unittest::createBoolMessage(valueStr, value, expected), userFailMsg, isAssert)

#define FTEST_COMPARISON_HELPER(opName, op)                   \
    struct CmpHelper##opName                                  \
    {                                                         \
        template<typename TLhs, typename TRhs>                \
        static bool compare(const TLhs& lhs, const TRhs& rhs) \
        {                                                     \
            return lhs op rhs;                                \
        }                                                     \
        static const char* asString()                         \
        {                                                     \
            return #op;                                       \
        }                                                     \
    }

FTEST_COMPARISON_HELPER(EQ, ==);
FTEST_COMPARISON_HELPER(NE, !=);
FTEST_COMPARISON_HELPER(LE, <=);
FTEST_COMPARISON_HELPER(GE, >=);
FTEST_COMPARISON_HELPER(LT, <);
FTEST_COMPARISON_HELPER(GT, >);

/// If the failedOpStr is not set, it means it hasn't failed
template<typename TCmpHelper, typename TLhs, typename TRhs>
inline std::optional<std::string> createBinaryMessage(std::string_view lhsStr, std::string_view rhsStr, const TLhs& lhs, const TRhs& rhs)
{
    // don't bother creating message when it won't be used
    if (TCmpHelper::compare(lhs, rhs))
        return std::optional<std::string>{};
    size_t maxSize = std::max(lhsStr.size(), rhsStr.size());
    return fmt::format(
        "Expected: ({}) {} ({}), actual:\n  {:<{}} = {}\n  {:<{}} = {}", lhsStr, TCmpHelper::asString(), rhsStr, lhsStr, maxSize, lhs,
        rhsStr, maxSize, rhs
    );
}

#define FTEST_TEST_BINARY(opHelper, lhs, rhs, userFailMsg, asserts) \
    FTEST_MESSAGE(::Falcor::unittest::createBinaryMessage<::Falcor::unittest::opHelper>(#lhs, #rhs, lhs, rhs), userFailMsg, asserts)

} // namespace unittest

///////////////////////////////////////////////////////////////////////////

/**
 * Start of user-facing API
 */

/**
 * Macro to define a CPU unit test. The optional skip message will
 * disable the test from running without leading to a failure.
 * The macro defines an instance of the |CPUUnitTestRegisterer| class,
 * which in turn registers the test with the test framework when its
 * constructor executes at program startup time. Next, it starts the
 * definition of the testing function, up to the point at which
 * the user should supply an open brace and start writing code.
 */
#define CPU_TEST(name, ...)                                               \
    static void CPUUnitTest##name(CPUUnitTestContext& ctx);               \
    struct CPUUnitTestRegisterer##name                                    \
    {                                                                     \
        CPUUnitTestRegisterer##name()                                     \
        {                                                                 \
            std::filesystem::path path = __FILE__;                        \
            const char* skipMessage = "" __VA_ARGS__;                     \
            registerCPUTest(path, #name, skipMessage, CPUUnitTest##name); \
        }                                                                 \
    } RegisterCPUTest##name;                                              \
    static void CPUUnitTest##name(CPUUnitTestContext& ctx) /* over to the user for the braces */

/**
 * Macro to define a GPU unit test. The optional skip message will
 * disable the test from running without leading to a failure.
 * The macro works in the same ways as CPU_TEST().
 */
#define GPU_TEST_INTERNAL(name, flags, ...)                                      \
    static void GPUUnitTest##name(GPUUnitTestContext& ctx);                      \
    struct GPUUnitTestRegisterer##name                                           \
    {                                                                            \
        GPUUnitTestRegisterer##name()                                            \
        {                                                                        \
            std::filesystem::path path = __FILE__;                               \
            const char* skipMessage = "" __VA_ARGS__;                            \
            registerGPUTest(path, #name, skipMessage, GPUUnitTest##name, flags); \
        }                                                                        \
    } RegisterGPUTest##name;                                                     \
    static void GPUUnitTest##name(GPUUnitTestContext& ctx) /* over to the user for the braces */

#define GPU_TEST(name, ...) GPU_TEST_INTERNAL(name, UnitTestDeviceFlags::All, __VA_ARGS__)

/**
 * Define GPU_TEST_D3D12 macro that defines a GPU unit test only supported on D3D12.
 */
#define GPU_TEST_D3D12(name, ...) GPU_TEST_INTERNAL(name, UnitTestDeviceFlags::D3D12, __VA_ARGS__)

/**
 * Define GPU_TEST_VULKAN macro that defines a GPU unit test only supported on Vulkan.
 */
#define GPU_TEST_VULKAN(name, ...) GPU_TEST_INTERNAL(name, UnitTestDeviceFlags::Vulkan, __VA_ARGS__)

/**
 * Macro definitions for the GPU unit testing framework. Note that they
 * are all a single statement (including any additional << printed
 * values).  Thus, it's perfectly fine to write code like:
 *
 * if (foo)  // look, no braces
 *     EXPECT_EQ(x, y);
 */
#define EXPECT_TRUE_MSG(expression, msg) FTEST_TEST_BOOLEAN(#expression, bool(expression), true, msg, FTEST_EXPECT_RESULT)
#define EXPECT_FALSE_MSG(expression, msg) FTEST_TEST_BOOLEAN(#expression, bool(expression), false, msg, FTEST_EXPECT_RESULT)
#define ASSERT_TRUE_MSG(expression, msg) FTEST_TEST_BOOLEAN(#expression, bool(expression), true, msg, FTEST_ASSERT_RESULT)
#define ASSERT_FALSE_MSG(expression, msg) FTEST_TEST_BOOLEAN(#expression, bool(expression), false, msg, FTEST_ASSERT_RESULT)

#define EXPECT_TRUE(expression) EXPECT_TRUE_MSG(expression, "")
#define EXPECT_FALSE(expression) EXPECT_FALSE_MSG(expression, "")
#define ASSERT_TRUE(expression) ASSERT_TRUE_MSG(expression, "")
#define ASSERT_FALSE(expression) ASSERT_FALSE_MSG(expression, "")

#define EXPECT(expression) EXPECT_TRUE(expression)
#define ASSERT(expression) ASSERT_TRUE(expression)
#define EXPECT_MSG(expression, msg) EXPECT_TRUE_MSG(expression, msg)
#define ASSERT_MSG(expression, msg) ASSERT_TRUE_MSG(expression, msg)

#define EXPECT_EQ_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperEQ, lhs, rhs, msg, FTEST_EXPECT_RESULT)
#define EXPECT_NE_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperNE, lhs, rhs, msg, FTEST_EXPECT_RESULT)
#define EXPECT_LE_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperLE, lhs, rhs, msg, FTEST_EXPECT_RESULT)
#define EXPECT_GE_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperGE, lhs, rhs, msg, FTEST_EXPECT_RESULT)
#define EXPECT_LT_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperLT, lhs, rhs, msg, FTEST_EXPECT_RESULT)
#define EXPECT_GT_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperGT, lhs, rhs, msg, FTEST_EXPECT_RESULT)

#define ASSERT_EQ_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperEQ, lhs, rhs, msg, FTEST_ASSERT_RESULT)
#define ASSERT_NE_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperNE, lhs, rhs, msg, FTEST_ASSERT_RESULT)
#define ASSERT_LE_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperLE, lhs, rhs, msg, FTEST_ASSERT_RESULT)
#define ASSERT_GE_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperGE, lhs, rhs, msg, FTEST_ASSERT_RESULT)
#define ASSERT_LT_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperLT, lhs, rhs, msg, FTEST_ASSERT_RESULT)
#define ASSERT_GT_MSG(lhs, rhs, msg) FTEST_TEST_BINARY(CmpHelperGT, lhs, rhs, msg, FTEST_ASSERT_RESULT)

#define EXPECT_EQ(lhs, rhs) EXPECT_EQ_MSG(lhs, rhs, "")
#define EXPECT_NE(lhs, rhs) EXPECT_NE_MSG(lhs, rhs, "")
#define EXPECT_LE(lhs, rhs) EXPECT_LE_MSG(lhs, rhs, "")
#define EXPECT_GE(lhs, rhs) EXPECT_GE_MSG(lhs, rhs, "")
#define EXPECT_LT(lhs, rhs) EXPECT_LT_MSG(lhs, rhs, "")
#define EXPECT_GT(lhs, rhs) EXPECT_GT_MSG(lhs, rhs, "")

#define ASSERT_EQ(lhs, rhs) ASSERT_EQ_MSG(lhs, rhs, "")
#define ASSERT_NE(lhs, rhs) ASSERT_NE_MSG(lhs, rhs, "")
#define ASSERT_LE(lhs, rhs) ASSERT_LE_MSG(lhs, rhs, "")
#define ASSERT_GE(lhs, rhs) ASSERT_GE_MSG(lhs, rhs, "")
#define ASSERT_LT(lhs, rhs) ASSERT_LT_MSG(lhs, rhs, "")
#define ASSERT_GT(lhs, rhs) ASSERT_GT_MSG(lhs, rhs, "")

} // namespace Falcor
