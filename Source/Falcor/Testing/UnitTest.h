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
#include "Core/Error.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Math/Vector.h"
#include "Utils/StringFormatters.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <filesystem>
#include <functional>
#include <map>
#include <set>
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

namespace unittest
{

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

struct RunOptions
{
    Device::Desc deviceDesc;
    std::string testSuiteFilter;
    std::string testCaseFilter;
    std::string tagFilter;
    std::filesystem::path xmlReportPath;
    uint32_t parallel = 1;
    uint32_t repeat = 1;
};

FALCOR_API int32_t runTests(const RunOptions& options);

class CPUUnitTestContext;
class GPUUnitTestContext;

using CPUTestFunc = std::function<void(CPUUnitTestContext& ctx)>;
using GPUTestFunc = std::function<void(GPUUnitTestContext& ctx)>;

struct Test
{
    std::string suiteName;
    std::string name;
    std::set<std::string> tags;
    std::string skipMessage;
    Device::Type deviceType;

    CPUTestFunc cpuFunc;
    GPUTestFunc gpuFunc;
};

/// Enumerate all tests.
FALCOR_API std::vector<Test> enumerateTests();

/// Filter tests by suite and case name.
FALCOR_API std::vector<Test> filterTests(
    std::vector<Test> tests,
    std::string testSuiteFilter,
    std::string testCaseFilter,
    std::string tagFilter,
    Device::Type deviceType
);

class FALCOR_API UnitTestContext
{
public:
    /**
     * Skip the current test at runtime.
     */
    void skip(const char* message) { throw SkippingTestException(message); }

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
    GPUUnitTestContext(ref<Device> pDevice) : mpDevice(pDevice) {}

    /**
     * createProgram creates a compute program from the source code at the
     * given path.  The entrypoint is assumed to be |main()| unless
     * otherwise specified with the |csEntry| parameter.  Preprocessor
     * defines and compiler flags can also be optionally provided.
     */
    void createProgram(
        const std::filesystem::path& path,
        const std::string& csEntry = "main",
        const DefineList& programDefines = DefineList(),
        SlangCompilerFlags flags = SlangCompilerFlags::None,
        ShaderModel shaderModel = ShaderModel::Unknown,
        bool createShaderVars = true
    );

    /**
     * Create compute program based on program desc and defines.
     */
    void createProgram(const ProgramDesc& desc, const DefineList& programDefines = DefineList(), bool createShaderVars = true);

    /**
     * (Re-)create the shader variables. Call this if vars were not
     * created in createProgram() (if createVars = false), or after
     * the shader variables have changed through specialization.
     */
    void createVars();

    /**
     * vars returns the ProgramVars for the program for use in binding
     * textures, etc.
     */
    ProgramVars& vars()
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
     * Read the contents of a structured buffer into a vector.
     */
    template<typename T>
    std::vector<T> readBuffer(const char* bufferName)
    {
        FALCOR_ASSERT(mStructuredBuffers.find(bufferName) != mStructuredBuffers.end());
        auto it = mStructuredBuffers.find(bufferName);
        if (it == mStructuredBuffers.end())
            throw ErrorRunningTestException(std::string(bufferName) + ": couldn't find buffer to map");
        ref<Buffer> buffer = it->second;
        std::vector<T> result = buffer->getElements<T>();
        return result;
    }

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
     * Returns the current Falcor render device.
     */
    const ref<Device>& getDevice() const { return mpDevice; }

    /**
     * Returns the current Falcor render context.
     */
    RenderContext* getRenderContext() const { return mpDevice->getRenderContext(); }

    /**
     * Returns the program.
     */
    Program* getProgram() const { return mpProgram.get(); }

    /**
     * Returns the program vars.
     */
    ProgramVars* getVars() const { return mpVars.get(); }

private:
    // Internal state
    ref<Device> mpDevice;
    ref<ComputeState> mpState;
    ref<Program> mpProgram;
    ref<ProgramVars> mpVars;
    uint3 mThreadGroupSize = {0, 0, 0};

    std::map<std::string, ref<Buffer>> mStructuredBuffers;
};

struct Tags
{
    Tags(std::string tag) { tags.push_back(std::move(tag)); }
    Tags(std::initializer_list<const char*> tags_)
    {
        for (const char* tag : tags_)
            tags.push_back(tag);
    }

    std::vector<std::string> tags;
};

struct Skip
{
    Skip(std::string msg_) : msg(std::move(msg_)) {}

    std::string msg;
};

struct DeviceTypes
{
    DeviceTypes(Device::Type deviceType) { deviceTypes.insert(deviceType); }
    DeviceTypes(std::initializer_list<Device::Type> deviceTypes_)
    {
        for (Device::Type type : deviceTypes_)
            deviceTypes.insert(type);
    }

    std::set<Device::Type> deviceTypes;
};

struct Options
{
    std::set<std::string> tags;
    std::string skipMessage;
    std::set<Device::Type> deviceTypes;
};

inline void applyArg(Options& options, Tags&& arg)
{
    options.tags.insert(arg.tags.begin(), arg.tags.end());
}

inline void applyArg(Options& options, Skip&& arg)
{
    options.skipMessage = std::move(arg.msg);
}

inline void applyArg(Options& options, DeviceTypes&& arg)
{
    options.deviceTypes.insert(arg.deviceTypes.begin(), arg.deviceTypes.end());
}

inline void applyArg(Options& options, Device::Type deviceType)
{
    options.deviceTypes.insert(deviceType);
}

template<size_t N>
inline void applyArg(Options& options, const char (&skipMsg)[N])
{
    options.skipMessage = std::string(skipMsg, N - 1);
}

template<typename... Args>
void applyArgs(Options& options, Args&&... args)
{
    (applyArg(options, std::forward<Args>(args)), ...);
}

FALCOR_API void registerCPUTest(std::filesystem::path path, std::string name, unittest::Options options, CPUTestFunc func);
FALCOR_API void registerGPUTest(std::filesystem::path path, std::string name, unittest::Options options, GPUTestFunc func);

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

struct CmpHelperEQ
{
    template<typename TLhs, typename TRhs>
    static bool compare(const TLhs& lhs, const TRhs& rhs)
    {
        if constexpr (std::is_same_v<TLhs, TRhs>)
            return ::std::equal_to<TLhs>{}(lhs, rhs);
        else
            return lhs == rhs;
    }
    static const char* asString() { return "=="; }
};

struct CmpHelperNE
{
    template<typename TLhs, typename TRhs>
    static bool compare(const TLhs& lhs, const TRhs& rhs)
    {
        if constexpr (std::is_same_v<TLhs, TRhs>)
            return ::std::not_equal_to<TLhs>{}(lhs, rhs);
        else
            return lhs != rhs;
    }
    static const char* asString() { return "!="; }
};

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
        "Expected: ({}) {} ({}), actual:\n  {:<{}} = {}\n  {:<{}} = {}",
        lhsStr,
        TCmpHelper::asString(),
        rhsStr,
        lhsStr,
        maxSize,
        lhs,
        rhsStr,
        maxSize,
        rhs
    );
}

#define FTEST_TEST_BINARY(opHelper, lhs, rhs, userFailMsg, asserts) \
    FTEST_MESSAGE(::Falcor::unittest::createBinaryMessage<::Falcor::unittest::opHelper>(#lhs, #rhs, lhs, rhs), userFailMsg, asserts)

} // namespace unittest

///////////////////////////////////////////////////////////////////////////

/**
 * Start of user-facing API
 */

using UnitTestContext = unittest::UnitTestContext;
using CPUUnitTestContext = unittest::CPUUnitTestContext;
using GPUUnitTestContext = unittest::GPUUnitTestContext;

/**
 * Macro to define a CPU unit test. The optional arguments include:
 *
 * - SKIP(msg): Skip the test with the given message (expands to unittest::Skip).
 * - TAGS(...): A list of tags to associate with the test (expands to unittest::Tags).
 *
 * Some examples:
 *
 * CPU_TEST(Test1) {} // Test is always run
 * CPU_TEST(Test2, SKIP("Not implemented")) {} // Test is skipped
 * CPU_TEST(Test3, TAGS("tag1", "tag2")) {} // Test is run and tagged with "tag1" and "tag2"
 *
 * For convenience, and for backwards compatibility, a string can be used as an
 * optional argument to skip the test:
 *
 * CPU_TEST(Test4, "Not implemented") {} // Test is skipped (same as above)
 *
 * Note: All CPU tests are implicitly tagged with "cpu".
 */
#define CPU_TEST(name, ...)                                                     \
    static void CPUUnitTest##name(CPUUnitTestContext& ctx);                     \
    struct CPUUnitTestRegisterer##name                                          \
    {                                                                           \
        CPUUnitTestRegisterer##name()                                           \
        {                                                                       \
            std::filesystem::path path = __FILE__;                              \
            unittest::Options options;                                          \
            applyArgs(options, ##__VA_ARGS__);                                  \
            options.tags.insert("cpu");                                         \
            unittest::registerCPUTest(path, #name, options, CPUUnitTest##name); \
        }                                                                       \
    } RegisterCPUTest##name;                                                    \
    static void CPUUnitTest##name(CPUUnitTestContext& ctx) /* over to the user for the braces */

/**
 * Macro to define a GPU unit test. The optional arguments include:
 *
 * - SKIP(msg): Skip the test with the given message (expands to unittest::Skip).
 * - TAGS(...): A list of tags to associate with the test (expands to unittest::Tags).
 * - DEVICE_TYPES(...): A list of device types to run the test on (expands to unittest::DeviceTypes).
 *
 * Some examples:
 *
 * GPU_TEST(Test1) {} // Test is always run
 * GPU_TEST(Test2, SKIP("Not implemented")) {} // Test is skipped
 * GPU_TEST(Test3, TAGS("tag1", "tag2")) {} // Test is run and tagged with "tag1" and "tag2"
 * GPU_TEST(Test4, DEVICE_TYPES(Device::Type::D3D12)) {} // Test is only run on D3D12
 *
 * For convenience, and for backwards compatibility, a string can be used as an
 * optional argument to skip the test:
 *
 * GPU_TEST(Test5, "Not implemented") {} // Test is skipped (same as above)
 *
 * Also, Device::Type values can be used as optional arguments to specify a
 * device type to run the test on:
 *
 * GPU_TEST(Test6, Device::Type::D3D12) {} // Test is only run on D3D12 (same as above)
 *
 * Note: All GPU tests are implicitly tagged with "gpu".
 */
#define GPU_TEST(name, ...)                                                     \
    static void GPUUnitTest##name(GPUUnitTestContext& ctx);                     \
    struct GPUUnitTestRegisterer##name                                          \
    {                                                                           \
        GPUUnitTestRegisterer##name()                                           \
        {                                                                       \
            std::filesystem::path path = __FILE__;                              \
            unittest::Options options;                                          \
            applyArgs(options, ##__VA_ARGS__);                                  \
            options.tags.insert("gpu");                                         \
            unittest::registerGPUTest(path, #name, options, GPUUnitTest##name); \
        }                                                                       \
    } RegisterGPUTest##name;                                                    \
    static void GPUUnitTest##name(GPUUnitTestContext& ctx) /* over to the user for the braces */

// clang-format off

/// Used as an argument of CPU_TEST/GPU_TEST to tag a test with a set of strings.
#define TAGS(...) ::Falcor::unittest::Tags{__VA_ARGS__}
/// Used as an argument of CPU_TEST/GPU_TEST to mark a test to be skipped.
#define SKIP(msg) ::Falcor::unittest::Skip{msg}
/// Used as an argument of GPU_TEST to mark a test to only run for certain devices.
#define DEVICE_TYPES(...) ::Falcor::unittest::DeviceTypes{__VA_ARGS__}

// clang-format on

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

#define EXPECT_THROW(expression) \
    try                          \
    {                            \
        expression;              \
        EXPECT(false);           \
    }                            \
    catch (...)                  \
    {                            \
        EXPECT(true);            \
    }

#define EXPECT_THROW_AS(expression, type) \
    try                                   \
    {                                     \
        expression;                       \
        EXPECT(false);                    \
    }                                     \
    catch (const type&)                   \
    {                                     \
        EXPECT(true);                     \
    }                                     \
    catch (...)                           \
    {                                     \
        EXPECT(false);                    \
    }

} // namespace Falcor
