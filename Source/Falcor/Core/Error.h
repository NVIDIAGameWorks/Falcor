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
#include "Macros.h"
#include <fstd/source_location.h> // TODO C++20: Replace with <source_location>
#include <fmt/format.h>           // TODO C++20: Replace with <format>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace Falcor
{

//
// Exceptions.
//

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4275) // allow dllexport on classes dervied from STL
#endif

/**
 * Base class for all Falcor exceptions.
 */
class FALCOR_API Exception : public std::exception
{
public:
    Exception() noexcept {}
    Exception(std::string_view what) : mpWhat(std::make_shared<std::string>(what)) {}
    Exception(const Exception& other) noexcept { mpWhat = other.mpWhat; }
    virtual ~Exception() override {}
    virtual const char* what() const noexcept override { return mpWhat ? mpWhat->c_str() : ""; }

protected:
    // Message is stored as a reference counted string in order to allow copy constructor to be noexcept.
    std::shared_ptr<std::string> mpWhat;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/**
 * Exception to be thrown when an error happens at runtime.
 */
class FALCOR_API RuntimeError : public Exception
{
public:
    RuntimeError() noexcept {}
    RuntimeError(std::string_view what) : Exception(what) {}
    RuntimeError(const RuntimeError& other) noexcept { mpWhat = other.mpWhat; }
    virtual ~RuntimeError() override {}
};

/**
 * Exception to be thrown on FALCOR_ASSERT.
 */
class FALCOR_API AssertionError : public Exception
{
public:
    AssertionError() noexcept {}
    AssertionError(std::string_view what) : Exception(what) {}
    AssertionError(const AssertionError& other) noexcept { mpWhat = other.mpWhat; }
    virtual ~AssertionError() override {}
};

//
// Exception helpers.
//

/// Throw a RuntimeError exception.
/// If ErrorDiagnosticFlags::AppendStackTrace is set, a stack trace will be appended to the exception message.
/// If ErrorDiagnosticFlags::BreakOnThrow is set, the debugger will be broken into (if attached).
[[noreturn]] FALCOR_API void throwException(const fstd::source_location& loc, std::string_view msg);

namespace detail
{
/// Overload to allow FALCOR_THROW to be called with a message only.
[[noreturn]] inline void throwException(const fstd::source_location& loc, std::string_view msg)
{
    ::Falcor::throwException(loc, msg);
}

/// Overload to allow FALCOR_THROW to be called with a format string and arguments.
template<typename... Args>
[[noreturn]] inline void throwException(const fstd::source_location& loc, fmt::format_string<Args...> fmt, Args&&... args)
{
    ::Falcor::throwException(loc, fmt::format(fmt, std::forward<Args>(args)...));
}
} // namespace detail
} // namespace Falcor

/// Helper for throwing a RuntimeError exception.
/// Accepts either a string or a format string and arguments:
/// FALCOR_THROW("This is an error message.");
/// FALCOR_THROW("Expected {} items, got {}.", expectedCount, actualCount);
#define FALCOR_THROW(...) ::Falcor::detail::throwException(fstd::source_location::current(), __VA_ARGS__)

/// Helper for throwing a RuntimeError exception if condition isn't met.
/// Accepts either a string or a format string and arguments.
/// FALCOR_CHECK(device != nullptr, "Device is null.");
/// FALCOR_CHECK(count % 3 == 0, "Count must be a multiple of 3, got {}.", count);
#define FALCOR_CHECK(cond, ...)        \
    do                                 \
    {                                  \
        if (!(cond))                   \
            FALCOR_THROW(__VA_ARGS__); \
    } while (0)

/// Helper for marking unimplemented functions.
#define FALCOR_UNIMPLEMENTED() FALCOR_THROW("Unimplemented")

/// Helper for marking unreachable code.
#define FALCOR_UNREACHABLE() FALCOR_THROW("Unreachable")

//
// Assertions.
//

namespace Falcor
{
/// Report an assertion.
/// If ErrorDiagnosticFlags::AppendStackTrace is set, a stack trace will be appended to the exception message.
/// If ErrorDiagnosticFlags::BreakOnAssert is set, the debugger will be broken into (if attached).
[[noreturn]] FALCOR_API void reportAssertion(const fstd::source_location& loc, std::string_view cond, std::string_view msg = {});

namespace detail
{
/// Overload to allow FALCOR_ASSERT to be called without a message.
[[noreturn]] inline void reportAssertion(const fstd::source_location& loc, std::string_view cond)
{
    ::Falcor::reportAssertion(loc, cond);
}

/// Overload to allow FALCOR_ASSERT to be called with a message only.
[[noreturn]] inline void reportAssertion(const fstd::source_location& loc, std::string_view cond, std::string_view msg)
{
    ::Falcor::reportAssertion(loc, cond, msg);
}

/// Overload to allow FALCOR_ASSERT to be called with a format string and arguments.
template<typename... Args>
[[noreturn]] inline void reportAssertion(
    const fstd::source_location& loc,
    std::string_view cond,
    fmt::format_string<Args...> fmt,
    Args&&... args
)
{
    ::Falcor::reportAssertion(loc, cond, fmt::format(fmt, std::forward<Args>(args)...));
}
} // namespace detail
} // namespace Falcor

#if FALCOR_ENABLE_ASSERTS

/// Helper for asserting a condition.
/// Accepts either only the condition, the condition with a string or the condition with a format string and arguments:
/// FALCOR_ASSERT(device != nullptr);
/// FALCOR_ASSERT(device != nullptr, "Device is null.");
/// FALCOR_ASSERT(count % 3 == 0, "Count must be a multiple of 3, got {}.", count);
#define FALCOR_ASSERT(cond, ...)                                                                   \
    if (!(cond))                                                                                   \
    {                                                                                              \
        ::Falcor::detail::reportAssertion(fstd::source_location::current(), #cond, ##__VA_ARGS__); \
    }

/// Helper for asserting a binary comparison between two variables.
/// Automatically prints the compared values.
#define FALCOR_ASSERT_OP(a, b, OP)                                                                                                       \
    if (!(a OP b))                                                                                                                       \
    {                                                                                                                                    \
        ::Falcor::detail::reportAssertion(fstd::source_location::current(), fmt::format("{} {} {} ({} {} {})", #a, #OP, #b, a, #OP, b)); \
    }

#define FALCOR_ASSERT_EQ(a, b) FALCOR_ASSERT_OP(a, b, ==)
#define FALCOR_ASSERT_NE(a, b) FALCOR_ASSERT_OP(a, b, !=)
#define FALCOR_ASSERT_GE(a, b) FALCOR_ASSERT_OP(a, b, >=)
#define FALCOR_ASSERT_GT(a, b) FALCOR_ASSERT_OP(a, b, >)
#define FALCOR_ASSERT_LE(a, b) FALCOR_ASSERT_OP(a, b, <=)
#define FALCOR_ASSERT_LT(a, b) FALCOR_ASSERT_OP(a, b, <)

#else // FALCOR_ENABLE_ASSERTS

#define FALCOR_ASSERT(cond, ...) \
    {}
#define FALCOR_ASSERT_OP(a, b, OP) \
    {}
#define FALCOR_ASSERT_EQ(a, b) FALCOR_ASSERT_OP(a, b, ==)
#define FALCOR_ASSERT_NE(a, b) FALCOR_ASSERT_OP(a, b, !=)
#define FALCOR_ASSERT_GE(a, b) FALCOR_ASSERT_OP(a, b, >=)
#define FALCOR_ASSERT_GT(a, b) FALCOR_ASSERT_OP(a, b, >)
#define FALCOR_ASSERT_LE(a, b) FALCOR_ASSERT_OP(a, b, <=)
#define FALCOR_ASSERT_LT(a, b) FALCOR_ASSERT_OP(a, b, <)

#endif // FALCOR_ENABLE_ASSERTS

//
// Error reporting.
//

namespace Falcor
{

/// Flags controlling the error diagnostic behavior.
enum class ErrorDiagnosticFlags
{
    None = 0,
    /// Break into debugger (if attached) when calling FALCOR_THROW.
    BreakOnThrow,
    /// Break into debugger (if attached) when calling FALCOR_ASSERT.
    BreakOnAssert,
    /// Append a stack trace to the exception error message when using FALCOR_THROW and FALCOR_ASSERT.
    AppendStackTrace = 2,
    /// Show a message box when reporting errors using the reportError() functions.
    ShowMessageBoxOnError = 4,
};
FALCOR_ENUM_CLASS_OPERATORS(ErrorDiagnosticFlags);

/// Set the global error diagnostic flags.
FALCOR_API void setErrorDiagnosticFlags(ErrorDiagnosticFlags flags);

/// Get the global error diagnostic flags.
FALCOR_API ErrorDiagnosticFlags getErrorDiagnosticFlags();

/**
 * Report an error by logging it and optionally showing a message box.
 * The message box is only shown if ErrorDiagnosticFlags::ShowMessageBoxOnError is set.
 * @param msg Error message.
 */
FALCOR_API void reportErrorAndContinue(std::string_view msg);

/**
 * Report an error by logging it and optionally showing a message box with the option to abort or retry.
 * The message box is only shown if ErrorDiagnosticFlags::ShowMessageBoxOnError is set.
 * If not message box is shown, the function always returns false (i.e. no retry).
 * @param msg Error message.
 * @return Returns true if the user chose to retry.
 */
FALCOR_API bool reportErrorAndAllowRetry(std::string_view msg);

/**
 * Report a fatal error.
 *
 * The following actions are taken:
 * - The error message is logged together with a stack trace.
 * - If ErrorDiagnosticFlags::ShowMessageBoxOnError is set:
 *   - A message box with the error + stack trace is shown.
 *   - If a debugger is attached there is a button to break into the debugger.
 * - If ErrorDiagnosticFlags::ShowMessageBoxOnError is not set:
 *   - If a debugger is attached there it is broken into.
 * - The application is immediately terminated (std::quick_exit(1)).
 * @param msg Error message.
 */
[[noreturn]] FALCOR_API void reportFatalErrorAndTerminate(std::string_view msg);

/// Helper to run a callback and catch/report all exceptions.
/// This is typically used in main() to guard the entire application.
template<typename CallbackT, typename ResultT = int>
int catchAndReportAllExceptions(CallbackT callback, ResultT errorResult = 1)
{
    ResultT result = errorResult;
    try
    {
        result = callback();
    }
    catch (const std::exception& e)
    {
        reportErrorAndContinue(std::string("Caught an exception:\n\n") + e.what());
    }
    catch (...)
    {
        reportErrorAndContinue("Caught an exception of unknown type!");
    }
    return result;
}

} // namespace Falcor
