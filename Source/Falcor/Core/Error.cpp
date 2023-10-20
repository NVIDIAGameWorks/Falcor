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
#include "Error.h"
#include "Platform/OS.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <atomic>

namespace Falcor
{

/// Global error diagnostic flags.
static ErrorDiagnosticFlags gErrorDiagnosticFlags = ErrorDiagnosticFlags::BreakOnThrow | ErrorDiagnosticFlags::BreakOnAssert;

void throwException(const fstd::source_location& loc, std::string_view msg)
{
    std::string fullMsg = fmt::format("{}\n\n{}:{} ({})", msg, loc.file_name(), loc.line(), loc.function_name());

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::AppendStackTrace))
        fullMsg += fmt::format("\n\nStacktrace:\n{}", getStackTrace(1));

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::BreakOnThrow) && isDebuggerPresent())
        debugBreak();

    throw RuntimeError(fullMsg);
}

void reportAssertion(const fstd::source_location& loc, std::string_view cond, std::string_view msg)
{
    std::string fullMsg = fmt::format(
        "Assertion failed: {}\n{}{}\n{}:{} ({})", cond, msg, msg.empty() ? "" : "\n", loc.file_name(), loc.line(), loc.function_name()
    );

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::AppendStackTrace))
        fullMsg += fmt::format("\n\nStacktrace:\n{}", getStackTrace(1));

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::BreakOnAssert) && isDebuggerPresent())
        debugBreak();

    throw AssertionError(fullMsg);
}

//
// Error handling.
//

void setErrorDiagnosticFlags(ErrorDiagnosticFlags flags)
{
    gErrorDiagnosticFlags = flags;
}

ErrorDiagnosticFlags getErrorDiagnosticFlags()
{
    return gErrorDiagnosticFlags;
}

void reportErrorAndContinue(std::string_view msg)
{
    logError(msg);

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::ShowMessageBoxOnError))
    {
        // Show message box
        msgBox("Error", std::string(msg), MsgBoxType::Ok, MsgBoxIcon::Error);
    }
}

bool reportErrorAndAllowRetry(std::string_view msg)
{
    logError(msg);

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::ShowMessageBoxOnError))
    {
        enum ButtonId
        {
            Retry,
            Abort
        };

        // Setup message box buttons
        std::vector<MsgBoxCustomButton> buttons;
        buttons.push_back({Retry, "Retry"});
        buttons.push_back({Abort, "Abort"});

        // Show message box
        auto result = msgBox("Error", std::string(msg), buttons, MsgBoxIcon::Error);
        return result == Retry;
    }

    return false;
}

[[noreturn]] void reportFatalErrorAndTerminate(std::string_view msg)
{
    // Immediately terminate on re-entry.
    static std::atomic<bool> entered;
    if (entered.exchange(true) == true)
        std::quick_exit(1);

    std::string fullMsg = fmt::format("{}\n\nStacktrace:\n{}", msg, getStackTrace(3));

    logFatal(fullMsg);

    if (is_set(gErrorDiagnosticFlags, ErrorDiagnosticFlags::ShowMessageBoxOnError))
    {
        enum ButtonId
        {
            Debug,
            Abort
        };

        // Setup message box buttons
        std::vector<MsgBoxCustomButton> buttons;
        if (isDebuggerPresent())
            buttons.push_back({Debug, "Debug"});
        buttons.push_back({Abort, "Abort"});

        // Show message box
        auto result = msgBox("Fatal Error", fullMsg, buttons, MsgBoxIcon::Error);
        if (result == Debug)
            debugBreak();
    }
    else
    {
        if (isDebuggerPresent())
            debugBreak();
    }

    std::quick_exit(1);
}

FALCOR_SCRIPT_BINDING(Error)
{
    pybind11::register_exception<RuntimeError>(m, "RuntimeError");
    pybind11::register_exception<AssertionError>(m, "AssertionError");
}

} // namespace Falcor
