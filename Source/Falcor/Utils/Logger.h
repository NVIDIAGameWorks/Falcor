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
#include "Core/Macros.h"
#include "Utils/StringFormatters.h"
#include <fmt/core.h>
#include <string_view>
#include <filesystem>

namespace Falcor
{
/**
 * Container class for logging messages.
 * Messages are only printed to the selected outputs if they match the verbosity level.
 */
class FALCOR_API Logger
{
public:
    /// Log message severity.
    enum class Level
    {
        Disabled, ///< Disable log messages.
        Fatal,    ///< Fatal messages.
        Error,    ///< Error messages.
        Warning,  ///< Warning messages.
        Info,     ///< Informative messages.
        Debug,    ///< Debugging messages.
        Count,    ///< Keep this last.
    };

    enum class Frequency
    {
        Always, ///< Reports the message always
        Once,   ///< Reports the message only first time the exact string appears
    };

    /// Log output.
    enum class OutputFlags
    {
        None = 0x0,        ///< No output.
        Console = 0x2,     ///< Output to console (stdout/stderr).
        File = 0x1,        ///< Output to log file.
        DebugWindow = 0x4, ///< Output to debug window (if debugger is attached).
    };

    /**
     * Shutdown the logger and close the log file.
     */
    static void shutdown();

    /**
     * Set the logger verbosity.
     * @param level Log level.
     */
    static void setVerbosity(Level level);

    /**
     * Get the logger verbosity.
     * @return Return the log level.
     */
    static Level getVerbosity();

    /**
     * Set the logger outputs.
     * @param outputs Log outputs.
     */
    static void setOutputs(OutputFlags outputs);

    /**
     * Get the logger outputs.
     * @return Return the log outputs.
     */
    static OutputFlags getOutputs();

    /**
     * Set the path of the logfile.
     * @param[in] path Logfile path
     */
    static void setLogFilePath(const std::filesystem::path& path);

    /**
     * Get the path of the logfile.
     * @return Returns the path of the logfile.
     */
    static std::filesystem::path getLogFilePath();

    /**
     * Log a message.
     * @param[in] level Log level.
     * @param[in] msg Log message.
     */
    static void log(Level level, const std::string_view msg, Frequency frequency = Frequency::Always);

private:
    Logger() = delete;
};

FALCOR_ENUM_CLASS_OPERATORS(Logger::OutputFlags);

// We define two types of logging helpers, one taking raw strings,
// the other taking formatted strings. We don't want string formatting and
// errors being thrown due to missing arguments when passing raw strings.

inline void logDebug(const std::string_view msg)
{
    Logger::log(Logger::Level::Debug, msg);
}

template<typename... Args>
inline void logDebug(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Debug, fmt::format(format, std::forward<Args>(args)...));
}

inline void logInfo(const std::string_view msg)
{
    Logger::log(Logger::Level::Info, msg);
}

template<typename... Args>
inline void logInfo(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Info, fmt::format(format, std::forward<Args>(args)...));
}

inline void logWarning(const std::string_view msg)
{
    Logger::log(Logger::Level::Warning, msg);
}

template<typename... Args>
inline void logWarning(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Warning, fmt::format(format, std::forward<Args>(args)...));
}

inline void logWarningOnce(const std::string_view msg)
{
    Logger::log(Logger::Level::Warning, msg, Logger::Frequency::Once);
}

template<typename... Args>
inline void logWarningOnce(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Warning, fmt::format(format, std::forward<Args>(args)...), Logger::Frequency::Once);
}

inline void logError(const std::string_view msg)
{
    Logger::log(Logger::Level::Error, msg);
}

template<typename... Args>
inline void logError(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Error, fmt::format(format, std::forward<Args>(args)...));
}

inline void logErrorOnce(const std::string_view msg)
{
    Logger::log(Logger::Level::Error, msg, Logger::Frequency::Once);
}

template<typename... Args>
inline void logErrorOnce(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Error, fmt::format(format, std::forward<Args>(args)...), Logger::Frequency::Once);
}

inline void logFatal(const std::string_view msg)
{
    Logger::log(Logger::Level::Fatal, msg);
}

template<typename... Args>
inline void logFatal(fmt::format_string<Args...> format, Args&&... args)
{
    Logger::log(Logger::Level::Fatal, fmt::format(format, std::forward<Args>(args)...));
}

} // namespace Falcor

#define FALCOR_PRINT(x)                      \
    do                                       \
    {                                        \
        ::Falcor::logInfo("{} = {}", #x, x); \
    } while (0)
