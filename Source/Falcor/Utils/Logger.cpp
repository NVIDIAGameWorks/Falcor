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
#include "Logger.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <iostream>
#include <string>
#include <mutex>
#include <set>

namespace Falcor
{
namespace
{
std::mutex sMutex;
Logger::Level sVerbosity = Logger::Level::Info;
Logger::OutputFlags sOutputs = Logger::OutputFlags::Console | Logger::OutputFlags::File | Logger::OutputFlags::DebugWindow;
std::filesystem::path sLogFilePath;

bool sInitialized = false;
FILE* sLogFile = nullptr;

std::filesystem::path generateLogFilePath()
{
    std::string prefix = getExecutableName();
    std::filesystem::path directory = getRuntimeDirectory();
    return findAvailableFilename(prefix, directory, "log");
}

FILE* openLogFile()
{
    FILE* pFile = nullptr;

    if (sLogFilePath.empty())
    {
        sLogFilePath = generateLogFilePath();
    }

    pFile = std::fopen(sLogFilePath.string().c_str(), "w");
    if (pFile != nullptr)
    {
        // Success
        return pFile;
    }

    // If we got here, we couldn't create a log file
    FALCOR_UNREACHABLE();
    return pFile;
}

void printToLogFile(const std::string& s)
{
    if (!sInitialized)
    {
        sLogFile = openLogFile();
        sInitialized = true;
    }

    if (sLogFile)
    {
        std::fprintf(sLogFile, "%s", s.c_str());
        std::fflush(sLogFile);
    }
}
} // namespace

void Logger::shutdown()
{
    if (sLogFile)
    {
        fclose(sLogFile);
        sLogFile = nullptr;
        sInitialized = false;
    }
}

inline const char* getLogLevelString(Logger::Level level)
{
    switch (level)
    {
    case Logger::Level::Fatal:
        return "(Fatal)";
    case Logger::Level::Error:
        return "(Error)";
    case Logger::Level::Warning:
        return "(Warning)";
    case Logger::Level::Info:
        return "(Info)";
    case Logger::Level::Debug:
        return "(Debug)";
    default:
        FALCOR_UNREACHABLE();
        return nullptr;
    }
}

class MessageDeduplicator
{
public:
    static MessageDeduplicator& instance()
    {
        static MessageDeduplicator sInstance;
        return sInstance;
    }

    bool isDuplicate(std::string_view msg)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mStrings.find(msg);
        if (it != mStrings.end())
            return true;
        mStrings.insert(std::string(msg));
        return false;
    }

private:
    MessageDeduplicator() = default;

    std::mutex mMutex;
    std::set<std::string, std::less<>> mStrings;
};

void Logger::log(Level level, const std::string_view msg, Frequency frequency)
{
    std::lock_guard<std::mutex> lock(sMutex);
    if (level <= sVerbosity)
    {
        std::string s = fmt::format("{} {}\n", getLogLevelString(level), msg);

        if (frequency == Frequency::Once && MessageDeduplicator::instance().isDuplicate(s))
            return;

        // Write to console.
        if (is_set(sOutputs, OutputFlags::Console))
        {
            auto& os = level > Logger::Level::Error ? std::cout : std::cerr;
            os << s;
            os.flush();
        }

        // Write to file.
        if (is_set(sOutputs, OutputFlags::File))
        {
            printToLogFile(s);
        }

        // Write to debug window if debugger is attached.
        if (is_set(sOutputs, OutputFlags::DebugWindow) && isDebuggerPresent())
        {
            printToDebugWindow(s);
        }
    }
}

void Logger::setVerbosity(Level level)
{
    std::lock_guard<std::mutex> lock(sMutex);
    sVerbosity = level;
}

Logger::Level Logger::getVerbosity()
{
    std::lock_guard<std::mutex> lock(sMutex);
    return sVerbosity;
}

void Logger::setOutputs(OutputFlags outputs)
{
    std::lock_guard<std::mutex> lock(sMutex);
    sOutputs = outputs;
}

Logger::OutputFlags Logger::getOutputs()
{
    std::lock_guard<std::mutex> lock(sMutex);
    return sOutputs;
}

void Logger::setLogFilePath(const std::filesystem::path& path)
{
    std::lock_guard<std::mutex> lock(sMutex);
    if (sLogFile)
    {
        fclose(sLogFile);
        sLogFile = nullptr;
        sInitialized = false;
    }
    sLogFilePath = path;
}

std::filesystem::path Logger::getLogFilePath()
{
    std::lock_guard<std::mutex> lock(sMutex);
    return sLogFilePath;
}

FALCOR_SCRIPT_BINDING(Logger)
{
    using namespace pybind11::literals;

    pybind11::class_<Logger> logger(m, "Logger");

    pybind11::enum_<Logger::Level> level(logger, "Level");
    level.value("Disabled", Logger::Level::Disabled);
    level.value("Fatal", Logger::Level::Fatal);
    level.value("Error", Logger::Level::Error);
    level.value("Warning", Logger::Level::Warning);
    level.value("Info", Logger::Level::Info);
    level.value("Debug", Logger::Level::Debug);

    pybind11::enum_<Logger::OutputFlags> outputFlags(logger, "OutputFlags");
    outputFlags.value("None_", Logger::OutputFlags::None);
    outputFlags.value("Console", Logger::OutputFlags::Console);
    outputFlags.value("File", Logger::OutputFlags::File);
    outputFlags.value("DebugWindow", Logger::OutputFlags::DebugWindow);

    logger.def_property_static(
        "verbosity",
        [](pybind11::object) { return Logger::getVerbosity(); },
        [](pybind11::object, Logger::Level verbosity) { Logger::setVerbosity(verbosity); }
    );
    logger.def_property_static(
        "outputs",
        [](pybind11::object) { return Logger::getOutputs(); },
        [](pybind11::object, Logger::OutputFlags outputs) { Logger::setOutputs(outputs); }
    );
    logger.def_property_static(
        "log_file_path",
        [](pybind11::object) { return Logger::getLogFilePath(); },
        [](pybind11::object, std::filesystem::path path) { Logger::setLogFilePath(path); }
    );

    logger.def_static(
        "log",
        [](Logger::Level level, const std::string_view msg) { Logger::log(level, msg, Logger::Frequency::Always); },
        "level"_a,
        "msg"_a
    );
}

} // namespace Falcor
