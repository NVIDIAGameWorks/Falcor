/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Logger.h"

namespace Falcor
{
    namespace
    {
        Logger::Level sVerbosity = Logger::Level::Info;
        Logger::OutputFlags sOutputs = Logger::OutputFlags::Console | Logger::OutputFlags::File | Logger::OutputFlags::DebugWindow;
        std::string sLogFilePath;

#if FALCOR_ENABLE_LOGGER
        bool sInitialized = false;
        FILE* sLogFile = nullptr;

        std::string generateLogFilePath()
        {
            // Get current process name
            std::string filename = getExecutableName();

            // Now we have a folder and a filename, look for an available filename (we don't overwrite existing files)
            std::string prefix = std::string(filename);
            std::string executableDir = getExecutableDirectory();
            std::string path;
            if (findAvailableFilename(prefix, executableDir, "log", path))
            {
                return path;
            }
            should_not_get_here();
            return "";
        }

        FILE* openLogFile()
        {
            FILE* pFile = nullptr;

            if (sLogFilePath.empty())
            {
                sLogFilePath = generateLogFilePath();
            }

            pFile = std::fopen(sLogFilePath.c_str(), "w");
            if (pFile != nullptr)
            {
                // Success
                return pFile;
            }

            // If we got here, we couldn't create a log file
            should_not_get_here();
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
#endif
    }

    void Logger::shutdown()
    {
#if FALCOR_ENABLE_LOGGER
        if(sLogFile)
        {
            fclose(sLogFile);
            sLogFile = nullptr;
            sInitialized = false;
        }
#endif
    }

    const char* getLogLevelString(Logger::Level level)
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
            should_not_get_here();
            return nullptr;
        }
    }

    void Logger::log(Level level, const std::string& msg)
    {
#if FALCOR_ENABLE_LOGGER
        if (level <= sVerbosity)
        {
            std::string s = getLogLevelString(level) + std::string(" ") + msg + "\n";

            // Write to console.
            if (is_set(sOutputs, OutputFlags::Console))
            {
                if (level > Logger::Level::Error) std::cout << s;
                else std::cerr << s;
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
#endif
    }

    bool Logger::setLogFilePath(const std::string& path)
    {
#if FALCOR_ENABLE_LOGGER
        if (sLogFile)
        {
            return false;
        }
        else
        {
            sLogFilePath = path;
            return true;
        }
#else
        return false;
#endif
    }

    void Logger::setVerbosity(Level level) { sVerbosity = level; }
    Logger::Level Logger::getVerbosity() { return sVerbosity; }

    void Logger::setOutputs(OutputFlags outputs) { sOutputs = outputs; }
    Logger::OutputFlags Logger::getOutputs() { return sOutputs; }

    const std::string& Logger::getLogFilePath() { return sLogFilePath; }
}
