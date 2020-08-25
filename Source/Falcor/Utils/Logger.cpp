/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
        std::string sLogFilePath;
        bool sLogToConsole = false;
        bool sShowBoxOnError = true;
        Logger::Level sVerbosity = Logger::Level::Info;

#if _LOG_ENABLED
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
#if _LOG_ENABLED
        if(sLogFile)
        {
            fclose(sLogFile);
            sLogFile = nullptr;
            sInitialized = false;
        }
#endif
    }

    const char* getLogLevelString(Logger::Level L)
    {
        const char* c = nullptr;
#define create_level_case(_l) case _l: c = "(" #_l ")" ;break;
        switch(L)
        {
            create_level_case(Logger::Level::Info);
            create_level_case(Logger::Level::Warning);
            create_level_case(Logger::Level::Error);
            create_level_case(Logger::Level::Fatal);
        default:
            should_not_get_here();
        }
#undef create_level_case
        return c;
    }

    void Logger::log(Level L, const std::string& msg, MsgBox mbox, bool terminateOnError)
    {
#if _LOG_ENABLED
        if (L >= sVerbosity)
        {
            std::string s = getLogLevelString(L) + std::string("\t") + msg + "\n";

            // Write to log file.
            printToLogFile(s);

            // Write to debug window if debugger is attached.
            if (isDebuggerPresent()) printToDebugWindow(s);

            // Write errors to stderr unconditionally, other messages to stdout if enabled.
            if (L < Logger::Level::Error)
            {
                if (sLogToConsole) std::cout << s;
            }
            else
            {
                std::cerr << s;
            }
        }
#endif

        if (sShowBoxOnError)
        {
            if (mbox == MsgBox::Auto)
            {
                mbox = (L >= Level::Error) ? MsgBox::ContinueAbort : MsgBox::None;
            }

            if (mbox != MsgBox::None)
            {
                enum ButtonId {
                    ContinueOrRetry,
                    Debug,
                    Abort
                };

                // Setup message box buttons
                std::vector<MsgBoxCustomButton> buttons;
                if (L != Level::Fatal) buttons.push_back({ContinueOrRetry, mbox == MsgBox::ContinueAbort ? "Continue" : "Retry"});
                if (isDebuggerPresent()) buttons.push_back({Debug, "Debug"});
                buttons.push_back({Abort, "Abort"});

                // Setup icon
                MsgBoxIcon icon = MsgBoxIcon::Info;
                if (L == Level::Warning) icon = MsgBoxIcon::Warning;
                else if (L >= Level::Error) icon = MsgBoxIcon::Error;

                // Show message box
                auto result = msgBox(msg, buttons, icon);
                if (result == Debug) debugBreak();
                else if (result == Abort) exit(1);
            }
        }

        // Terminate on errors if not displaying message box and terminateOnError is enabled
        if (L == Level::Error && !sShowBoxOnError && terminateOnError) exit(1);

        // Always terminate on fatal errors
        if (L == Level::Fatal) exit(1);
    }

    bool Logger::setLogFilePath(const std::string& path)
    {
#if _LOG_ENABLED
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

    const std::string& Logger::getLogFilePath() { return sLogFilePath; }
    void Logger::logToConsole(bool enable) { sLogToConsole = enable; }
    bool Logger::shouldLogToConsole() { return sLogToConsole; }
    void Logger::showBoxOnError(bool showBox) { sShowBoxOnError = showBox; }
    bool Logger::isBoxShownOnError() { return sShowBoxOnError; }
    void Logger::setVerbosity(Level level) { sVerbosity = level; }
}
