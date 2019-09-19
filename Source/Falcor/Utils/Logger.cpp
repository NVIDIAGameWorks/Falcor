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
#include "stdafx.h"
#include "Logger.h"

namespace Falcor
{
    namespace
    {
        bool sShowErrorBox = true;
        FILE* sLogFile = nullptr;
        Logger::Level sVerbosity = Logger::Level::Warning;

        FILE* openLogFile()
        {
            FILE* pFile = nullptr;

            // Get current process name
            std::string filename = getExecutableName();

            // Now we have a folder and a filename, look for an available filename (we don't overwrite existing files)
            std::string prefix = std::string(filename);
            std::string executableDir = getExecutableDirectory();
            std::string logFile;
            if (findAvailableFilename(prefix, executableDir, "log", logFile))
            {
                pFile = std::fopen(logFile.c_str(), "w");
                if (pFile != nullptr)
                {
                    // Success
                    return pFile;
                }
            }
            // If we got here, we couldn't create a log file
            should_not_get_here();
            return pFile;
        }

        bool init()
        {
            bool b = false;
#if _LOG_ENABLED
            sLogFile = openLogFile();
            b = sLogFile != nullptr;
            assert(b);
#endif
            return b;
        }

        bool sInit = init();
    }

    void Logger::shutdown()
    {
#if _LOG_ENABLED
        if(sLogFile)
        {
            fclose(sLogFile);
            sLogFile = nullptr;
            sInit = false;
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
        default:
            should_not_get_here();
        }
#undef create_level_case
        return c;
    }

    void Logger::log(Level L, const std::string& msg, MsgBox mbox)
    {
#if _LOG_ENABLED
        if(sInit)
        {
            if(L >= sVerbosity)
            {
                std::string s = getLogLevelString(L) + std::string("\t") + msg + "\n";
                std::fprintf(sLogFile, "%s", s.c_str());
                if (isDebuggerPresent())
                {
                    printToDebugWindow(s);
                }
            }
        }
#endif

        bool showMsgBox = false;

        switch (mbox)
        {
        case MsgBox::Auto:
            showMsgBox = (L >= Level::Error) ? sShowErrorBox : false;
            break;
        case MsgBox::Nope:
            showMsgBox = false;
            break;
        case MsgBox::Show:
            showMsgBox = true;
            break;
        default:
            should_not_get_here();
        }

        if (showMsgBox)
        {
            static bool exiting = false;
            bool quit = false;
            if (exiting == false)
            {
                if (isDebuggerPresent())
                {
                    auto res = msgBox(msg + "\n\nPress Abort to quit, Retry to debug or Ignore to continue", MsgBoxType::AbortRetryIgnore);
                    if (res == MsgBoxButton::Retry) debugBreak();
                    if (res == MsgBoxButton::Abort) quit = true;
                }
                else
                {
                    quit = msgBox(msg + "\n\nContinue execution?", MsgBoxType::YesNo) == MsgBoxButton::No;
                }
            }

            if (quit) exit(1); // Don't post quit message. It will cause execution of the current frame to resume, which might crash the app
        }

        if (L >= Level::Fatal) assert(false);   // Assert on errors even without debugger attached
    }

    void Logger::showBoxOnError(bool showBox) { sShowErrorBox = showBox; }
    bool Logger::isBoxShownOnError() { return sShowErrorBox; }
    void Logger::setVerbosity(Level level) { sVerbosity = level; }
}
