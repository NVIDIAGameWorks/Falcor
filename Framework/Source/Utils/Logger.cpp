/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "Logger.h"
#include "Utils/Platform/OS.h"
#include <cstdio>

namespace Falcor
{
#ifdef _DEBUG
    bool Logger::sShowErrorBox = true;
#else
    bool Logger::sShowErrorBox = false;
#endif

    bool Logger::sInit = false;
    FILE* Logger::sLogFile = nullptr;
    Logger::Level Logger::sVerbosity = Logger::Level::Warning;

    static FILE* openLogFile()
    {
        FILE* pFile = nullptr;

        // Get current process name
        std::string filename = getExecutableName();

        // Now we have a folder and a filename, look for an available filename (we don't overwrite existing files)
        std::string prefix = std::string(filename);
        std::string executableDir = getExecutableDirectory();
        std::string logFile;
        if(findAvailableFilename(prefix, executableDir, "log", logFile))
        {
            pFile = std::fopen(logFile.c_str(), "w");
            if(pFile != nullptr)
            {
                // Success
                return pFile;
            }
        }
        // If we got here, we couldn't create a log file
        should_not_get_here();
        return pFile;
    }

    void Logger::init()
    {
#if _LOG_ENABLED
        if(sInit == false)
        {
            sLogFile = openLogFile();
            sInit = sLogFile != nullptr;
            assert(sInit);
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

    void Logger::log(Level L, const std::string& msg, bool forceMsgBox)
    {
#if _LOG_ENABLED
        if(sInit)
        {
            if(L >= sVerbosity)
            {
                std::string s = getLogLevelString(L) + std::string("\t") + msg + "\n";
                std::fprintf(sLogFile, "%s", s.c_str());
                fflush(sLogFile);   // Slows down execution, but ensures that the message will be printed in case of a crash
                if (isDebuggerPresent())
                {
                    printToDebugWindow(s);
                }
            }
        }
#endif

        if(L >= Level::Error)
        {
            if(isDebuggerPresent())
            {
                debugBreak();
            }

            forceMsgBox = sShowErrorBox;
        }

        if (forceMsgBox)
        {
            msgBox(msg);
        }
    }
}