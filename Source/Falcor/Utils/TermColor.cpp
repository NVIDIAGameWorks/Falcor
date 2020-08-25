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
#include "TermColor.h"

#include <iostream>
#include <unordered_map>

#if _WIN32
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif

namespace Falcor
{
#if _WIN32
    /** The Windows console does not have ANSI support by default,
        but it can be enabled through SetConsoleMode().
        We use static initialization to do so.
    */
    struct EnableVirtualTerminal
    {
        EnableVirtualTerminal()
        {
            auto enableVirtualTerminal = [] (DWORD handle)
            {
                HANDLE console = GetStdHandle(handle);
                if (console == INVALID_HANDLE_VALUE) return;
                DWORD mode;
                GetConsoleMode(console, &mode);
                mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(console, mode);
            };
            enableVirtualTerminal(STD_OUTPUT_HANDLE);
            enableVirtualTerminal(STD_ERROR_HANDLE);
        }
    };

    static EnableVirtualTerminal sEnableVirtualTerminal;
#endif

    static const std::unordered_map<TermColor, std::string> kBeginTag =
    {
        { TermColor::Gray,    "\33[90m" },
        { TermColor::Red,     "\33[91m" },
        { TermColor::Green,   "\33[92m" },
        { TermColor::Yellow,  "\33[93m" },
        { TermColor::Blue,    "\33[94m" },
        { TermColor::Magenta, "\33[95m" }
    };

    static const std::string kEndTag = "\033[0m";

    inline bool isTTY(const std::ostream& stream)
    {
        if (&stream == &std::cout && ISATTY(FILENO(stdout))) return true;
        if (&stream == &std::cerr && ISATTY(FILENO(stderr))) return true;
        return false;
    }

    std::string colored(const std::string& str, TermColor color, const std::ostream& stream)
    {
        return isTTY(stream) ? (kBeginTag.at(color) + str + kEndTag) : str;
    }
}
