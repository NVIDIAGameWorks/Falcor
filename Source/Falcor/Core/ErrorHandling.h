/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include <string>

namespace Falcor
{
    /** Enable/disable showing a message box when reporting an error.
    */
    FALCOR_API void setShowMessageBoxOnError(bool enable);

    /** Return if showing a message box when reporting an error is enabled/disabled.
    */
    FALCOR_API bool getShowMessageBoxOnError();

    /** Report an error by logging it and showing a message box with the option to abort,
        continue or enter the debugger (if one is attached).
        If message boxes are disabled, this will terminate the application after logging the error.
        \param msg Error message.
    */
    FALCOR_API void reportError(const std::string& msg);

    /** Report an error by logging it and showing a message box with the option to abort,
        retry or enter the debugger (if one is attached).
        If message boxes are disabled, this will terminate the application after logging the error.
        \param msg Error message.
    */
    FALCOR_API void reportErrorAndAllowRetry(const std::string& msg);

    /** Report a fatal error by logging it and showing a message box with the option to abort
        or enter the debugger (if one is attached).
        If message boxes are disabled, this will terminate the application after logging the error.
        \param msg Error message.
    */
    [[noreturn]] FALCOR_API void reportFatalError(const std::string& msg);
}
