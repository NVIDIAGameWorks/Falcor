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
#include "ErrorHandling.h"
#include "Platform/OS.h"
#include "Utils/Logger.h"

namespace Falcor
{
    static bool sShowMessageBoxOnError = true;

    void setShowMessageBoxOnError(bool enable) { sShowMessageBoxOnError = enable; }
    bool getShowMessageBoxOnError() { return sShowMessageBoxOnError;  }

    void reportError(const std::string& msg)
    {
        logError(msg);

        if (sShowMessageBoxOnError)
        {
            enum ButtonId {
                Continue,
                Debug,
                Abort
            };

            // Setup message box buttons
            std::vector<MsgBoxCustomButton> buttons;
            buttons.push_back({ Continue, "Continue" });
            if (isDebuggerPresent()) buttons.push_back({ Debug, "Debug" });
            buttons.push_back({ Abort, "Abort" });

            // Show message box
            auto result = msgBox(msg, buttons, MsgBoxIcon::Error);
            if (result == Continue) return;
            if (result == Debug) debugBreak();
        }

        std::quick_exit(1);
    }

    void reportErrorAndAllowRetry(const std::string& msg)
    {
        logError(msg);

        if (sShowMessageBoxOnError)
        {
            enum ButtonId {
                Retry,
                Debug,
                Abort
            };

            // Setup message box buttons
            std::vector<MsgBoxCustomButton> buttons;
            buttons.push_back({ Retry, "Retry" });
            if (isDebuggerPresent()) buttons.push_back({ Debug, "Debug" });
            buttons.push_back({ Abort, "Abort" });

            // Show message box
            auto result = msgBox(msg, buttons, MsgBoxIcon::Error);
            if (result == Retry) return;
            if (result == Debug) debugBreak();
        }

        std::quick_exit(1);
    }

    [[noreturn]] void reportFatalError(const std::string& msg)
    {
        logFatal(msg);

        if (sShowMessageBoxOnError)
        {
            enum ButtonId {
                Debug,
                Abort
            };

            // Setup message box buttons
            std::vector<MsgBoxCustomButton> buttons;
            if (isDebuggerPresent()) buttons.push_back({ Debug, "Debug" });
            buttons.push_back({ Abort, "Abort" });

            // Show message box
            auto result = msgBox(msg, buttons, MsgBoxIcon::Error);
            if (result == Debug) debugBreak();
        }

        std::quick_exit(1);
    }
}
