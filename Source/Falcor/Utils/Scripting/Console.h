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
#include "Core/Macros.h"
#include <optional>
#include <string>
#include <vector>

// Forward declaration.
struct ImGuiInputTextCallbackData;

namespace Falcor
{
    class Gui;

    class FALCOR_API Console
    {
    public:
        /** Clears the console.
        */
        void clear();

        /** Renders the console and handles important keyboard input events:
            - The "`" key is used to open/close the console.
            - The ESC key is used to close the console if currently open.
            - The UP/DOWN keys are used to browse through the history.
            \param[in] pGui GUI.
            \param[in,out] show Flag to indicate if console is shown.
        */
        void render(Gui* pGui, bool& show);

        /** Processes console input. Should be called once at the end of every frame.
            \return Returns true if some processing occured.
        */
        bool flush();

        /** Global console instance.
            \return Returns the global console instance.
        */
        static Console& instance();

    private:
        Console() = default;

        void enterCommand();
        std::optional<std::string> browseHistory(bool upOrDown);
        static int inputTextCallback(ImGuiInputTextCallbackData* data);

        std::string mLog;
        char mCmdBuffer[2048] = {};
        std::string mCmdPending;
        std::vector<std::string> mHistory;
        int32_t mHistoryIndex = -1;
        bool mScrollToBottom = true;
    };
}
