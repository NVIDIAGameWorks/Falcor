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
#include "Console.h"
#include "dear_imgui/imgui.h"

namespace Falcor
{
    namespace
    {
        static const uint32_t kLineCount = 16;
        class GuiWindow
        {
        public:
            GuiWindow(Gui* pGui) : mpGui(pGui)
            {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                height = (float)ImGui::GetTextLineHeight() * kLineCount;
                ImGui::SetNextWindowSize({ ImGui::GetIO().DisplaySize.x, 0 }, ImGuiCond_Always);
                ImGui::SetNextWindowPos({0, ImGui::GetIO().DisplaySize.y - height}, ImGuiCond_Always);
                
                ImGui::Begin("##Console", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
                ImGui::PushFont(pGui->getFont("monospace"));
            }

            ~GuiWindow()
            {
                ImGui::PopFont();
                ImGui::PopStyleVar();
                ImGui::End();
                ImGui::PopStyleVar();
            }

            float height = 0;
        private:
            Gui* mpGui;
        };

        std::string sLog;
        char sCmd[2048] = {};
        bool sFlush = false;
        bool scrollToBottom = true;

        SCRIPT_BINDING(Console)
        {
            auto cls = []() {sLog = {}; };
            m.func_("cls", cls);
        }
    }

    bool Console::flush()
    {
        if (!sFlush) return false;
        std::string cmd(sCmd); // We need to use a temporary copy so that we could reset `sCmd`, otherwise we will end up with an endless loop
        sCmd[0] = 0;
        sFlush = false;

        try
        {
            sLog += Scripting::runScript(cmd);
        }
        catch (std::exception e)
        {
            sLog += std::string(e.what()) + "\n";
        };
        return true;
    }

    void Console::render(Gui* pGui)
    {
        GuiWindow w(pGui);

        ImGui::BeginChild("log", {0, w.height - ImGui::GetTextLineHeight() - 5 });
        ImGui::TextUnformatted(sLog.c_str());
        if(scrollToBottom)
        {
            ImGui::SetScrollHere(1.0f);
            scrollToBottom = false;
        }
        ImGui::EndChild();

        ImGui::PushItemWidth(ImGui::GetWindowWidth());
        if(ImGui::InputText("##console", sCmd, arraysize(sCmd), ImGuiInputTextFlags_EnterReturnsTrue))
        {
            sFlush = true;
            sLog += std::string(sCmd) + "\n";
            scrollToBottom = true;
            ImGui::SetKeyboardFocusHere();
            ImGui::GetIO().KeysDown[(uint32_t)KeyboardEvent::Key::Enter] = false;
        }
        if(ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        pGui->setActiveFont("");
    }
}
