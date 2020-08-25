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
#include "Console.h"
#include "dear_imgui/imgui.h"

namespace Falcor
{
    namespace
    {
        static const ImVec4 kBackgroundColor = ImVec4(0.f, 0.f, 0.f, 0.8f);
        static const uint32_t kLineCount = 20;

        class ConsoleWindow
        {
        public:
            ConsoleWindow(Gui* pGui)
            {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.f, 4.f));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
                ImGui::PushStyleColor(ImGuiCol_WindowBg, kBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, kBackgroundColor);
                height = (float)ImGui::GetTextLineHeight() * kLineCount;
                ImGui::SetNextWindowSize({ ImGui::GetIO().DisplaySize.x, 0 }, ImGuiCond_Always);
                ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);

                ImGui::Begin("##Console", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.f, 0.f));
                ImGui::PushFont(pGui->getFont("monospace"));
            }

            ~ConsoleWindow()
            {
                ImGui::PopFont();
                ImGui::PopStyleVar();
                ImGui::End();
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar(2);
            }

            float height = 0;
        };
    }

    void Console::clear()
    {
        mLog.clear();
    }

    void Console::render(Gui* pGui, bool& show)
    {
        // Toggle console with "`" key.
        if (ImGui::IsKeyPressed('`')) show = !show;

        // Allow closing console with escape key.
        if (show && ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) show = false;

        if (!show) return;

        ConsoleWindow w(pGui);

        ImGui::BeginChild("log", {0, w.height - ImGui::GetTextLineHeight() - 5});
        ImGui::PushTextWrapPos();
        ImGui::TextUnformatted(mLog.c_str());
        ImGui::PopTextWrapPos();
        if (mScrollToBottom)
        {
            ImGui::SetScrollHere(1.0f);
            mScrollToBottom = false;
        }
        ImGui::EndChild();

        ImGui::PushItemWidth(ImGui::GetWindowWidth());
        if (ImGui::InputText("##console", mCmdBuffer, arraysize(mCmdBuffer), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory | ImGuiInputTextFlags_CallbackCharFilter, &inputTextCallback, this))
        {
            enterCommand();
            ImGui::GetIO().KeysDown[(uint32_t)KeyboardEvent::Key::Enter] = false;
        }
        // Stick focus to console text input.
        ImGui::SetWindowFocus();
        ImGui::SetKeyboardFocusHere();
        pGui->setActiveFont("");
    }

    bool Console::flush()
    {
        if (mCmdPending.empty()) return false;

        try
        {
            mLog += Scripting::interpretScript(mCmdPending);
        }
        catch (const std::exception& e)
        {
            mLog += std::string(e.what()) + "\n";
        };

        mCmdPending.clear();

        return true;
    }

    Console& Console::instance()
    {
        static std::unique_ptr<Console> pInstance;
        if (!pInstance) pInstance = std::unique_ptr<Console>(new Console());
        return *pInstance;
    }

    void Console::enterCommand()
    {
        auto cmd = std::string(mCmdBuffer);
        mCmdBuffer[0] = '\0';
        mLog += cmd + "\n";
        mScrollToBottom = true;

        // Push command to history.
        if (!cmd.empty() && (mHistory.empty() || cmd != mHistory.back())) mHistory.push_back(cmd);
        mHistoryIndex = -1;

        std::swap(cmd, mCmdPending);
    }

    std::optional<std::string> Console::browseHistory(bool upOrDown)
    {
        if (upOrDown)
        {
            if (mHistoryIndex + 1 < (int32_t)mHistory.size())
            {
                mHistoryIndex++;
                return mHistory[mHistory.size() - mHistoryIndex - 1];
            }
        }
        else
        {
            if (mHistoryIndex >= 0)
            {
                mHistoryIndex--;
                return mHistoryIndex >= 0 ? mHistory[mHistory.size() - mHistoryIndex - 1] : "";
            }
        }
        return {};
    }

    int Console::inputTextCallback(ImGuiInputTextCallbackData* data)
    {
        assert(data->UserData != nullptr);
        Console& console = *static_cast<Console*>(data->UserData);

        if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter)
        {
            if (data->EventChar == '`') return 1;
        }
        if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory)
        {
            if (auto cmd = console.browseHistory(data->EventKey == ImGuiKey_UpArrow))
            {
                strncpy(data->Buf, cmd->c_str(), cmd->length() + 1);
                data->BufTextLen = data->CursorPos = (int)cmd->length();
                data->BufDirty = true;
            }
        }
        return 0;
    }

    SCRIPT_BINDING(Console)
    {
        auto cls = []() { Console::instance().clear(); };
        m.def("cls", cls);
    }
}
