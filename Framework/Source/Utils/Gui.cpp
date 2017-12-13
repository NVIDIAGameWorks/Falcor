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
#include "Gui.h"
#include <sstream>
#include "Utils/Platform/OS.h"
#include "Utils/UserInput.h"
#include "API/RenderContext.h"
#include "Externals/dear_imgui/imgui.h"
#include "Utils/Math/FalcorMath.h"
#include "glm/gtc/type_ptr.hpp"
#include "Utils/StringUtils.h"

#pragma warning (disable : 4756) // overflow in constant arithmetic caused by calculating the setFloat*() functions (when calculating the step and min/max are +/- INF)
namespace Falcor
{
    void Gui::init()
    {
        ImGuiIO& io = ImGui::GetIO();
        io.KeyMap[ImGuiKey_Tab] = (uint32_t)KeyboardEvent::Key::Tab;
        io.KeyMap[ImGuiKey_LeftArrow] = (uint32_t)KeyboardEvent::Key::Left;
        io.KeyMap[ImGuiKey_RightArrow] = (uint32_t)KeyboardEvent::Key::Right;
        io.KeyMap[ImGuiKey_UpArrow] = (uint32_t)KeyboardEvent::Key::Up;
        io.KeyMap[ImGuiKey_DownArrow] = (uint32_t)KeyboardEvent::Key::Down;
        io.KeyMap[ImGuiKey_PageUp] = (uint32_t)KeyboardEvent::Key::PageUp;
        io.KeyMap[ImGuiKey_PageDown] = (uint32_t)KeyboardEvent::Key::PageDown;
        io.KeyMap[ImGuiKey_Home] = (uint32_t)KeyboardEvent::Key::Home;
        io.KeyMap[ImGuiKey_End] = (uint32_t)KeyboardEvent::Key::End;
        io.KeyMap[ImGuiKey_Delete] = (uint32_t)KeyboardEvent::Key::Del;
        io.KeyMap[ImGuiKey_Backspace] = (uint32_t)KeyboardEvent::Key::Backspace;
        io.KeyMap[ImGuiKey_Enter] = (uint32_t)KeyboardEvent::Key::Enter;
        io.KeyMap[ImGuiKey_Escape] = (uint32_t)KeyboardEvent::Key::Escape;
        io.KeyMap[ImGuiKey_A] = (uint32_t)KeyboardEvent::Key::A;
        io.KeyMap[ImGuiKey_C] = (uint32_t)KeyboardEvent::Key::C;
        io.KeyMap[ImGuiKey_V] = (uint32_t)KeyboardEvent::Key::V;
        io.KeyMap[ImGuiKey_X] = (uint32_t)KeyboardEvent::Key::X;
        io.KeyMap[ImGuiKey_Y] = (uint32_t)KeyboardEvent::Key::Y;
        io.KeyMap[ImGuiKey_Z] = (uint32_t)KeyboardEvent::Key::Z;
        io.IniFilename = nullptr;

        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_WindowBg].w = 0.85f;
        style.Colors[ImGuiCol_FrameBg].x *= 0.5f;
        style.Colors[ImGuiCol_FrameBg].y *= 0.5f;
        style.Colors[ImGuiCol_FrameBg].z *= 0.5f;
        style.ScrollbarSize *= 0.7f;

        // Create the pipeline state cache
        mpPipelineState = GraphicsState::create();

        // Create the program
        mpProgram = GraphicsProgram::createFromFile("Framework/Shaders/Gui.vs.slang", "Framework/Shaders/Gui.ps.slang");
        mpProgramVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());
        mpPipelineState->setProgram(mpProgram);

        // Create and set the texture
        uint8_t* pFontData;
        int32_t width, height;
        std::string fontFile;
        if(findFileInDataDirectories("Framework/Fonts/trebucbd.ttf", fontFile))
        {
            io.Fonts->AddFontFromFileTTF(fontFile.c_str(), 14);
        }
        io.Fonts->GetTexDataAsAlpha8(&pFontData, &width, &height);
        Texture::SharedPtr pTexture = Texture::create2D(width, height, ResourceFormat::R8Unorm, 1, 1, pFontData);
        mpProgramVars->setTexture("gFont", pTexture);

        // Create the blend state
        BlendState::Desc blendDesc;
        blendDesc.setRtBlend(0, true).setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha, BlendState::BlendFunc::Zero);
        mpPipelineState->setBlendState(BlendState::create(blendDesc));

        // Create the rasterizer state
        RasterizerState::Desc rsDesc;
        rsDesc.setFillMode(RasterizerState::FillMode::Solid).setCullMode(RasterizerState::CullMode::None).setScissorTest(true).setDepthClamp(false);
        mpPipelineState->setRasterizerState(RasterizerState::create(rsDesc));

        // Create the depth-stencil state
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthFunc(DepthStencilState::Func::Disabled);
        mpPipelineState->setDepthStencilState(DepthStencilState::create(dsDesc));

        // Create the VAO
        VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
        pBufLayout->addElement("POSITION", offsetof(ImDrawVert, pos), ResourceFormat::RG32Float, 1, 0);
        pBufLayout->addElement("TEXCOORD", offsetof(ImDrawVert, uv), ResourceFormat::RG32Float, 1, 1);
        pBufLayout->addElement("COLOR", offsetof(ImDrawVert, col), ResourceFormat::RGBA8Unorm, 1, 2);
        mpLayout = VertexLayout::create();
        mpLayout->addBufferLayout(0, pBufLayout);
    }

    Gui::UniquePtr Gui::create(uint32_t width, uint32_t height)
    {
        UniquePtr pGui = UniquePtr(new Gui);
        pGui->init();
        pGui->onWindowResize(width, height);
        return pGui;
    }

    void Gui::createVao(uint32_t vertexCount, uint32_t indexCount)
    {
        static_assert(sizeof(ImDrawIdx) == sizeof(uint16_t), "ImDrawIdx expected size is a word");
        uint32_t requiredVbSize = vertexCount * sizeof(ImDrawVert);
        uint32_t requiredIbSize = indexCount * sizeof(uint16_t);
        bool createVB = true;
        bool createIB = true;

        if (mpVao)
        {
            createVB = mpVao->getVertexBuffer(0)->getSize() <= requiredVbSize;
            createIB = mpVao->getIndexBuffer()->getSize() <= requiredIbSize;

            if (!createIB && !createVB)
            {
                return;
            }
        }

        // Need to create a new VAO
        std::vector<Buffer::SharedPtr> pVB(1);
        pVB[0] = createVB ? Buffer::create(requiredVbSize + sizeof(ImDrawVert) * 1000, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, nullptr) : mpVao->getVertexBuffer(0);
        Buffer::SharedPtr pIB = createIB ? Buffer::create(requiredIbSize, Buffer::BindFlags::Index, Buffer::CpuAccess::Write, nullptr): mpVao->getIndexBuffer();
        mpVao = Vao::create(Vao::Topology::TriangleList, mpLayout, pVB, pIB, ResourceFormat::R16Uint);
    }
    
    void Gui::onWindowResize(uint32_t width, uint32_t height)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize.x = (float)width;
        io.DisplaySize.y = (float)height;
#ifdef FALCOR_VK
        mpProgramVars["PerFrameCB"]["scale"] = 2.0f / vec2(io.DisplaySize.x, io.DisplaySize.y);
        mpProgramVars["PerFrameCB"]["offset"] = vec2(-1.0f);
#else
        mpProgramVars["PerFrameCB"]["scale"] = 2.0f / vec2(io.DisplaySize.x, -io.DisplaySize.y);
        mpProgramVars["PerFrameCB"]["offset"] = vec2(-1.0f, 1.0f);
#endif
    }

    void Gui::setIoMouseEvents()
    {
        ImGuiIO& io = ImGui::GetIO();
        memcpy(io.MouseDown, mMouseEvents.buttonPressed, sizeof(mMouseEvents.buttonPressed));
    }

    void Gui::resetMouseEvents()
    {
        for (uint32_t i = 0; i < arraysize(mMouseEvents.buttonPressed); i++)
        {
            if (mMouseEvents.buttonReleased[i])
            {
                mMouseEvents.buttonPressed[i] = mMouseEvents.buttonReleased[i] = false;
            }
        }
    }

    void Gui::beginFrame()
    {
        ImGui::NewFrame();
    }

    void Gui::render(RenderContext* pContext, float elapsedTime)
    {
        while (mGroupStackSize)
        {
            endGroup();
        }

        pContext->setGraphicsVars(mpProgramVars);
        // Set the mouse state
        setIoMouseEvents();

        ImGui::Render();
        ImDrawData* pDrawData = ImGui::GetDrawData();
    
        resetMouseEvents();
        // Update the VAO

        createVao(pDrawData->TotalVtxCount, pDrawData->TotalIdxCount);
        mpPipelineState->setVao(mpVao);

        // Upload the data
        ImDrawVert* pVerts = (ImDrawVert*)mpVao->getVertexBuffer(0)->map(Buffer::MapType::WriteDiscard);
        uint16_t* pIndices = (uint16_t*)mpVao->getIndexBuffer()->map(Buffer::MapType::WriteDiscard);

        for (int n = 0; n < pDrawData->CmdListsCount; n++)
        {
            const ImDrawList* pCmdList = pDrawData->CmdLists[n];
            memcpy(pVerts, pCmdList->VtxBuffer.Data, pCmdList->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(pIndices, pCmdList->IdxBuffer.Data, pCmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
            pVerts += pCmdList->VtxBuffer.Size;
            pIndices += pCmdList->IdxBuffer.Size;
        }
        mpVao->getVertexBuffer(0)->unmap();
        mpVao->getIndexBuffer()->unmap();
        mpPipelineState->setFbo(pContext->getGraphicsState()->getFbo());
        pContext->pushGraphicsState(mpPipelineState);

        // Setup viewport
        GraphicsState::Viewport vp;
        vp.originX = 0;
        vp.originY = 0;
        vp.width = ImGui::GetIO().DisplaySize.x;
        vp.height = ImGui::GetIO().DisplaySize.y;
        vp.minDepth = 0;
        vp.maxDepth = 1;
        mpPipelineState->setViewport(0, vp);

        // Render command lists
        uint32_t vtxOffset = 0;
        uint32_t idxOffset = 0;

        for (int n = 0; n < pDrawData->CmdListsCount; n++)
        {
            const ImDrawList* pCmdList = pDrawData->CmdLists[n];
            for (int32_t cmd = 0; cmd < pCmdList->CmdBuffer.Size; cmd++)
            {
                const ImDrawCmd* pCmd = &pCmdList->CmdBuffer[cmd];
                GraphicsState::Scissor scissor((int32_t)pCmd->ClipRect.x, (int32_t)pCmd->ClipRect.y, (int32_t)pCmd->ClipRect.z, (int32_t)pCmd->ClipRect.w);
                mpPipelineState->setScissors(0, scissor);
                pContext->drawIndexed(pCmd->ElemCount, idxOffset, vtxOffset);
                idxOffset += pCmd->ElemCount;
            }
            vtxOffset += pCmdList->VtxBuffer.Size;
        }
 
        // Prepare for the next frame
        ImGuiIO& io = ImGui::GetIO();
        io.DeltaTime = elapsedTime;
        mGroupStackSize = 0;
        pContext->popGraphicsState();
    }

    bool Gui::addCheckBox(const char label[], bool& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::Checkbox(label, &var);
    }

    void Gui::addText(const char text[], bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        ImGui::Text(text, "");
    }

    bool Gui::addButton(const char label[], bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::Button(label);
    }

    bool Gui::addRadioButtons(const RadioButtonGroup& buttons, int32_t& activeID)
    {
        int32_t oldValue = activeID;

        for (const auto& button : buttons)
        {
            if (button.sameLine) ImGui::SameLine();
            ImGui::RadioButton(button.label.c_str(), &activeID, button.buttonID);
        }

        return oldValue != activeID;
    }

    bool Gui::beginGroup(const char name[], bool beginExpanded)
    {
        ImGuiTreeNodeFlags flags = beginExpanded ? ImGuiTreeNodeFlags_DefaultOpen :  0;
        bool visible = mGroupStackSize ? ImGui::TreeNode(name) : ImGui::CollapsingHeader(name, flags);
        if (visible)
        {
            mGroupStackSize++;
        }
        return visible;
    }

    void Gui::endGroup()
    {
        assert(mGroupStackSize >= 1);
        mGroupStackSize--;
        if (mGroupStackSize)
        {
            ImGui::TreePop();
        }
    }

    bool Gui::addFloatVar(const char label[], float& var, float minVal, float maxVal, float step, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragFloat(label, &var, step);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addFloat2Var(const char label[], glm::vec2& var, float minVal, float maxVal, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        float speed = min(1.0f, (maxVal - minVal) * 0.01f);
        bool b = ImGui::DragFloat2(label, glm::value_ptr(var), speed, minVal, maxVal);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addFloat3Var(const char label[], glm::vec3& var, float minVal, float maxVal, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        float speed = min(1.0f, (maxVal - minVal) * 0.01f);
        bool b = ImGui::DragFloat3(label, glm::value_ptr(var), speed, minVal, maxVal);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addFloat4Var(const char label[], glm::vec4& var, float minVal, float maxVal, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        float speed = min(1.0f, (maxVal - minVal) * 0.01f);
        bool b = ImGui::DragFloat4(label, glm::value_ptr(var), speed, maxVal, minVal);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addIntVar(const char label[], int32_t& var, int minVal, int maxVal, int step, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::InputInt(label, &var, step);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addRgbColor(const char label[], glm::vec3& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::ColorEdit3(label, glm::value_ptr(var));
    }

    bool Gui::addRgbaColor(const char label[], glm::vec4& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::ColorEdit4(label, glm::value_ptr(var));
    }

    bool Gui::onKeyboardEvent(const KeyboardEvent& event)
    {
        ImGuiIO& io = ImGui::GetIO();

        if (event.type == KeyboardEvent::Type::Input)
        {
            std::string u8str = utf32ToUtf8(event.codepoint);
            io.AddInputCharactersUTF8(u8str.c_str());

            // Gui consumes keyboard input
            return true;
        }
        else
        {
            uint32_t key = (uint32_t)(event.key == KeyboardEvent::Key::KeypadEnter ? KeyboardEvent::Key::Enter : event.key);

            switch (event.type)
            {
            case KeyboardEvent::Type::KeyPressed:
                io.KeysDown[key] = true;
                break;
            case KeyboardEvent::Type::KeyReleased:
                io.KeysDown[key] = false;
                break;
            default:
                should_not_get_here();
            }

            io.KeyCtrl = event.mods.isCtrlDown;
            io.KeyAlt = event.mods.isAltDown;
            io.KeyShift = event.mods.isShiftDown;
            io.KeySuper = false;
            return io.WantCaptureKeyboard;
        }
    }

    bool Gui::onMouseEvent(const MouseEvent& event)
    {
        ImGuiIO& io = ImGui::GetIO();
        switch (event.type)
        {
        case MouseEvent::Type::LeftButtonDown:
            mMouseEvents.buttonPressed[0] = true;
            break;
        case MouseEvent::Type::LeftButtonUp:
            mMouseEvents.buttonReleased[0] = true;
            break;
        case MouseEvent::Type::RightButtonDown:
            mMouseEvents.buttonPressed[1] = true;
            break;
        case MouseEvent::Type::RightButtonUp:
            mMouseEvents.buttonReleased[1] = true;
            break;
        case MouseEvent::Type::MiddleButtonDown:
            mMouseEvents.buttonPressed[2] = true;
            break;
        case MouseEvent::Type::MiddleButtonUp:
            mMouseEvents.buttonReleased[2] = true;
            break;
        case MouseEvent::Type::Move:
            io.MousePos.x = event.pos.x * io.DisplaySize.x;
            io.MousePos.y = event.pos.y * io.DisplaySize.y;
            break;
        case MouseEvent::Type::Wheel:
            io.MouseWheel += event.wheelDelta.y;
            break;
        }

        return io.WantCaptureMouse;
    }

    void Gui::pushWindow(const char label[], uint32_t width, uint32_t height, uint32_t x, uint32_t y, bool showTitleBar)
    {
        ImVec2 pos{ float(x), float(y) };
        ImVec2 size{ float(width), float(height) };
        ImGui::SetNextWindowSize(size, ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowPos(pos, ImGuiSetCond_FirstUseEver);
        int flags = 0;
        if (!showTitleBar)
        {
            flags |= ImGuiWindowFlags_NoTitleBar;
        }
        ImGui::Begin(label, nullptr, flags);
    }

    void Gui::popWindow()
    {
        ImGui::End();
    }

    void Gui::addSeparator()
    {
        ImGui::Separator();
    }

    bool Gui::addDropdown(const char label[], const DropdownList& values, uint32_t& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        // Check if we need to update the currentItem
        const auto& iter = mDropDownValues.find(label);
        int curItem;
        if ((iter == mDropDownValues.end()) || (iter->second.lastVal != var))
        {
            // Search the current val
            for (uint32_t i = 0 ; i < values.size() ; i++)
            {
                if (values[i].value == var)
                {
                    curItem = i;
                    mDropDownValues[label].currentItem = i;
                    break;
                }
            }
        }
        else
        {
            curItem = mDropDownValues[label].currentItem;
        }

        std::string comboStr;
        for (const auto& v : values)
        {
            comboStr += v.label + '\0';
        }
        comboStr += '\0';
        uint32_t prevItem = curItem;
        //This returns true if the combo is interacted with at all
        bool b = ImGui::Combo(label, &curItem, comboStr.c_str());
        mDropDownValues[label].currentItem = curItem;
        mDropDownValues[label].lastVal = values[curItem].value;
        var = values[curItem].value;
        //Only return true if value is changed
        return b && prevItem != curItem;
    }

    void Gui::setGlobalFontScaling(float scale)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.FontGlobalScale = scale;
    }

    bool Gui::addTextBox(const char label[], char buf[], size_t bufSize, uint32_t lineCount)
    {
        const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;

        if (lineCount > 1)
        {
            return ImGui::InputTextMultiline(label, buf, bufSize, ImVec2(-1.0f, ImGui::GetTextLineHeight() * lineCount), flags);
        }
        else
        {
            return ImGui::InputText(label, buf, bufSize, flags);
        }
    }

    bool Gui::addTextBox(const char label[], std::string& text, uint32_t lineCount /*= 1*/)
    {
        const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;

        static const int maxSize = 2048;
        char buf[maxSize];
        copyStringToBuffer(buf, maxSize, text);

        if (lineCount > 1)
        {
            return ImGui::InputTextMultiline(label, buf, maxSize, ImVec2(-1.0f, ImGui::GetTextLineHeight() * lineCount), flags);
        }
        else
        {
            return ImGui::InputText(label, buf, maxSize, flags);
        }
    }

    void Gui::addTooltip(const char tip[], bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(450.0f);
            ImGui::TextUnformatted(tip);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void Gui::addGraph(const char label[], GraphCallback func, void* pUserData, uint32_t sampleCount, int32_t sampleOffset, float yMin, float yMax, uint32_t width, uint32_t height)
    {
        ImVec2 imSize{ (float)width, (float)height };
        ImGui::PlotLines(label, func, pUserData, (int32_t)sampleCount, sampleOffset, nullptr, yMin, yMax, imSize);
    }

    bool Gui::addDirectionWidget(const char label[], glm::vec3& direction)
    {
        glm::vec3 dir = direction;
        bool b = addFloat3Var(label, dir, -1, 1);
        direction = glm::normalize(dir);
        return b;
    }
}
