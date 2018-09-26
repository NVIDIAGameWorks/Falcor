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
    void Gui::init(float scaleFactor)
    {
        mScaleFactor = scaleFactor;
        ImGui::CreateContext();
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
        style.Colors[ImGuiCol_WindowBg].w = 0.9f;
        style.Colors[ImGuiCol_FrameBg].x *= 0.1f;
        style.Colors[ImGuiCol_FrameBg].y *= 0.1f;
        style.Colors[ImGuiCol_FrameBg].z *= 0.1f;
        style.ScrollbarSize *= 0.7f;

        style.Colors[ImGuiCol_MenuBarBg] = style.Colors[ImGuiCol_WindowBg];
        style.ScaleAllSizes(scaleFactor);

        // Create the pipeline state cache
        mpPipelineState = GraphicsState::create();

        // Create the program
        mpProgram = GraphicsProgram::createFromFile("Framework/Shaders/Gui.slang", "vs", "ps");
        mpProgramVars = GraphicsVars::create(mpProgram->getReflector());
        mpPipelineState->setProgram(mpProgram);

        // Add the default font
        addFont("", "Framework/Fonts/trebucbd.ttf");
        setActiveFont("");

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

        mGuiImageLoc = mpProgram->getReflector()->getDefaultParameterBlock()->getResourceBinding("guiImage");
    }

    Gui::~Gui()
    {
        ImGui::DestroyContext();
    }

    Gui::UniquePtr Gui::create(uint32_t width, uint32_t height, float scaleFactor)
    {
        UniquePtr pGui = UniquePtr(new Gui);
        pGui->init(scaleFactor);
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
                if (pCmd->TextureId) 
                {
                    mpProgramVars->getDefaultBlock()->setSrv(mGuiImageLoc, 0, (mpImages[reinterpret_cast<size_t>(pCmd->TextureId) - 1])->getSRV());
                    mpProgramVars["PerFrameCB"]["useGuiImage"] = true;
                }
                else
                {
                    mpProgramVars["PerFrameCB"]["useGuiImage"] = false;
                }
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

        mpImages.clear();
    }

    glm::vec4 Gui::pickUniqueColor(const std::string& key)
    {
        union hashedValue
        {
            size_t st;
            int32_t i32[2];
        };
        hashedValue color;
        color.st = std::hash<std::string>()(key);

        return glm::vec4(color.i32[0] % 1000 / 2000.0f, color.i32[1] % 1000 / 2000.0f, (color.i32[0] * color.i32[1]) % 1000 / 2000.0f, 1.0f);
    }

    void Gui::addDummyItem(const char label[], const glm::vec2& size, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        ImGui::PushID(label);
        ImGui::Dummy({ size.x, size.y });
        ImGui::PopID();
    }

    void Gui::addRect(const glm::vec2& size, const glm::vec4& color, bool filled, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        
        const ImVec2& cursorPos = ImGui::GetCursorScreenPos();
        ImVec2 bottomLeft{ cursorPos.x + size.x, cursorPos.y + size.y };
        ImVec4 rectColor{color.x, color.y, color.z, color.w};
        
        if (filled)
        {
            ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetCursorScreenPos(), bottomLeft, ImGui::ColorConvertFloat4ToU32(rectColor));
        }
        else
        {
            ImGui::GetWindowDrawList()->AddRect(ImGui::GetCursorScreenPos(), bottomLeft, ImGui::ColorConvertFloat4ToU32(rectColor));
        }
    }

    void Gui::beginColumns(uint32_t numColumns)
    {
        ImGui::Columns(numColumns);
    }

    void Gui::nextColumn()
    {
        ImGui::NextColumn();
    }

    bool Gui::addCheckBox(const char label[], bool& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::Checkbox(label, &var);
    }

    bool Gui::addCheckBox(const char label[], int& var, bool sameLine)
    {
        bool value = (var != 0);
        bool modified = addCheckBox(label, value, sameLine);
        var = (value ? 1 : 0);
        return modified;
    }

    bool Gui::addCheckboxes(const char label[], bool* pData, uint32_t numCheckboxes, bool sameLine)
    {
        bool modified = false;
        std::string labelString(std::string("##") + label + '0');

        for (uint32_t i = 0; i < numCheckboxes - 1; ++i)
        {
            labelString[labelString.size() - 1] = '0' + static_cast<int32_t>(i);
            modified |= addCheckBox(labelString.c_str(), pData[i], (!i) ? sameLine : true);
        }

        addCheckBox(label, pData[numCheckboxes - 1], true );

        return modified;
    }

    bool Gui::addBool2Var(const char label[], glm::bvec2& var, bool sameLine)
    {
        return addCheckboxes(label, glm::value_ptr(var), 2, sameLine);
    }

    bool Gui::addBool3Var(const char label[], glm::bvec3& var, bool sameLine)
    {
        return addCheckboxes(label, glm::value_ptr(var), 3, sameLine);
    }

    bool Gui::addBool4Var(const char label[], glm::bvec4& var, bool sameLine)
    {
        return addCheckboxes(label, glm::value_ptr(var), 4, sameLine);
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

    bool Gui::addFloatVar(const char label[], float& var, float minVal, float maxVal, float step, bool sameLine, const char* displayFormat)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragFloat(label, &var, step, minVal, maxVal, displayFormat);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addFloat2Var(const char label[], glm::vec2& var, float minVal, float maxVal, float step, bool sameLine, const char* displayFormat)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragFloat2(label, glm::value_ptr(var), step, minVal, maxVal, displayFormat);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addFloat3Var(const char label[], glm::vec3& var, float minVal, float maxVal, float step, bool sameLine, const char* displayFormat)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragFloat3(label, glm::value_ptr(var), step, minVal, maxVal, displayFormat);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addFloat4Var(const char label[], glm::vec4& var, float minVal, float maxVal, float step, bool sameLine, const char* displayFormat)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragFloat4(label, glm::value_ptr(var), step, minVal, maxVal, displayFormat);
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

    bool Gui::addInt2Var(const char label[], glm::ivec2& var, int32_t minVal, int32_t maxVal, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::InputInt2(label, static_cast<int*>(glm::value_ptr(var)), 0);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addInt3Var(const char label[], glm::ivec3& var, int32_t minVal, int32_t maxVal, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::InputInt3(label, static_cast<int*>(glm::value_ptr(var)), 0);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    bool Gui::addInt4Var(const char label[], glm::ivec4& var, int32_t minVal, int32_t maxVal, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::InputInt4(label, static_cast<int*>(glm::value_ptr(var)), 0);
        var = clamp(var, minVal, maxVal);
        return b;
    }

    template <>
    bool Gui::addFloatVecVar<glm::vec2>(const char label[], glm::vec2& var, float minVal, float maxVal, float step, bool sameLine)
    {
        return addFloat2Var(label, var, minVal, maxVal, step, sameLine);
    }

    template <>
    bool Gui::addFloatVecVar<glm::vec3>(const char label[], glm::vec3& var, float minVal, float maxVal, float step, bool sameLine)
    {
        return addFloat3Var(label, var, minVal, maxVal, step, sameLine);
    }

    template <>
    bool Gui::addFloatVecVar<glm::vec4>(const char label[], glm::vec4& var, float minVal, float maxVal, float step, bool sameLine)
    {
        return addFloat4Var(label, var, minVal, maxVal, step, sameLine);
    }

#define add_matrix_var(TypeName) template bool Gui::addMatrixVar(const char label[], TypeName& var, float minVal , float maxVal, bool sameLine)

    add_matrix_var(glm::mat2x2);
    add_matrix_var(glm::mat2x3);
    add_matrix_var(glm::mat2x4);
    add_matrix_var(glm::mat3x2);
    add_matrix_var(glm::mat3x3);
    add_matrix_var(glm::mat3x4);
    add_matrix_var(glm::mat4x2);
    add_matrix_var(glm::mat4x3);
    add_matrix_var(glm::mat4x4);

#undef add_matrix_var


    template<typename MatrixType>
    bool Gui::addMatrixVar(const char label[], MatrixType& var, float minVal, float maxVal, bool sameLine)
    {
        std::string labelString(label);
        std::string hiddenLabelString("##");
        hiddenLabelString += labelString + "[0]";
        
        ImVec2 topLeft = ImGui::GetCursorScreenPos();
        ImVec2 bottomRight;
        
        bool b = false;
        
        for (uint32_t i = 0; i < static_cast<uint32_t>(var.length()); ++i)
        {
            std::string& stringToDisplay = hiddenLabelString;
            hiddenLabelString[hiddenLabelString.size() - 2] = '0' + static_cast<int32_t>(i);
            if (i == var.length() - 1)
            {
               stringToDisplay = labelString;
            }

            b |= addFloatVecVar<typename MatrixType::col_type>(stringToDisplay.c_str(), var[i], minVal, maxVal, 0.001f, sameLine);
            
            if (i == 0)
            {
                ImGui::SameLine();
                bottomRight = ImGui::GetCursorScreenPos();
                float oldSpacing = ImGui::GetStyle().ItemSpacing.y;
                ImGui::GetStyle().ItemSpacing.y = 0.0f;
                ImGui::Dummy({});
                ImGui::Dummy({});
                ImGui::GetStyle().ItemSpacing.y = oldSpacing;
                ImVec2 correctedCursorPos = ImGui::GetCursorScreenPos();
                correctedCursorPos.y += oldSpacing;
                ImGui::SetCursorScreenPos(correctedCursorPos);
                bottomRight.y = ImGui::GetCursorScreenPos().y;
            }
            else if(i == 1)
            {
                bottomRight.y = topLeft.y + (bottomRight.y - topLeft.y) * (var.length());
                bottomRight.x -= ImGui::GetStyle().ItemInnerSpacing.x * 3 - 1;
                bottomRight.y -= ImGui::GetStyle().ItemInnerSpacing.y  - 1;
                topLeft.x -= 1; topLeft.y -= 1;
                auto colorVec4 =ImGui::GetStyleColorVec4(ImGuiCol_ScrollbarGrab); colorVec4.w *= 0.25f;
                ImU32 color = ImGui::ColorConvertFloat4ToU32(colorVec4);
                ImGui::GetWindowDrawList()->AddRect(topLeft, bottomRight, color);
            }
        }
        return b;
    }

    bool Gui::beginMainMenuBar()
    {
        return ImGui::BeginMainMenuBar();
    }

    bool Gui::beginDropDownMenu(const char label[])
    {
        return ImGui::BeginMenu(label);
    }

    void Gui::endDropDownMenu()
    {
        ImGui::EndMenu();
    }

    bool Gui::addMenuItem(const char label[])
    {
        return ImGui::MenuItem(label);
    }

    void Gui::endMainMenuBar()
    {
        ImGui::EndMainMenuBar();
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

    bool Gui::dragDropSource(const char label[], const char dataLabel[], const std::string& payloadString)
    {
        if (ImGui::IsItemHovered() && (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1))) ImGui::SetWindowFocus();
        if (!(ImGui::IsWindowFocused())) return false;
        bool b = ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID);
        if (b)
        {
            ImGui::SetDragDropPayload(dataLabel, payloadString.data(), payloadString.size() * sizeof(payloadString[0]), ImGuiCond_Once);
            ImGui::EndDragDropSource();
        }
        return b;
    }

    bool Gui::dragDropDest(const char dataLabel[], std::string& payloadString)
    {
        bool b = false;
        if (ImGui::BeginDragDropTarget())
        {
            auto dragDropPayload = ImGui::AcceptDragDropPayload(dataLabel);
            b = dragDropPayload && dragDropPayload->IsDataType(dataLabel) && (dragDropPayload->Data != nullptr);
            if (b)
            {
                payloadString.resize(dragDropPayload->DataSize);
                std::memcpy(&payloadString.front(), dragDropPayload->Data, dragDropPayload->DataSize);
            }

            ImGui::EndDragDropTarget();
        }
        
        return b;
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

    void Gui::addImage(const char label[], const Texture::SharedPtr& pTex, glm::vec2 size, bool maintainRatio, bool sameLine)
    {
        if (size == vec2(0)) size = getCurrentWindowSize();

        ImGui::PushID(label);
        if (sameLine) ImGui::SameLine();
        mpImages.push_back(pTex);
        float aspectRatio = maintainRatio ? (static_cast<float>(pTex->getHeight()) / static_cast<float>(pTex->getWidth())) : 1.0f;
        ImGui::Image(reinterpret_cast<ImTextureID>(mpImages.size()), { size.x, maintainRatio ? size.x  * aspectRatio : size.y });
        ImGui::PopID();
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

    void Gui::pushWindow(const char label[], uint32_t width, uint32_t height, uint32_t x, uint32_t y, bool showTitleBar, bool allowMove)
    {
        ImVec2 pos{ float(x), float(y) };
        ImVec2 size{ float(width), float(height) };
        ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver);
        int flags = 0;
        if (!showTitleBar)
        {
            flags |= ImGuiWindowFlags_NoTitleBar;
        }
        if (!allowMove)
        {
            flags |= ImGuiWindowFlags_NoMove;
        }

        ImGui::Begin(label, nullptr, flags);
        ImGui::PushFont(mpActiveFont);
    }

    void Gui::popWindow()
    {
        ImGui::PopFont();
        ImGui::End();
    }

    void Gui::setCurrentWindowPos(uint32_t x, uint32_t y)
    {
        ImGui::SetWindowPos({ static_cast<float>(x), static_cast<float>(y) });
    }

    glm::vec2 Gui::getCurrentWindowPos()
    {
        ImVec2 windowPos = ImGui::GetWindowPos();
        return { windowPos.x, windowPos.y };
    }

    void Gui::setCurrentWindowSize(uint32_t width, uint32_t height)
    {
        ImGui::SetWindowSize({ static_cast<float>(width), static_cast<float>(height) });
    }

    glm::vec2 Gui::getCurrentWindowSize()
    {
        ImVec2 windowSize = ImGui::GetWindowSize();
        return {windowSize.x, windowSize.y};
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

    void Gui::setGlobalGuiScaling(float scale)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.FontGlobalScale = scale;
        ImGui::GetStyle().ScaleAllSizes(scale);
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

        bool result = false;
        if (lineCount > 1)
        {
            result = ImGui::InputTextMultiline(label, buf, maxSize, ImVec2(-1.0f, ImGui::GetTextLineHeight() * lineCount), flags);
        }
        else
        {
            result = ImGui::InputText(label, buf, maxSize, flags);
        }

        text = std::string(buf);
        return result;
    }

    bool Gui::addMultiTextBox(const char label[], const std::vector<std::string>& textLabels, std::vector<std::string>& textEntries)
    {
        static uint32_t sIdOffset = 0;
        bool result = false;

        for (uint32_t i = 0; i < textEntries.size(); ++i)
        {
            result |= addTextBox(std::string(textLabels[i] + "##" + std::to_string(sIdOffset)).c_str(), textEntries[i]);
        }

        return addButton(label) | result;
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

    void Gui::compileFonts()
    {
        uint8_t* pFontData;
        int32_t width, height;

        // Initialize font data
        ImGui::GetIO().Fonts->GetTexDataAsAlpha8(&pFontData, &width, &height);
        Texture::SharedPtr pTexture = Texture::create2D(width, height, ResourceFormat::R8Unorm, 1, 1, pFontData);
        mpProgramVars->setTexture("gFont", pTexture);
    }

    void Gui::addFont(const std::string& name, const std::string& filename)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logWarning("Can't find font file `" + filename + "`");
            return;
        }

        float size = 14.0f * mScaleFactor;
        ImFont* pFont = ImGui::GetIO().Fonts->AddFontFromFileTTF(fullpath.c_str(), size);
        mFontMap[name] = pFont;
        compileFonts();
    }

    void Gui::setActiveFont(const std::string& font)
    {
        const auto& it = mFontMap.find(font);
        if (it == mFontMap.end())
        {
            logWarning("Can't find a font named `" + font + "`");
            mpActiveFont = nullptr;
        }
        mpActiveFont = it->second;
    }
}
