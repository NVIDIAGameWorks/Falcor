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
#include "Gui.h"
#include "dear_imgui/imgui.h"
#include "UserInput.h"
#include "Core/API/RenderContext.h"
#include "glm/gtc/type_ptr.hpp"
#include "Utils/StringUtils.h"

#pragma warning (disable : 4756) // overflow in constant arithmetic caused by calculating the setFloat*() functions (when calculating the step and min/max are +/- INF)
namespace Falcor
{
    class GuiImpl
    {
    public:
        GuiImpl() = default;

    private:
        friend class Gui;
        void init(Gui* pGui, float scaleFactor);
        void createVao(uint32_t vertexCount, uint32_t indexCount);
        void compileFonts();

        // Helper to create multiple inline text boxes
        bool addCheckboxes(const char label[], bool* pData, uint32_t numCheckboxes, bool sameLine);

        struct ComboData
        {
            uint32_t lastVal = -1;
            int32_t currentItem = -1;
        };
        std::unordered_map<std::string, ComboData> mDropDownValues;

        // This struct is used to cache the mouse events
        struct MouseEvents
        {
            bool buttonPressed[3] = { 0 };
            bool buttonReleased[3] = { 0 };
        };

        MouseEvents mMouseEvents;
        void setIoMouseEvents();
        void resetMouseEvents();

        Vao::SharedPtr mpVao;
        VertexLayout::SharedPtr mpLayout;
        GraphicsState::SharedPtr mpPipelineState;
        uint32_t mGroupStackSize = 0;

        GraphicsProgram::SharedPtr mpProgram;
        GraphicsVars::SharedPtr mpProgramVars;
        std::vector<Texture::SharedPtr> mpImages;
        ParameterBlockReflection::BindLocation mGuiImageLoc;
        float mScaleFactor = 1.0f;
        std::unordered_map<std::string, ImFont*> mFontMap;
        ImFont* mpActiveFont = nullptr;

        bool beginMenu(const char* name);
        void endMenu();
        bool beginMainMenuBar();
        void endMainMenuBar();
        bool beginDropDownMenu(const char label[]);
        void endDropDownMenu();
        bool addMenuItem(const char label[], bool& var, const char shortcut[] = nullptr);
        bool addMenuItem(const char label[], const char shortcut[] = nullptr);

        bool pushWindow(const char label[], bool& open, uint2 size = { 250, 200 }, uint2 pos = { 20, 40 }, Gui::WindowFlags flags = Gui::WindowFlags::Default);
        void popWindow();
        void setCurrentWindowPos(uint32_t x, uint32_t y);
        void setCurrentWindowSize(uint32_t width, uint32_t height);
        void beginColumns(uint32_t numColumns);
        void nextColumn();

        bool beginGroup(const char label[], bool beginExpanded = false);
        bool beginGroup(const std::string& label, bool beginExpanded = false) { return beginGroup(label.c_str(), beginExpanded); }
        void endGroup();

        void indent(float i);
        void addSeparator(uint32_t count = 1);
        void addDummyItem(const char label[], const float2& size, bool sameLine = false);
        void addRect(const float2& size, const float4& color = float4(1.0f, 1.0f, 1.0f, 1.0f), bool filled = false, bool sameLine = false);
        bool addDropdown(const char label[], const Gui::DropdownList& values, uint32_t& var, bool sameLine = false);
        bool addButton(const char label[], bool sameLine = false);
        bool addRadioButtons(const Gui::RadioButtonGroup& buttons, uint32_t& activeID);
        bool addDirectionWidget(const char label[], float3& direction);
        bool addCheckbox(const char label[], bool& var, bool sameLine = false);
        bool addCheckbox(const char label[], int& var, bool sameLine = false);
        template<typename T>
        bool addBoolVecVar(const char label[], T& var, bool sameLine = false);
        bool addDragDropSource(const char label[], const char dataLabel[], const std::string& payloadString);
        bool addDragDropDest(const char dataLabel[], std::string& payloadString);

        void addText(const char text[], bool sameLine = false);
        bool addTextbox(const char label[], std::string& text, uint32_t lineCount = 1, Gui::TextFlags flags = Gui::TextFlags::Empty);
        bool addTextbox(const char label[], char buf[], size_t bufSize, uint32_t lineCount = 1, Gui::TextFlags flags = Gui::TextFlags::Empty);
        bool addMultiTextbox(const char label[], const std::vector<std::string>& textLabels, std::vector<std::string>& textEntries);
        void addTooltip(const char tip[], bool sameLine = true);

        bool addRgbColor(const char label[], float3& var, bool sameLine = false);
        bool addRgbaColor(const char label[], float4& var, bool sameLine = false);

        void addImage(const char label[], const Texture::SharedPtr& pTex, float2 size = float2(0), bool maintainRatio = true, bool sameLine = false);
        bool addImageButton(const char label[], const Texture::SharedPtr& pTex, float2 size, bool maintainRatio = true, bool sameLine = false);

        template<typename T>
        bool addScalarVar(const char label[], T& var, T minVal = std::numeric_limits<T>::lowest(), T maxVal = std::numeric_limits<T>::max(), float step = 1.0f, bool sameLine = false, const char* displayFormat = nullptr);
        template<typename T>
        bool addScalarSlider(const char label[], T& var, T minVal = std::numeric_limits<T>::lowest(), T maxVal = std::numeric_limits<T>::max(), bool sameLine = false, const char* displayFormat = nullptr);

        template<typename T>
        bool addVecVar(const char label[], T& var, typename T::value_type minVal = std::numeric_limits<typename T::value_type>::lowest(), typename T::value_type maxVal = std::numeric_limits<typename T::value_type>::max(), float step = 1.0f, bool sameLine = false, const char* displayFormat = nullptr);
        template<typename T>
        bool addVecSlider(const char label[], T& var, typename T::value_type minVal = std::numeric_limits<typename T::value_type>::lowest(), typename T::value_type maxVal = std::numeric_limits<typename T::value_type>::max(), bool sameLine = false, const char* displayFormat = nullptr);

        template <typename MatrixType>
        bool addMatrixVar(const char label[], MatrixType& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, bool sameLine = false);

        void addGraph(const char label[], Gui::GraphCallback func, void* pUserData, uint32_t sampleCount, int32_t sampleOffset, float yMin = FLT_MAX, float yMax = FLT_MAX, uint32_t width = 0, uint32_t height = 100);
    };

    void GuiImpl::init(Gui* pGui, float scaleFactor)
    {
        mScaleFactor = scaleFactor;
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
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
        mpProgram = GraphicsProgram::createFromFile("Utils/UI/Gui.slang", "vs", "ps");
        mpProgramVars = GraphicsVars::create(mpProgram->getReflector());
        mpPipelineState->setProgram(mpProgram);

        // Add the default font
        pGui->addFont("", "Framework/Fonts/trebucbd.ttf");
        pGui->setActiveFont("");

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
        dsDesc.setDepthEnabled(false);
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

    void GuiImpl::createVao(uint32_t vertexCount, uint32_t indexCount)
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
        Buffer::SharedPtr pIB = createIB ? Buffer::create(requiredIbSize, Buffer::BindFlags::Index, Buffer::CpuAccess::Write, nullptr) : mpVao->getIndexBuffer();
        mpVao = Vao::create(Vao::Topology::TriangleList, mpLayout, pVB, pIB, ResourceFormat::R16Uint);
    }

    void GuiImpl::compileFonts()
    {
        uint8_t* pFontData;
        int32_t width, height;

        // Initialize font data
        ImGui::GetIO().Fonts->GetTexDataAsAlpha8(&pFontData, &width, &height);
        Texture::SharedPtr pTexture = Texture::create2D(width, height, ResourceFormat::R8Unorm, 1, 1, pFontData);
        mpProgramVars->setTexture("gFont", pTexture);
    }

    bool GuiImpl::addCheckboxes(const char label[], bool* pData, uint32_t numCheckboxes, bool sameLine)
    {
        bool modified = false;
        std::string labelString(std::string("##") + label + '0');

        for (uint32_t i = 0; i < numCheckboxes - 1; ++i)
        {
            labelString[labelString.size() - 1] = '0' + static_cast<int32_t>(i);
            modified |= addCheckbox(labelString.c_str(), pData[i], (!i) ? sameLine : true);
        }

        addCheckbox(label, pData[numCheckboxes - 1], true);

        return modified;
    }

    void GuiImpl::setIoMouseEvents()
    {
        ImGuiIO& io = ImGui::GetIO();
        memcpy(io.MouseDown, mMouseEvents.buttonPressed, sizeof(mMouseEvents.buttonPressed));
    }

    void GuiImpl::resetMouseEvents()
    {
        for (uint32_t i = 0; i < arraysize(mMouseEvents.buttonPressed); i++)
        {
            if (mMouseEvents.buttonReleased[i])
            {
                mMouseEvents.buttonPressed[i] = mMouseEvents.buttonReleased[i] = false;
            }
        }
    }

    bool GuiImpl::beginMenu(const char* name)
    {
        return ImGui::BeginMenu(name);
    }

    void GuiImpl::endMenu()
    {
        return ImGui::EndMenu();
    }

    bool GuiImpl::beginMainMenuBar()
    {
        bool isOpen = ImGui::BeginMainMenuBar();
        return isOpen;
    }

    void GuiImpl::endMainMenuBar()
    {
        ImGui::EndMainMenuBar();
    }

    bool GuiImpl::beginDropDownMenu(const char label[])
    {
        return ImGui::BeginMenu(label);
    }

    void GuiImpl::endDropDownMenu()
    {
        ImGui::EndMenu();
    }

    bool GuiImpl::addMenuItem(const char label[], const char shortcut[])
    {
        return ImGui::MenuItem(label, shortcut);
    }

    bool GuiImpl::addMenuItem(const char label[], bool& var, const char shortcut[])
    {
        return ImGui::MenuItem(label, shortcut, &var);
    }

    bool GuiImpl::pushWindow(const char label[], bool& open, uint2 size, uint2 pos, Gui::WindowFlags flags)
    {
        bool allowClose = is_set(flags, Gui::WindowFlags::CloseButton);
        if (allowClose)
        {
            if (!is_set(flags, Gui::WindowFlags::ShowTitleBar))
            {
                std::string warning("Asking for a close button on  window ");
                logWarning(warning.append(label).append(", but the ShowTitleBar flag is not set on the window. The window will not be able to display a close button."));
            }
        }

        float2 posFloat(pos);
        posFloat *= mScaleFactor;
        ImVec2 fPos(posFloat.x, posFloat.y);
        ImVec2 fSize(float(size.x), float(size.y));
        ImGui::SetNextWindowSize(fSize, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(fPos, ImGuiCond_FirstUseEver);
        int imguiFlags = 0;
        if (!is_set(flags, Gui::WindowFlags::ShowTitleBar)) imguiFlags |= ImGuiWindowFlags_NoTitleBar;
        if (!is_set(flags, Gui::WindowFlags::AllowMove))  imguiFlags |= ImGuiWindowFlags_NoMove;
        if (!is_set(flags, Gui::WindowFlags::SetFocus)) imguiFlags |= ImGuiWindowFlags_NoFocusOnAppearing;
        if (is_set(flags, Gui::WindowFlags::NoResize)) imguiFlags |= ImGuiWindowFlags_NoResize;
        if (is_set(flags, Gui::WindowFlags::AutoResize)) imguiFlags |= ImGuiWindowFlags_AlwaysAutoResize;

        ImGui::Begin(label, allowClose ? &open : nullptr, imguiFlags);

        if (!open) ImGui::End();
        else ImGui::PushFont(mpActiveFont);
        return open;
    }

    void GuiImpl::popWindow()
    {
        ImGui::PopFont();
        ImGui::End();
    }

    void GuiImpl::setCurrentWindowPos(uint32_t x, uint32_t y)
    {
        ImGui::SetWindowPos({ static_cast<float>(x), static_cast<float>(y) });
    }

    void GuiImpl::setCurrentWindowSize(uint32_t width, uint32_t height)
    {
        ImGui::SetWindowSize({ static_cast<float>(width), static_cast<float>(height) });
    }

    void GuiImpl::beginColumns(uint32_t numColumns)
    {
        ImGui::Columns(numColumns);
    }

    void GuiImpl::nextColumn()
    {
        ImGui::NextColumn();
    }

    bool GuiImpl::beginGroup(const char name[], bool beginExpanded)
    {
        std::string nameString(name);
        ImGuiTreeNodeFlags flags = beginExpanded ? ImGuiTreeNodeFlags_DefaultOpen : 0;
        bool visible = mGroupStackSize ? ImGui::TreeNodeEx(name, flags) : ImGui::CollapsingHeader(name, flags);
        if (visible) mGroupStackSize++;

        std::string popupName = std::string("HeaderOptions##") + nameString;
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(1)) ImGui::OpenPopup(popupName.c_str());

        if (ImGui::BeginPopup(popupName.c_str()))
        {
            if (ImGui::Button("Open in Window")) ImGui::CloseCurrentPopup();
            if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
        }

        return visible;
    }

    void GuiImpl::endGroup()
    {
        assert(mGroupStackSize >= 1);
        mGroupStackSize--;
        if (mGroupStackSize) ImGui::TreePop();
    }

    void GuiImpl::indent(float i)
    {
        ImGui::Indent(i);
    }

    void GuiImpl::addSeparator(uint32_t count)
    {
        for (uint32_t i = 0; i < count; i++) ImGui::Separator();
    }

    void GuiImpl::addDummyItem(const char label[], const float2& size, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        ImGui::PushID(label);
        ImGui::Dummy({ size.x, size.y });
        ImGui::PopID();
    }

    void GuiImpl::addRect(const float2& size, const float4& color, bool filled, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();

        const ImVec2& cursorPos = ImGui::GetCursorScreenPos();
        ImVec2 bottomLeft{ cursorPos.x + size.x, cursorPos.y + size.y };
        ImVec4 rectColor{ color.x, color.y, color.z, color.w };

        if (filled)
        {
            ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetCursorScreenPos(), bottomLeft, ImGui::ColorConvertFloat4ToU32(rectColor));
        }
        else
        {
            ImGui::GetWindowDrawList()->AddRect(ImGui::GetCursorScreenPos(), bottomLeft, ImGui::ColorConvertFloat4ToU32(rectColor));
        }
    }

    bool GuiImpl::addDropdown(const char label[], const Gui::DropdownList& values, uint32_t& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        // Check if we need to update the currentItem
        const auto& iter = mDropDownValues.find(label);
        int curItem;
        if ((iter == mDropDownValues.end()) || (iter->second.lastVal != var))
        {
            // Search the current val
            for (uint32_t i = 0; i < values.size(); i++)
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
        auto prevItem = curItem;
        //This returns true if the combo is interacted with at all
        bool b = ImGui::Combo(label, &curItem, comboStr.c_str());
        mDropDownValues[label].currentItem = curItem;
        mDropDownValues[label].lastVal = values[curItem].value;
        var = values[curItem].value;
        //Only return true if value is changed
        return b && prevItem != curItem;
    }

    bool GuiImpl::addButton(const char label[], bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::Button(label);
    }

    bool GuiImpl::addRadioButtons(const Gui::RadioButtonGroup& buttons, uint32_t& activeID)
    {
        auto oldValue = activeID;

        for (const auto& button : buttons)
        {
            if (button.sameLine) ImGui::SameLine();
            ImGui::RadioButton(button.label.c_str(), (int*)&activeID, button.buttonID);
        }

        return oldValue != activeID;
    }

    bool GuiImpl::addDirectionWidget(const char label[], float3& direction)
    {
        float3 dir = direction;
        bool b = addVecVar(label, dir, -1.f, 1.f, 0.001f, false, "%.3f");
        direction = glm::normalize(dir);
        return b;
    }

    bool GuiImpl::addCheckbox(const char label[], bool& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::Checkbox(label, &var);
    }

    bool GuiImpl::addCheckbox(const char label[], int& var, bool sameLine)
    {
        bool value = (var != 0);
        bool modified = addCheckbox(label, value, sameLine);
        var = (value ? 1 : 0);
        return modified;
    }

    template<typename T>
    bool GuiImpl::addBoolVecVar(const char label[], T& var, bool sameLine)
    {
        return addCheckboxes(label, glm::value_ptr(var), var.length(), sameLine);
    }

    bool GuiImpl::addDragDropSource(const char label[], const char dataLabel[], const std::string& payloadString)
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

    bool GuiImpl::addDragDropDest(const char dataLabel[], std::string& payloadString)
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

    void GuiImpl::addText(const char text[], bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        ImGui::TextUnformatted(text);
    }

    bool GuiImpl::addTextbox(const char label[], char buf[], size_t bufSize, uint32_t lineCount, Gui::TextFlags flags)
    {
        bool fitWindow = is_set(flags, Gui::TextFlags::FitWindow);
        if (fitWindow) ImGui::PushItemWidth(ImGui::GetWindowWidth());

        if (lineCount > 1)
        {
            const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine;
            return ImGui::InputTextMultiline(label, buf, bufSize, ImVec2(-1.0f, ImGui::GetTextLineHeight() * lineCount), flags);
        }
        else
        {
            return ImGui::InputText(label, buf, bufSize, ImGuiInputTextFlags_EnterReturnsTrue);
        }

        if (fitWindow) ImGui::PopItemWidth();
    }

    bool GuiImpl::addTextbox(const char label[], std::string& text, uint32_t lineCount, Gui::TextFlags flags)
    {
        static const int maxSize = 2048;
        char buf[maxSize];
        copyStringToBuffer(buf, maxSize, text);

        bool result = addTextbox(label, buf, maxSize, lineCount, flags);
        text = std::string(buf);
        return result;
    }

    bool GuiImpl::addMultiTextbox(const char label[], const std::vector<std::string>& textLabels, std::vector<std::string>& textEntries)
    {
        static uint32_t sIdOffset = 0;
        bool result = false;

        for (uint32_t i = 0; i < textEntries.size(); ++i)
        {
            result |= addTextbox(std::string(textLabels[i] + "##" + std::to_string(sIdOffset)).c_str(), textEntries[i]);
        }

        return addButton(label) | result;
    }

    void GuiImpl::addTooltip(const char tip[], bool sameLine)
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

    bool GuiImpl::addRgbColor(const char label[], float3& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::ColorEdit3(label, glm::value_ptr(var));
    }

    bool GuiImpl::addRgbaColor(const char label[], float4& var, bool sameLine)
    {
        if (sameLine) ImGui::SameLine();
        return ImGui::ColorEdit4(label, glm::value_ptr(var));
    }

    void GuiImpl::addImage(const char label[], const Texture::SharedPtr& pTex, float2 size, bool maintainRatio, bool sameLine)
    {
        assert(pTex);
        if (size == float2(0))
        {
            ImVec2 windowSize = ImGui::GetWindowSize();
            size = { windowSize.x, windowSize.y };
        }

        ImGui::PushID(label);
        if (sameLine) ImGui::SameLine();
        mpImages.push_back(pTex);
        float aspectRatio = maintainRatio ? (static_cast<float>(pTex->getHeight()) / static_cast<float>(pTex->getWidth())) : 1.0f;
        ImGui::Image(reinterpret_cast<ImTextureID>(mpImages.size()), { size.x, maintainRatio ? size.x  * aspectRatio : size.y });
        ImGui::PopID();
    }

    bool GuiImpl::addImageButton(const char label[], const Texture::SharedPtr& pTex, float2 size, bool maintainRatio, bool sameLine)
    {
        assert(pTex);
        mpImages.push_back(pTex);
        if (sameLine) ImGui::SameLine();
        float aspectRatio = maintainRatio ? (static_cast<float>(pTex->getHeight()) / static_cast<float>(pTex->getWidth())) : 1.0f;
        return ImGui::ImageButton(reinterpret_cast<ImTextureID>(mpImages.size()), { size.x, maintainRatio ? size.x  * aspectRatio : size.y });
    }

    template<typename T>
    bool addScalarVarHelper(const char label[], T& var, ImGuiDataType_ imguiType, T minVal, T maxVal, float step, bool sameLine, const char* displayFormat)
    {
        ImGui::PushItemWidth(200);
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragScalar(label, imguiType, &var, step, &minVal, &maxVal, displayFormat);
        var = glm::clamp(var, T(minVal), T(maxVal));
        ImGui::PopItemWidth();
        return b;
    }

    template<typename T>
    bool GuiImpl::addScalarVar(const char label[], T& var, T minVal, T maxVal, float step, bool sameLine, const char* displayFormat)
    {
        if (std::is_same<T, int32_t>::value)
        {
            return addScalarVarHelper(label, var, ImGuiDataType_S32, minVal, maxVal, step, sameLine, displayFormat);
        }
        else if (std::is_same<T, uint32_t>::value)
        {
            return addScalarVarHelper(label, var, ImGuiDataType_U32, minVal, maxVal, step, sameLine, displayFormat);
        }
        else if (std::is_same<T, float>::value)
        {
            return addScalarVarHelper(label, var, ImGuiDataType_Float, minVal, maxVal, step, sameLine, displayFormat);
        }
        else if (std::is_same<T, uint64_t>::value)
        {
            return addScalarVarHelper(label, var, ImGuiDataType_U64, minVal, maxVal, step, sameLine, displayFormat);
        }
        else
        {
            logError("Unsupported slider type");
            return false;
        }
    }

    template<typename T>
    bool addScalarSliderHelper(const char label[], T& var, ImGuiDataType_ imguiType, T minVal, T maxVal, bool sameLine, const char* displayFormat)
    {
        ImGui::PushItemWidth(200);
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::SliderScalar(label, imguiType, &var, &minVal, &maxVal, displayFormat);
        ImGui::PopItemWidth();
        return b;
    }

    template<typename T>
    bool GuiImpl::addScalarSlider(const char label[], T& var, T minVal, T maxVal, bool sameLine, const char* displayFormat)
    {
        if (std::is_same<T, int32_t>::value)
        {
            return addScalarSliderHelper(label, var, ImGuiDataType_S32, minVal, maxVal, sameLine, displayFormat);
        }
        else if (std::is_same<T, uint32_t>::value)
        {
            return addScalarSliderHelper(label, var, ImGuiDataType_U32, minVal, maxVal, sameLine, displayFormat);
        }
        else if (std::is_same<T, float>::value)
        {
            return addScalarSliderHelper(label, var, ImGuiDataType_Float, minVal, maxVal, sameLine, displayFormat);
        }
        else
        {
            logError("Unsupported slider type");
            return false;
        }
    }

    template<typename T>
    bool addVecVarHelper(const char label[], T& var, ImGuiDataType_ imguiType, typename T::value_type minVal, typename T::value_type maxVal, float step, bool sameLine, const char* displayFormat)
    {
        ImGui::PushItemWidth(200);
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::DragScalarN(label, imguiType, glm::value_ptr(var), var.length(), step, &minVal, &maxVal, displayFormat);
        var = glm::clamp(var, T(minVal), T(maxVal));
        ImGui::PopItemWidth();
        return b;
    }

    template<typename T>
    bool GuiImpl::addVecVar(const char label[], T& var, typename T::value_type minVal, typename T::value_type maxVal, float step, bool sameLine, const char* displayFormat)
    {
        if (std::is_same<T::value_type, int32_t>::value)
        {
            return addVecVarHelper(label, var, ImGuiDataType_S32, minVal, maxVal, step, sameLine, displayFormat);
        }
        else if (std::is_same<T::value_type, uint32_t>::value)
        {
            return addVecVarHelper(label, var, ImGuiDataType_U32, minVal, maxVal, step, sameLine, displayFormat);
        }
        else if (std::is_same<T::value_type, float>::value)
        {
            return addVecVarHelper(label, var, ImGuiDataType_Float, minVal, maxVal, step, sameLine, displayFormat);
        }
        else if (std::is_same<T::value_type, uint64_t>::value)
        {
            return addVecVarHelper(label, var, ImGuiDataType_U64, minVal, maxVal, step, sameLine, displayFormat);
        }
        else
        {
            logError("Unsupported slider type");
            return false;
        }
    }

    template<typename T>
    bool addVecSliderHelper(const char label[], T& var, ImGuiDataType_ imguiType, typename T::value_type minVal, typename T::value_type maxVal, bool sameLine, const char* displayFormat)
    {
        ImGui::PushItemWidth(200);
        if (sameLine) ImGui::SameLine();
        bool b = ImGui::SliderScalarN(label, imguiType, glm::value_ptr(var), var.length(), &minVal, &maxVal, displayFormat);
        ImGui::PopItemWidth();
        return b;
    }

    template<typename T>
    bool GuiImpl::addVecSlider(const char label[], T& var, typename T::value_type minVal, typename T::value_type maxVal, bool sameLine, const char* displayFormat)
    {
        if (std::is_same<T::value_type, int32_t>::value)
        {
            return addVecSliderHelper(label, var, ImGuiDataType_S32, minVal, maxVal, sameLine, displayFormat);
        }
        else if (std::is_same<T::value_type, uint32_t>::value)
        {
            return addVecSliderHelper(label, var, ImGuiDataType_U32, minVal, maxVal, sameLine, displayFormat);
        }
        else if (std::is_same<T::value_type, float>::value)
        {
            return addVecSliderHelper(label, var, ImGuiDataType_Float, minVal, maxVal, sameLine, displayFormat);
        }
        else
        {
            logError("Unsupported slider type");
            return false;
        }
    }

    template<typename MatrixType>
    bool GuiImpl::addMatrixVar(const char label[], MatrixType& var, float minVal, float maxVal, bool sameLine)
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

            b |= addVecVar<typename MatrixType::col_type>(stringToDisplay.c_str(), var[i], minVal, maxVal, 0.001f, sameLine);

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
            else if (i == 1)
            {
                bottomRight.y = topLeft.y + (bottomRight.y - topLeft.y) * (var.length());
                bottomRight.x -= ImGui::GetStyle().ItemInnerSpacing.x * 3 - 1;
                bottomRight.y -= ImGui::GetStyle().ItemInnerSpacing.y - 1;
                topLeft.x -= 1; topLeft.y -= 1;
                auto colorVec4 = ImGui::GetStyleColorVec4(ImGuiCol_ScrollbarGrab); colorVec4.w *= 0.25f;
                ImU32 color = ImGui::ColorConvertFloat4ToU32(colorVec4);
                ImGui::GetWindowDrawList()->AddRect(topLeft, bottomRight, color);
            }
        }
        return b;
    }

    void GuiImpl::addGraph(const char label[], Gui::GraphCallback func, void* pUserData, uint32_t sampleCount, int32_t sampleOffset, float yMin, float yMax, uint32_t width, uint32_t height)
    {
        ImVec2 imSize{ (float)width, (float)height };
        ImGui::PlotLines(label, func, pUserData, (int32_t)sampleCount, sampleOffset, nullptr, yMin, yMax, imSize);
    }

    Gui::~Gui()
    {
        ImGui::DestroyContext();
    }

    Gui::UniquePtr Gui::create(uint32_t width, uint32_t height, float scaleFactor)
    {
        UniquePtr pGui = UniquePtr(new Gui);
        pGui->mpWrapper = new GuiImpl;
        pGui->mpWrapper->init(pGui.get(), scaleFactor);
        pGui->onWindowResize(width, height);
        return pGui;
    }

    float4 Gui::pickUniqueColor(const std::string& key)
    {
        union hashedValue
        {
            size_t st;
            int32_t i32[2];
        };
        hashedValue color;
        color.st = std::hash<std::string>()(key);

        return float4(color.i32[0] % 1000 / 2000.0f, color.i32[1] % 1000 / 2000.0f, (color.i32[0] * color.i32[1]) % 1000 / 2000.0f, 1.0f);
    }

    void Gui::addFont(const std::string& name, const std::string& filename)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logWarning("Can't find font file '" + filename + "'");
            return;
        }

        float size = 14.0f * mpWrapper->mScaleFactor;
        ImFont* pFont = ImGui::GetIO().Fonts->AddFontFromFileTTF(fullpath.c_str(), size);
        mpWrapper->mFontMap[name] = pFont;
        mpWrapper->compileFonts();
    }

    void Gui::setActiveFont(const std::string& font)
    {
        const auto& it = mpWrapper->mFontMap.find(font);
        if (it == mpWrapper->mFontMap.end())
        {
            logWarning("Can't find a font named '" + font + "'");
            mpWrapper->mpActiveFont = nullptr;
        }
        mpWrapper->mpActiveFont = it->second;
    }

    ImFont* Gui::getFont(std::string f)
    {
        if (f.size()) return mpWrapper->mFontMap.at(f);
        else return mpWrapper->mpActiveFont;
    }

    void Gui::beginFrame()
    {
        ImGui::NewFrame();
    }

    void Gui::setGlobalGuiScaling(float scale)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.FontGlobalScale = scale;
        ImGui::GetStyle().ScaleAllSizes(scale);
    }

    void Gui::render(RenderContext* pContext, const Fbo::SharedPtr& pFbo, float elapsedTime)
    {
        while (mpWrapper->mGroupStackSize) mpWrapper->endGroup();

        // Set the mouse state
        mpWrapper->setIoMouseEvents();

        ImGui::Render();
        ImDrawData* pDrawData = ImGui::GetDrawData();

        mpWrapper->resetMouseEvents();

        // Update the VAO
        mpWrapper->createVao(pDrawData->TotalVtxCount, pDrawData->TotalIdxCount);
        mpWrapper->mpPipelineState->setVao(mpWrapper->mpVao);

        // Upload the data
        ImDrawVert* pVerts = (ImDrawVert*)mpWrapper->mpVao->getVertexBuffer(0)->map(Buffer::MapType::WriteDiscard);
        uint16_t* pIndices = (uint16_t*)mpWrapper->mpVao->getIndexBuffer()->map(Buffer::MapType::WriteDiscard);

        for (int n = 0; n < pDrawData->CmdListsCount; n++)
        {
            const ImDrawList* pCmdList = pDrawData->CmdLists[n];
            memcpy(pVerts, pCmdList->VtxBuffer.Data, pCmdList->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(pIndices, pCmdList->IdxBuffer.Data, pCmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
            pVerts += pCmdList->VtxBuffer.Size;
            pIndices += pCmdList->IdxBuffer.Size;
        }
        mpWrapper->mpVao->getVertexBuffer(0)->unmap();
        mpWrapper->mpVao->getIndexBuffer()->unmap();
        mpWrapper->mpPipelineState->setFbo(pFbo);

        // Setup viewport
        GraphicsState::Viewport vp;
        vp.originX = 0;
        vp.originY = 0;
        vp.width = ImGui::GetIO().DisplaySize.x;
        vp.height = ImGui::GetIO().DisplaySize.y;
        vp.minDepth = 0;
        vp.maxDepth = 1;
        mpWrapper->mpPipelineState->setViewport(0, vp);

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
                    mpWrapper->mpProgramVars->setSrv(mpWrapper->mGuiImageLoc, (mpWrapper->mpImages[reinterpret_cast<size_t>(pCmd->TextureId) - 1])->getSRV());
                    mpWrapper->mpProgramVars["PerFrameCB"]["useGuiImage"] = true;
                }
                else
                {
                    mpWrapper->mpProgramVars["PerFrameCB"]["useGuiImage"] = false;
                }
                mpWrapper->mpPipelineState->setScissors(0, scissor);
                pContext->drawIndexed(mpWrapper->mpPipelineState.get(), mpWrapper->mpProgramVars.get(), pCmd->ElemCount, idxOffset, vtxOffset);
                idxOffset += pCmd->ElemCount;
            }
            vtxOffset += pCmdList->VtxBuffer.Size;
        }

        // Prepare for the next frame
        ImGuiIO& io = ImGui::GetIO();
        io.DeltaTime = elapsedTime;
        mpWrapper->mGroupStackSize = 0;

        mpWrapper->mpImages.clear();
    }

    void Gui::onWindowResize(uint32_t width, uint32_t height)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize.x = (float)width;
        io.DisplaySize.y = (float)height;
#ifdef FALCOR_VK
        mpWrapper->mpProgramVars["PerFrameCB"]["scale"] = 2.0f / float2(io.DisplaySize.x, io.DisplaySize.y);
        mpWrapper->mpProgramVars["PerFrameCB"]["offset"] = float2(-1.0f);
#else
        mpWrapper->mpProgramVars["PerFrameCB"]["scale"] = 2.0f / float2(io.DisplaySize.x, -io.DisplaySize.y);
        mpWrapper->mpProgramVars["PerFrameCB"]["offset"] = float2(-1.0f, 1.0f);
#endif
    }

    bool Gui::onMouseEvent(const MouseEvent& event)
    {
        ImGuiIO& io = ImGui::GetIO();
        switch (event.type)
        {
        case MouseEvent::Type::LeftButtonDown:
            mpWrapper->mMouseEvents.buttonPressed[0] = true;
            break;
        case MouseEvent::Type::LeftButtonUp:
            mpWrapper->mMouseEvents.buttonReleased[0] = true;
            break;
        case MouseEvent::Type::RightButtonDown:
            mpWrapper->mMouseEvents.buttonPressed[1] = true;
            break;
        case MouseEvent::Type::RightButtonUp:
            mpWrapper->mMouseEvents.buttonReleased[1] = true;
            break;
        case MouseEvent::Type::MiddleButtonDown:
            mpWrapper->mMouseEvents.buttonPressed[2] = true;
            break;
        case MouseEvent::Type::MiddleButtonUp:
            mpWrapper->mMouseEvents.buttonReleased[2] = true;
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

    Gui::Group Gui::Widgets::group(const std::string& label, bool beginExpanded)
    {
        return Group(mpGui, label, beginExpanded);
    }

    void Gui::Widgets::indent(float i)
    {
        if (mpGui) mpGui->mpWrapper->indent(i);
    }

    void Gui::Widgets::separator(uint32_t count)
    {
        if (mpGui) mpGui->mpWrapper->addSeparator(count);
    }

    void Gui::Widgets::dummy(const char label[], const float2& size, bool sameLine)
    {
        if (mpGui) mpGui->mpWrapper->addDummyItem(label, size, sameLine);
    }

    void Gui::Widgets::rect(const float2& size, const float4& color, bool filled, bool sameLine)
    {
        if (mpGui) mpGui->mpWrapper->addRect(size, color, filled, sameLine);
    }

    bool Gui::Widgets::dropdown(const char label[], const DropdownList& values, uint32_t& var, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addDropdown(label, values, var, sameLine) : false;
    }

    bool Gui::Widgets::button(const char label[], bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addButton(label, sameLine) : false;
    }

    bool Gui::Widgets::radioButtons(const RadioButtonGroup& buttons, uint32_t& activeID)
    {
        return mpGui ? mpGui->mpWrapper->addRadioButtons(buttons, activeID) : false;
    }

    bool Gui::Widgets::direction(const char label[], float3& direction)
    {
        return mpGui ? mpGui->mpWrapper->addDirectionWidget(label, direction) : false;
    }

    template<>
    dlldecl bool Gui::Widgets::checkbox<bool>(const char label[], bool& var, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addCheckbox(label, var, sameLine) : false;
    }

    template<>
    dlldecl bool Gui::Widgets::checkbox<int>(const char label[], int& var, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addCheckbox(label, var, sameLine) : false;
    }

    template<typename T>
    bool Gui::Widgets::checkbox(const char label[], T& var, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addBoolVecVar<T>(label, var, sameLine) : false;
    }

#define add_bool_vec_type(TypeName)  template dlldecl bool Gui::Widgets::checkbox<TypeName>(const char[], TypeName&, bool)

    add_bool_vec_type(bool2);
    add_bool_vec_type(bool3);
    add_bool_vec_type(bool4);

#undef add_bool_vec_type

    bool Gui::Widgets::dragDropSource(const char label[], const char dataLabel[], const std::string& payloadString)
    {
        return mpGui ? mpGui->mpWrapper->addDragDropSource(label, dataLabel, payloadString) : false;
    }

    bool Gui::Widgets::dragDropDest(const char dataLabel[], std::string& payloadString)
    {
        return mpGui ? mpGui->mpWrapper->addDragDropDest(dataLabel, payloadString) : false;
    }

    template<typename T, std::enable_if_t<!is_vector<T>::value, bool>>
    bool Gui::Widgets::var(const char label[], T& var, T minVal, T maxVal, float step, bool sameLine, const char* displayFormat)
    {
        return mpGui ? mpGui->mpWrapper->addScalarVar(label, var, minVal, maxVal, step, sameLine, displayFormat) : false;
    }

#define add_scalarVar_type(TypeName) template dlldecl bool Gui::Widgets::var<TypeName>(const char[], TypeName&, TypeName, TypeName, float, bool, const char*)

    add_scalarVar_type(int32_t);
    add_scalarVar_type(uint32_t);
    add_scalarVar_type(uint64_t);
    add_scalarVar_type(float);

#undef add_scalarVar_type

    template<typename T, std::enable_if_t<!is_vector<T>::value, bool>>
    bool Gui::Widgets::slider(const char label[], T& var, T minVal, T maxVal, bool sameLine, const char* displayFormat)
    {
        T lowerBound = glm::clamp(minVal, std::numeric_limits<T>::lowest() / 2, std::numeric_limits<T>::max() / 2);
        T upperBound = glm::clamp(maxVal, std::numeric_limits<T>::lowest() / 2, std::numeric_limits<T>::max() / 2);
        return mpGui ? mpGui->mpWrapper->addScalarSlider(label, var, lowerBound, upperBound, sameLine, displayFormat) : false;
    }

#define add_scalarSlider_type(TypeName) template dlldecl bool Gui::Widgets::slider<TypeName>(const char[], TypeName&, TypeName, TypeName, bool, const char*)

    add_scalarSlider_type(int32_t);
    add_scalarSlider_type(uint32_t);
    add_scalarSlider_type(uint64_t);
    add_scalarSlider_type(float);

#undef add_scalarSlider_type

    template<typename T, std::enable_if_t<is_vector<T>::value, bool>>
    bool Gui::Widgets::var(const char label[], T& var, typename T::value_type minVal, typename T::value_type maxVal, float step, bool sameLine, const char* displayFormat)
    {
        return mpGui ? mpGui->mpWrapper->addVecVar(label, var, minVal, maxVal, step, sameLine, displayFormat) : false;
    }

#define add_vecVar_type(TypeName) template dlldecl bool Gui::Widgets::var<TypeName>(const char[], TypeName&, typename TypeName::value_type, typename TypeName::value_type, float, bool, const char*)

    add_vecVar_type(int2);
    add_vecVar_type(int3);
    add_vecVar_type(int4);
    add_vecVar_type(uint2);
    add_vecVar_type(uint3);
    add_vecVar_type(uint4);
    add_vecVar_type(float2);
    add_vecVar_type(float3);
    add_vecVar_type(float4);

#undef add_vecVar_type

    template<typename T, std::enable_if_t<is_vector<T>::value, bool>>
    bool Gui::Widgets::slider(const char label[], T& var, typename T::value_type minVal, typename T::value_type maxVal, bool sameLine, const char* displayFormat)
    {
        typename T::value_type lowerBound = glm::clamp(minVal, std::numeric_limits<typename T::value_type>::lowest() / 2, std::numeric_limits<typename T::value_type>::max() / 2);
        typename T::value_type upperBound = glm::clamp(maxVal, std::numeric_limits<typename T::value_type>::lowest() / 2, std::numeric_limits<typename T::value_type>::max() / 2);
        return mpGui ? mpGui->mpWrapper->addVecSlider(label, var, lowerBound, upperBound, sameLine, displayFormat) : false;
    }

#define add_vecSlider_type(TypeName) template dlldecl bool Gui::Widgets::slider<TypeName>(const char[], TypeName&, typename TypeName::value_type, typename TypeName::value_type, bool, const char*)

    add_vecSlider_type(int2);
    add_vecSlider_type(int3);
    add_vecSlider_type(int4);
    add_vecSlider_type(uint2);
    add_vecSlider_type(uint3);
    add_vecSlider_type(uint4);
    add_vecSlider_type(float2);
    add_vecSlider_type(float3);
    add_vecSlider_type(float4);

#undef add_vecSlider_type

    void Gui::Widgets::text(const std::string& text, bool sameLine)
    {
        if (mpGui) mpGui->mpWrapper->addText(text.c_str(), sameLine);
    }

    bool Gui::Widgets::textbox(const std::string& label, std::string& text, TextFlags flags)
    {
        return mpGui ? mpGui->mpWrapper->addTextbox(label.c_str(), text, 1, flags) : false;
    }

    bool Gui::Widgets::textbox(const char label[], char buf[], size_t bufSize, uint32_t lineCount, TextFlags flags)
    {
        return mpGui ? mpGui->mpWrapper->addTextbox(label, buf, bufSize, lineCount, flags) : false;
    }

    bool Gui::Widgets::multiTextbox(const char label[], const std::vector<std::string>& textLabels, std::vector<std::string>& textEntries)
    {
        return mpGui ? mpGui->mpWrapper->addMultiTextbox(label, textLabels, textEntries) : false;
    }

    void Gui::Widgets::tooltip(const std::string& text, bool sameLine)
    {
        if (mpGui) mpGui->mpWrapper->addTooltip(text.c_str(), sameLine);
    }

    bool Gui::Widgets::rgbColor(const char label[], float3& var, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addRgbColor(label, var, sameLine) : false;
    }

    bool Gui::Widgets::rgbaColor(const char label[], float4& var, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addRgbaColor(label, var, sameLine) : false;
    }

    bool Gui::Widgets::imageButton(const char label[], const Texture::SharedPtr& pTex, float2 size, bool maintainRatio, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addImageButton(label, pTex, size, maintainRatio, sameLine) : false;
    }

    void Gui::Widgets::image(const char label[], const Texture::SharedPtr& pTex, float2 size, bool maintainRatio, bool sameLine)
    {
        if (mpGui) mpGui->mpWrapper->addImage(label, pTex, size, maintainRatio, sameLine);
    }

    template<typename MatrixType>
    bool Gui::Widgets::matrix(const char label[], MatrixType& var, float minVal, float maxVal, bool sameLine)
    {
        return mpGui ? mpGui->mpWrapper->addMatrixVar(label, var, minVal, maxVal, sameLine) : false;
    }

#define add_matrix_var(TypeName) template dlldecl bool Gui::Widgets::matrix<TypeName>(const char[], TypeName&, float, float, bool)

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

    void Gui::Widgets::graph(const char label[], GraphCallback func, void* pUserData, uint32_t sampleCount, int32_t sampleOffset, float yMin, float yMax, uint32_t width, uint32_t height)
    {
        if (mpGui) mpGui->mpWrapper->addGraph(label, func, pUserData, sampleCount, sampleOffset, yMin, yMax, width, height);
    }

    Gui::Menu::Menu(Gui* pGui, const char* name)
    {
        if (pGui && pGui->mpWrapper->beginMenu(name)) mpGui = pGui;
    }

    Gui::Menu::~Menu()
    {
        release();
    }

    void Gui::Menu::release()
    {
        if (mpGui) mpGui->mpWrapper->endMenu();
        mpGui = nullptr;
    }

    Gui::Menu::Dropdown Gui::Menu::dropdown(const std::string& label)
    {
        return Dropdown(mpGui, label.c_str());
    }

    bool Gui::Menu::item(const std::string& label)
    {
        return mpGui && mpGui->mpWrapper->addMenuItem(label.c_str());
    }

    Gui::Menu::Dropdown::Dropdown(Gui* pGui, const char label[])
    {
        if (pGui && pGui->mpWrapper->beginDropDownMenu(label)) mpGui = pGui;
    }

    Gui::Menu::Dropdown::~Dropdown()
    {
        if (mpGui) release();
    }

    void Gui::Menu::Dropdown::release()
    {
        if (mpGui) mpGui->mpWrapper->endDropDownMenu(); mpGui = nullptr;
    }
    bool Gui::Menu::Dropdown::item(const std::string& label, bool& var, const std::string& shortcut)
    {
        return mpGui && mpGui->mpWrapper->addMenuItem(label.c_str(), var, shortcut.size() ? shortcut.c_str() : nullptr);
    }

    bool Gui::Menu::Dropdown::item(const std::string& label, const std::string& shortcut)
    {
        return mpGui && mpGui->mpWrapper->addMenuItem(label.c_str(), shortcut.size() ? shortcut.c_str() : nullptr);
    }

    void Gui::Menu::Dropdown::separator()
    {
        if (mpGui) mpGui->mpWrapper->addSeparator();
    }

    Gui::Menu Gui::Menu::Dropdown::menu(const char* name)
    {
        return Menu(mpGui, name);
    }

    Gui::Group::Group(Gui* pGui, const std::string& label, bool beginExpanded)
    {
        if (pGui && pGui->mpWrapper->beginGroup(label, beginExpanded)) mpGui = pGui;
    }

    bool Gui::Group::open() const
    {
        return mpGui != nullptr;
    }

    Gui::Group::~Group()
    {
        release();
    }

    void Gui::Group::release()
    {
        if (mpGui) mpGui->mpWrapper->endGroup(); mpGui = nullptr;
    }

    Gui::Window::Window(Gui* pGui, const char* name, uint2 size, uint2 pos, Gui::WindowFlags flags)
    {
        bool open = true;
        if (pGui->mpWrapper->pushWindow(name, open, size, pos, flags)) mpGui = pGui;
    }

    Gui::Window::Window(Gui* pGui, const char* name, bool& open, uint2 size, uint2 pos, Gui::WindowFlags flags)
    {
        if (pGui->mpWrapper->pushWindow(name, open, size, pos, flags)) mpGui = pGui;
    }

    Gui::Window::~Window()
    {
        release();
    }

    void Gui::Window::release()
    {
        if (mpGui) mpGui->mpWrapper->popWindow();
        mpGui = nullptr;
    }

    void Gui::Window::columns(uint32_t numColumns)
    {
        if (mpGui) mpGui->mpWrapper->beginColumns(numColumns);
    }

    void Gui::Window::nextColumn()
    {
        if (mpGui) mpGui->mpWrapper->nextColumn();
    }

    void Gui::Window::windowPos(uint32_t x, uint32_t y)
    {
        if (mpGui) mpGui->mpWrapper->setCurrentWindowPos(x, y);
    }

    void Gui::Window::windowSize(uint32_t width, uint32_t height)
    {
        if (mpGui) mpGui->mpWrapper->setCurrentWindowSize(width, height);
    }

    Gui::MainMenu::MainMenu(Gui* pGui) : Gui::Menu()
    {
        if (pGui->mpWrapper->beginMainMenuBar()) mpGui = pGui;
    }

    Gui::MainMenu::~MainMenu()
    {
        release();
    }

    void Gui::MainMenu::release()
    {
        if (mpGui)
        {
            mpGui->mpWrapper->endMainMenuBar();
            mpGui = nullptr;
        }
    }
}
