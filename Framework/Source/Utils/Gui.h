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
#pragma once
#include <vector>
#include <unordered_map>
#include "glm/vec3.hpp"
#include "UserInput.h"
#include "Graphics/Program//ProgramVars.h"
#include "Graphics/Program//Program.h"
#include "Graphics/GraphicsState.h"

namespace Falcor
{
    class RenderContext;
    struct KeyboardEvent;
    struct MouseEvent;

    /** A class wrapping the external GUI library
    */
    class Gui
    {
        friend class Sample;
    public:
        using UniquePtr = std::unique_ptr<Gui>;
        using UniqueConstPtr = std::unique_ptr<const Gui>;

        /** These structs used to initialize dropdowns
        */
        struct DropdownValue
        {
            int32_t value;      ///< User defined index. Should be unique between different options.
            std::string label;  ///< Label of the dropdown option.
        };

        using DropdownList = std::vector <DropdownValue>;

        struct RadioButton
        {
            int32_t buttonID;  ///< User defined index. Should be unique between different options in the same group.
            std::string label; ///< Label of the radio button.
            bool sameLine;     ///< Whether the button should appear on the same line as the previous widget/button.
        };

        using RadioButtonGroup = std::vector<RadioButton>;

        /** Create a new GUI object. Each object is essentially a container for a GUI window
        */
        static UniquePtr create(uint32_t width, uint32_t height);

        /** Render the GUI
        */
        void render(RenderContext* pContext, float elapsedTime);

        /** Handle window resize events
        */
        void onWindowResize(uint32_t width, uint32_t height);

        /** Handle mouse events
        */
        bool onMouseEvent(const MouseEvent& event);

        /** Handle keyboard events
        */
        bool onKeyboardEvent(const KeyboardEvent& event);

        /** Static text
            \param[in] text The string to display
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
        */
        void addText(const char text[], bool sameLine = false);

        /** Button. Will return true if the button was pressed
            \param[in] label Text to display on the button
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
        */
        bool addButton(const char label[], bool sameLine = false);

        /** Adds a group of radio buttons.
            \param[in] buttons List of buttons to show.
            \param[out] activeID If a button was clicked, activeID will be set to the ID of the clicked button.
            \return Whether activeID changed.
        */
        bool addRadioButtons(const RadioButtonGroup& buttons, int32_t& activeID);

        /** Begin a collapsible group block
            \param[in] label Display name of the group
            \param[in] beginExpanded Whether group should be expanded initially
            \return Returns true if the group is expanded, otherwise false. Use it to avoid making unnecessary calls
        */
        bool beginGroup(const char label[], bool beginExpanded = false);
        bool beginGroup(const std::string& label, bool beginExpanded = false) { return beginGroup(label.c_str(), beginExpanded); }

        /** End a collapsible group block
        */
        void endGroup();

        /** Adds a floating-point UI element.
            \param[in] label The name of the widget.
            \param[in] var A reference to a float that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for the float.
            \param[in] maxVal Optional. The maximum allowed value for the float.
            \param[in] step Optional. The step rate for the float.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addFloatVar(const char label[], float& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, float step = 0.001f, bool sameLine = false);

        /** Adds a 2-elements floating-point vector UI element.
            \param[in] label The name of the widget.
            \param[in] var A reference to a float2 that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for each element of the vector.
            \param[in] maxVal Optional. The maximum allowed value for each element ofthe vector.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addFloat2Var(const char label[], glm::vec2& var, float minVal = -1, float maxVal = 1, bool sameLine = false);

        /** Adds a 3-elements floating-point vector UI element.
            \param[in] label The name of the widget.
            \param[in] var A reference to a float3 that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for each element of the vector.
            \param[in] maxVal Optional. The maximum allowed value for each element of the vector.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addFloat3Var(const char label[], glm::vec3& var, float minVal = -1, float maxVal = 1, bool sameLine = false);

        /** Adds a 4-elements floating-point vector UI element.
            \param[in] label The name of the widget.
            \param[in] var A reference to a float4 that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for each element of the vector.
            \param[in] maxVal Optional. The maximum allowed value for each element of the vector.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addFloat4Var(const char label[], glm::vec4& var, float minVal = -1, float maxVal = 1, bool sameLine = false);

        /** Adds a checkbox.
            \param[in] label The name of the checkbox.
            \param[in] var A reference to a boolean that will be updated directly when the checkbox state changes.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addCheckBox(const char label[], bool& pVar, bool sameLine = false);

        /** Adds an RGB color UI widget.
            \param[in] label The name of the widget.
            \param[in] var A reference to a vector that will be updated directly when the widget state changes.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addRgbColor(const char label[], glm::vec3& var, bool sameLine = false);

        /** Adds an RGBA color UI widget.
            \param[in] label The name of the widget.
            \param[in] var A reference to a vector that will be updated directly when the widget state changes.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addRgbaColor(const char label[], glm::vec4& var, bool sameLine = false);

        /** Adds an integer UI widget.
            \param[in] label The name of the widget.
            \param[in] var A reference to an integer that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for the variable.
            \param[in] maxVal Optional. The maximum allowed value for the variable.
            \param[in] step Optional. The step rate.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addIntVar(const char label[], int32_t& var, int minVal = -INT32_MAX, int maxVal = INT32_MAX, int step = 1, bool sameLine = false);

        /** Add a separator
        */
        void addSeparator();

        /** Adds a dropdown menu. This will update a user variable directly, so the user has to keep track of that for changes.
            If you want notifications whenever the select option changed, use Gui#addDropdownWithCallback().
            \param[in] label The name of the dropdown menu.
            \param[in] values A list of options to show in the dropdown menu.
            \param[in] var A reference to a user variable that will be updated directly when a dropdown option changes. This correlates to the 'pValue' field in Gui#SDropdownValue struct.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addDropdown(const char label[], const DropdownList& values, uint32_t& var, bool sameLine = false);

        /** Render a tooltip. This will display a small question mark next to the last label item rendered and will display the tooltip if the user hover over it
            \param[in] tip The tooltip's text
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
        */
        void addTooltip(const char tip[], bool sameLine = true);

        /** Adds a text box.
            \param[in] label The name of the variable.
            \param[in] buf A character buffer with the initialize text. The buffer will be updated if a text is entered.
            \param[in] bufSize The size of the text buffer
            \param[in] lineCount Number of lines in the text-box. If larger then 1 will create a multi-line box
            \return true if the value changed, otherwise false
        */
        bool addTextBox(const char label[], char buf[], size_t bufSize, uint32_t lineCount = 1);

        /** Adds a text box.
            \param[in] label The name of the variable.
            \param[in] text A string with the initialize text. The string will be updated if a text is entered.
            \param[in] lineCount Number of lines in the text-box. If larger then 1 will create a multi-line box
            \return true if the value changed, otherwise false
        */
        bool addTextBox(const char label[], std::string& text, uint32_t lineCount = 1);

        using GraphCallback = float(*)(void*, int32_t index);

        /** Adds a graph based on a function
            \param[in] label The name of the widget.
            \param[in] func A function pointer to calculate the values in the graph
            \param[in] pUserData A user-data pointer to pass to the callback function
            \param[in] sampleCount Number of sample-points in the graph
            \param[in] sampleOffset Optional. Determines the value for the center of the x-axis
            \param[in] yMin Optional. The minimum value of the y-axis. Use FLT_MAX to auto-detect the range based on the function and the provided x-range
            \param[in] yMax Optional. The maximum value of the y-axis. Use FLT_MAX to auto-detect the range based on the function and the provided x-range
            \param[in] width Optional. The width of the graph widget. 0 means auto-detect (fits the widget to the GUI width)
            \param[in] height Optional. The height of the graph widget. 0 means auto-detect (no idea what's the logic. Too short.)
        */
        void addGraph(const char label[], GraphCallback func, void* pUserData, uint32_t sampleCount, int32_t sampleOffset, float yMin = FLT_MAX, float yMax = FLT_MAX, uint32_t width = 0, uint32_t height = 100);

        /** Adds a direction widget
            \param[in] label The name of the widget.
            \param[in] direction A reference for the direction variable
            \return true if the value changed, otherwise false
        */
        bool addDirectionWidget(const char label[], glm::vec3& direction);

        /** Set global font size scaling
        */
        static void setGlobalFontScaling(float scale);

        /** Create a new window on the stack
        */
        void pushWindow(const char label[], uint32_t width = 0, uint32_t height = 0, uint32_t x = 0, uint32_t y = 0, bool showTitleBar = true);

        /** End a window block
        */
        void popWindow();

        /** Start a new frame. Must be called at the start of each frame
        */
        void beginFrame();

    protected:
        bool keyboardCallback(const KeyboardEvent& keyEvent);
        bool mouseCallback(const MouseEvent& mouseEvent);
        void windowSizeCallback(uint32_t width, uint32_t height);

    private:
        Gui() = default;
        void init();
        void createVao(uint32_t vertexCount, uint32_t indexCount);

        struct ComboData
        {
            uint32_t lastVal = -1;
            int32_t currentItem = -1;
        };
        std::map<std::string, ComboData> mDropDownValues;

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
        GraphicsVars::SharedPtr mpProgramVars;
        GraphicsProgram::SharedPtr mpProgram;
        GraphicsState::SharedPtr mpPipelineState;
        uint32_t mGroupStackSize = 0;
        float mFontScale = 1;
    };
}
