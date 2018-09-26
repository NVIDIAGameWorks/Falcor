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

struct ImFont;

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
        static UniquePtr create(uint32_t width, uint32_t height, float scaleFactor = 1.0f);

        ~Gui();

        static glm::vec4 pickUniqueColor(const std::string& key);

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

        /** Display image within imgui
            \param[in] label. Name for id for item.
            \param[in] pTex. Pointer to texture resource to draw in imgui
            \param[in] size. Size in pixels of the image to draw. 0 means fit to window
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
        */
        void addImage(const char label[], const Texture::SharedPtr& pTex, glm::vec2 size = vec2(0), bool maintainRatio = true, bool sameLine = false);

        /** Display rectangle with specified color
            \param[in] size size in pixels of rectangle
            \param[in] color Optional. color as an rgba vec4
            \param[in] filled Optional. If set to true, rectangle will be filled
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
        */
        void addRect(const glm::vec2& size, const glm::vec4& color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), bool filled = false, bool sameLine = false);

        /** Dummy object especially useful for spacing
            \param[in] label. Name for id of item
            \param[in] size. size in pixels of the item.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
        */
        void addDummyItem(const char label[], const glm::vec2& size, bool sameLine = false);

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

        /** Begins main menu bar for top of the window
        */
        bool beginMainMenuBar();

        /** Begins a collapsable menu in the main menu bar of menu items
            \param[in] label name of drop down menu to be displayed.
        */
        bool beginDropDownMenu(const char label[]);

        /** End the drop down menu list of items.
        */
        void endDropDownMenu();

        /** Item to be displayed in dropdown menu for the main menu bar
            \param[in] label name of item to list in the menu.
            \return true if the option was selected in the dropdown
         */
        bool addMenuItem(const char label[]);

        /** Ends the main menu bar for top of the window
        */
        void endMainMenuBar();

        /** Adds a floating-point UI element.
            \param[in] label The name of the widget.
            \param[in] var A reference to a float that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for the float.
            \param[in] maxVal Optional. The maximum allowed value for the float.
            \param[in] step Optional. The step rate for the float.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget.
            \param[in] displayFormat Optional. Formatting string.
            \return true if the value changed, otherwise false
        */
        bool addFloatVar(const char label[], float& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, float step = 0.001f, bool sameLine = false, const char* displayFormat = "%.3f");
        bool addFloat2Var(const char label[], glm::vec2& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, float step = 0.001f, bool sameLine = false, const char* displayFormat = "%.3f");
        bool addFloat3Var(const char label[], glm::vec3& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, float step = 0.001f, bool sameLine = false, const char* displayFormat = "%.3f");
        bool addFloat4Var(const char label[], glm::vec4& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, float step = 0.001f, bool sameLine = false, const char* displayFormat = "%.3f");

        /** Adds a checkbox.
            \param[in] label The name of the checkbox.
            \param[in] var A reference to a boolean that will be updated directly when the checkbox state changes.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addCheckBox(const char label[], bool& pVar, bool sameLine = false);

        /** Adds a checkbox.
            \param[in] label The name of the checkbox.
            \param[in] var A reference to an integer that will be updated directly when the checkbox state changes (0 = unchecked, 1 = checked).
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addCheckBox(const char label[], int& pVar, bool sameLine = false);

        /** Adds a UI widget for multiple checkboxes.
            \param[in] label The name of the widget.
            \param[in] var A reference to the bools that will be updated directly when the checkbox state changes (0 = unchecked, 1 = checked).
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addBool2Var(const char lable[], glm::bvec2& var, bool sameLine = false);
        bool addBool3Var(const char lable[], glm::bvec3& var, bool sameLine = false);
        bool addBool4Var(const char lable[], glm::bvec4& var, bool sameLine = false);

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
        
        /** The source for drag and drop. Call this to allow users to drag out of last gui item.
            \param[in] label The name of the drag and drop widget
            \param[in] dataLabel Destination that has same dataLabel can accept the payload
            \param[in] payloadString Data in payload to be sent and accepted by destination.
            \return true if user is clicking and dragging
         */
        bool dragDropSource(const char label[], const char dataLabel[], const std::string& payloadString);

        /** Destination area for dropping data in drag and drop of last gui item.
            \param[in] dataLabel Named label needs to be the same as source datalabel to accept payload.
            \param[in] payloadString Data sent from the drag and drop source
            \return true if payload is dropped.
        */
        bool dragDropDest(const char dataLabel[], std::string& payloadString);

        /** Adds a UI widget for integers.
            \param[in] label The name of the widget.
            \param[in] var A reference to an integer that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for the variable.
            \param[in] maxVal Optional. The maximum allowed value for the variable.
            \param[in] step Optional. The step rate.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        bool addIntVar(const char label[], int32_t& var, int minVal = -INT32_MAX, int maxVal = INT32_MAX, int step = 1, bool sameLine = false);
        bool addInt2Var(const char label[], glm::ivec2& var, int32_t minVal = -INT32_MAX, int32_t maxVal = INT32_MAX, bool sameLine = false);
        bool addInt3Var(const char label[], glm::ivec3& var, int32_t minVal = -INT32_MAX, int32_t maxVal = INT32_MAX, bool sameLine = false);
        bool addInt4Var(const char label[], glm::ivec4& var, int32_t minVal = -INT32_MAX, int32_t maxVal = INT32_MAX, bool sameLine = false);

        template<typename VectorType>
        bool addFloatVecVar(const char label[], VectorType& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, float step = 0.001f, bool sameLine = false);

        /** Adds an matrix UI widget.
            \param[in] label The name of the widget.
            \param[in] var A reference to the matrix struct that will be updated directly when the widget state changes.
            \param[in] minVal Optional. The minimum allowed value for the variable.
            \param[in] maxVal Optional. The maximum allowed value for the variable.
            \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            \return true if the value changed, otherwise false
        */
        template <typename MatrixType>
        bool addMatrixVar(const char label[], MatrixType& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, bool sameLine = false);

        /** Begin a column within the current window
            \param[in] numColumns requires number of columns within the window.
         */
        void beginColumns(uint32_t numColumns);

        /** Proceed to the next column within the window.
         */
        void nextColumn();

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

        /** Adds multiple text boxes for one confirmation button
         * 
         */
        bool addMultiTextBox(const char label[], const std::vector<std::string>& textLabels, std::vector<std::string>& textEntries);

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
        static void setGlobalGuiScaling(float scale);

        /** Create a new window on the stack
        */
        void pushWindow(const char label[], uint32_t width = 0, uint32_t height = 0, uint32_t x = 0, uint32_t y = 0, bool showTitleBar = true, bool allowMove = true);

        /** End a window block
        */
        void popWindow();

        /** Set the current window position in pixels
            \param[in] x horizontal window position in pixels
            \param[in] y vertical window position in pixels
        */
        void setCurrentWindowPos(uint32_t x, uint32_t y);

        /** Get the position of the current window.
            \return vec2 Value of window position
        */
        glm::vec2 getCurrentWindowPos();

        /**  Set the size of the current window in pixels
            \param[in] width Window width in pixels
            \param[in] height Window height in pixels
        */
        void setCurrentWindowSize(uint32_t width, uint32_t height);

        /** Get the size of the current window in pixels.
            \return vec2 Value of window size
        */
        glm::vec2 getCurrentWindowSize();

        /** Start a new frame. Must be called at the start of each frame
        */
        void beginFrame();

        /** Add a font
        */
        void addFont(const std::string& name, const std::string& filename);

        /** Set the active font
        */
        void setActiveFont(const std::string& font);

    private:
        Gui() = default;
        void init(float scaleFactor);
        void createVao(uint32_t vertexCount, uint32_t indexCount);
        void compileFonts();

        // Helper to create multiple inline text boxes
        bool addCheckboxes(const char label[], bool* pData, uint32_t numCheckboxes, bool sameLine);

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

        std::vector<Texture::SharedPtr> mpImages;
        ParameterBlockReflection::BindLocation mGuiImageLoc;
        float mScaleFactor = 1.0f;
        std::unordered_map<std::string, ImFont*> mFontMap;
        ImFont* mpActiveFont = nullptr;
    };
}
