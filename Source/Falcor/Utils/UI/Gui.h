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
#include "Core/API/Texture.h"
#include "Core/API/FBO.h"
#include "Utils/Math/Vector.h"
#include "Utils/Color/SampledSpectrum.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct ImFont;

namespace Falcor
{
    class RenderContext;

    struct MouseEvent;
    struct KeyboardEvent;

    class GuiImpl;

    template<typename T> class SpectrumUI;

    // Helper to check if a class is a vector
    template<typename T, typename = void>
    struct is_vector : std::false_type {};

    template<typename T>
    struct is_vector<T, std::void_t<typename T::value_type>> : std::true_type {};

    /** A class wrapping the external GUI library
    */
    class FALCOR_API Gui
    {
    public:
        using UniquePtr = std::unique_ptr<Gui>;
        using UniqueConstPtr = std::unique_ptr<const Gui>;
        using GraphCallback = float(*)(void*, int32_t index);

        /** These structs used to initialize dropdowns
        */
        struct DropdownValue
        {
            uint32_t value;      ///< User defined index. Should be unique between different options.
            std::string label;  ///< Label of the dropdown option.
        };

        using DropdownList = std::vector <DropdownValue>;

        struct RadioButton
        {
            uint32_t buttonID;  ///< User defined index. Should be unique between different options in the same group.
            std::string label; ///< Label of the radio button.
            bool sameLine;     ///< Whether the button should appear on the same line as the previous widget/button.
        };

        using RadioButtonGroup = std::vector<RadioButton>;

        enum class TextFlags
        {
            Empty = 0x0,
            FitWindow = 0x1,  // Also hides the label
        };

        enum class WindowFlags
        {
            Empty = 0x0,        ///< No flags
            ShowTitleBar = 0x1,        ///< Show a title bar
            AllowMove = 0x2,        ///< Allow the window move
            SetFocus = 0x4,        ///< Take focus when the window appears
            CloseButton = 0x8,        ///< Add a close button
            NoResize = 0x10,       ///< Disable manual resizing
            AutoResize = 0x20,       ///< Auto resize the window to fit it's content every frame

            Default = ShowTitleBar | AllowMove | SetFocus | CloseButton
        };

        enum class WidgetFlags
        {
            Empty = 0x0,     ///< No flags
            SameLine = 0x1,     ///< Show a title bar
            Inactive = 0x2,     ///< Inactive widget, disallow edits
        };

        class FALCOR_API Group;

        class FALCOR_API Widgets
        {
        public:
            /** Begin a new group
                \param[in] label the name of the group
                \param[in] beginExpanded Optional whether the group is open or closed by default
            */
            Group group(const std::string& label, bool beginExpanded = false);

            /** Indent the next item
            */
            void indent(float i);

            /** Add a separator
            */
            void separator(uint32_t count = 1);

            /** Dummy object especially useful for spacing
                \param[in] label. Name for id of item
                \param[in] size. size in pixels of the item.
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            */
            void dummy(const char label[], const float2& size, bool sameLine = false);

            /** Display rectangle with specified color
                \param[in] size size in pixels of rectangle
                \param[in] color Optional. color as an rgba float4
                \param[in] filled Optional. If set to true, rectangle will be filled
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            */
            void rect(const float2& size, const float4& color = float4(1.0f, 1.0f, 1.0f, 1.0f), bool filled = false, bool sameLine = false);

            /** Adds a dropdown menu. This will update a user variable directly, so the user has to keep track of that for changes.
                If you want notifications whenever the select option changed, use Gui#addDropdownWithCallback().
                \param[in] label The name of the dropdown menu.
                \param[in] values A list of options to show in the dropdown menu.
                \param[in] var A reference to a user variable that will be updated directly when a dropdown option changes. This correlates to the 'pValue' field in Gui#SDropdownValue struct.
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
                \return true if the value changed, otherwise false
            */
            bool dropdown(const char label[], const DropdownList& values, uint32_t& var, bool sameLine = false);

            /** Button. Will return true if the button was pressed
                \param[in] label Text to display on the button
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            */
            bool button(const char label[], bool sameLine = false);

            /** Adds a group of radio buttons.
                \param[in] buttons List of buttons to show.
                \param[out] activeID If a button was clicked, activeID will be set to the ID of the clicked button.
                \return Whether activeID changed.
            */
            bool radioButtons(const RadioButtonGroup& buttons, uint32_t& activeID);

            /** Adds a direction widget
                \param[in] label The name of the widget.
                \param[in] direction A reference for the direction variable
                \return true if the value changed, otherwise false
            */
            bool direction(const char label[], float3& direction);

            /** Adds a UI widget for multiple checkboxes.
                \param[in] label The name of the widget.
                \param[in] var A reference to the bools that will be updated directly when the checkbox state changes (0 = unchecked, 1 = checked).
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
                \return true if the value changed, otherwise false
            */
            template<typename T>
            bool checkbox(const char label[], T& var, bool sameLine = false);

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

            // Text
            /** Static text
                \param[in] text The string to display
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            */
            void text(const std::string& text, bool sameLine = false);

            /** Static text wrapped to the window
                \param[in] text The string to display
            */
            void textWrapped(const std::string& text);

            /** Adds a text box.
                \param[in] label The name of the variable.
                \param[in] text A string with the initialize text. The string will be updated if a text is entered.
                \param[in] lineCount Number of lines in the text-box. If larger then 1 will create a multi-line box
                \return true if the value changed, otherwise false
            */
            bool textbox(const std::string& label, std::string& text, TextFlags flags = TextFlags::Empty);

            /** Adds a text box.
                \param[in] label The name of the variable.
                \param[in] buf A character buffer with the initialize text. The buffer will be updated if a text is entered.
                \param[in] bufSize The size of the text buffer
                \param[in] lineCount Number of lines in the text-box. If larger then 1 will create a multi-line box
                \return true if the value changed, otherwise false
            */
            bool textbox(const char label[], char buf[], size_t bufSize, uint32_t lineCount = 1, TextFlags flags = TextFlags::Empty);

            /** Adds multiple text boxes for one confirmation button
            */
            bool multiTextbox(const char label[], const std::vector<std::string>& textLabels, std::vector<std::string>& textEntries);

            /** Render a tooltip. This will display a small question mark next to the last label item rendered and will display the tooltip if the user hover over it
                \param[in] tip The tooltip's text
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            */
            void tooltip(const std::string& text, bool sameLine = true);

            // Colors
            /** Adds an RGB color UI widget.
                \param[in] label The name of the widget.
                \param[in] var A reference to a vector that will be updated directly when the widget state changes.
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
                \return true if the value changed, otherwise false
            */
            bool rgbColor(const char label[], float3& var, bool sameLine = false);

            /** Adds an RGBA color UI widget.
                \param[in] label The name of the widget.
                \param[in] var A reference to a vector that will be updated directly when the widget state changes.
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
                \return true if the value changed, otherwise false
            */
            bool rgbaColor(const char label[], float4& var, bool sameLine = false);

            // Images
            /** Display image within imgui
                \param[in] label. Name for id for item.
                \param[in] pTex. Pointer to texture resource to draw in imgui
                \param[in] size. Size in pixels of the image to draw. 0 means fit to window
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
            */
            void image(const char label[], const Texture::SharedPtr& pTex, float2 size = float2(0), bool maintainRatio = true, bool sameLine = false);
            bool imageButton(const char label[], const Texture::SharedPtr& pTex, float2 size, bool maintainRatio = true, bool sameLine = false);

            // Scalars
            /** Adds a UI element for setting scalar values.
                \param[in] label The name of the widget.
                \param[in] var A reference that will be updated directly when the widget state changes.
                \param[in] minVal Optional. The minimum allowed value for the float.
                \param[in] maxVal Optional. The maximum allowed value for the float.
                \param[in] step Optional. The step rate for the float. (Only used for non-sliders)
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget.
                \param[in] displayFormat Optional. Formatting string.
                \return true if the value changed, otherwise false
            */
            template<typename T, std::enable_if_t<!is_vector<T>::value, bool> = true>
            bool var(const char label[], T& var, T minVal = std::numeric_limits<T>::lowest(), T maxVal = std::numeric_limits<T>::max(),
                     float step = std::is_floating_point_v<T> ? 0.001f : 1.0f, bool sameLine = false, const char* displayFormat = nullptr);

            template<typename T, std::enable_if_t<!is_vector<T>::value, bool> = true>
            bool slider(const char label[], T& var, T minVal = std::numeric_limits<T>::lowest() / 2, T maxVal = std::numeric_limits<T>::max() / 2, bool sameLine = false, const char* displayFormat = nullptr);

            // Vectors
            /** Adds a UI element for setting vector values.
                \param[in] label The name of the widget.
                \param[in] var A reference that will be updated directly when the widget state changes.
                \param[in] minVal Optional. The minimum allowed value for the float.
                \param[in] maxVal Optional. The maximum allowed value for the float.
                \param[in] step Optional. The step rate for the float. (Only used for non-sliders)
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget.
                \param[in] displayFormat Optional. Formatting string.
                \return true if the value changed, otherwise false
            */
            template<typename T, std::enable_if_t<is_vector<T>::value, bool> = true>
            bool var(const char label[], T& var, typename T::value_type minVal = std::numeric_limits<typename T::value_type>::lowest(), typename T::value_type maxVal = std::numeric_limits<typename T::value_type>::max(),
                     float step = std::is_floating_point_v<typename T::value_type> ? 0.001f : 1.0f, bool sameLine = false, const char* displayFormat = nullptr);

            template<typename T, std::enable_if_t<is_vector<T>::value, bool> = true>
            bool slider(const char label[], T& var, typename T::value_type minVal = std::numeric_limits<typename T::value_type>::lowest() / 2,
                        typename T::value_type maxVal = std::numeric_limits<typename T::value_type>::max() / 2, bool sameLine = false, const char* displayFormat = nullptr);

            // Matrices
            /** Adds an matrix UI widget.
                \param[in] label The name of the widget.
                \param[in] var A reference to the matrix struct that will be updated directly when the widget state changes.
                \param[in] minVal Optional. The minimum allowed value for the variable.
                \param[in] maxVal Optional. The maximum allowed value for the variable.
                \param[in] sameLine Optional. If set to true, the widget will appear on the same line as the previous widget
                \return true if the value changed, otherwise false
            */
            template<typename MatrixType>
            bool matrix(const char label[], MatrixType& var, float minVal = -FLT_MAX, float maxVal = FLT_MAX, bool sameLine = false);

            // Graph
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
            void graph(const char label[], GraphCallback func, void* pUserData, uint32_t sampleCount, int32_t sampleOffset, float yMin = FLT_MAX, float yMax = FLT_MAX, uint32_t width = 0, uint32_t height = 100);

            /** Adds a Spectrum user interface. Since there is no SpectrumUI class as input, this call will only disply the spectrum curve and its points and final RGB color.
                Use the spectrum() function below with a SpectrumUI parameters as well if you want to have the spectrum UI as well.
                \param[in] label The name of the widget.
                \param[in] spectrum The spectrum to be visualized (and possibly edited).
            */
            template<typename T>
            bool spectrum(const char label[], SampledSpectrum<T>& spectrum);

            /** Adds a Spectrum user interface.
                \param[in] label The name of the widget.
                \param[in] spectrum The spectrum to be visualized (and possibly edited).
                \param[in] spectrumUI The spectrum UI with modifiable parameters.
            */
            template<typename T>
            bool spectrum(const char label[], SampledSpectrum<T>& spectrum, SpectrumUI<T>& spectrumUI);

            Gui* gui() const { return mpGui; }

        protected:
            Widgets() = default;
            Gui* mpGui = nullptr;
        };

        class FALCOR_API Menu
        {
        public:
            /** Create a new menu
                \param[in] pGui a pointer to the current Gui object
                \param[in] name the name of the menu
            */
            Menu(Gui* pGui, const char* name);

            ~Menu();

            /** End the menu of items.
            */
            void release();

            class FALCOR_API Dropdown
            {
            public:
                /** Create a new dropdown menu
                    \param[in] pGui a pointer to the current Gui object
                    \param[in] label the name of the menu
                */
                Dropdown(Gui* pGui, const char label[]);

                ~Dropdown();

                /** End the drop down menu list of items.
                */
                void release();

                /** Item to be displayed in dropdown menu for the main menu bar
                    \param[in] label name of item to list in the menu.
                    \param[in] var if the label is selected or not.
                    \param[in] shortcut Shortcut key. It's just for display purposes, it doesn't handle the keystroke
                    \return true if the option was selected in the dropdown
                */
                bool item(const std::string& label, bool& var, const std::string& shortcut = "");

                /** Item to be displayed in dropdown menu for the main menu bar
                    \param[in] label Name of item to list in the menu.
                    \param[in] shortcut Shortcut key. It's just for display purposes, it doesn't handle the keystroke
                    \return true if the option was selected in the dropdown
                */
                bool item(const std::string& label, const std::string& shortcut = "");

                /** Add a separator between menu items.
                */
                void separator();

                /** Adds a sub-menu within the current dropdown menu.
                    \param[in] name the name of the menu
                */
                Menu menu(const char* name);

                Gui* mpGui = nullptr;
            };

            /** Begins a collapsible menu in the menu bar of menu items
                \param[in] label name of drop down menu to be displayed.
            */
            Dropdown dropdown(const std::string& label);

            /** Add an item to the menu.
                \param[in] label name of the item to list in the menu.
            */
            bool item(const std::string& label);

        protected:
            Menu() = default;
            Gui* mpGui = nullptr;
        };

        class FALCOR_API Group : public Widgets
        {
        public:
            Group() = default;

            /** Create a collapsible group block
                \param[in] a pointer to the current pGui object
                \param[in] label Display name of the group
                \param[in] beginExpanded Whether group should be expanded initially
            */
            Group(Gui* pGui, const std::string& label, bool beginExpanded = false);

            /** Create a collapsible group block
                \param[in] w a reference to a Widgets object
                \param[in] label Display name of the group
                \param[in] beginExpanded Whether group should be expanded initially
            */
            Group(const Widgets& w, const std::string& label, bool beginExpanded = false) : Group(w.gui(), label, beginExpanded) {}

            /** Check if the current group is open or closed.
            */
            bool open() const;

            /** Bool operator to check if the current group is open or closed.
            */
            operator bool() const { return open(); }

            ~Group();

            /** End a collapsible group block
            */
            void release();
        };

        class FALCOR_API Window : public Widgets
        {
        public:
            /** Create a new window
                \param[in] pGui a pointer to the current Gui object
                \param[in] size size in pixels of the window
                \param[in] pos position in pixels of the window
                \param[in] flags Window flags to apply
            */
            Window(Gui* pGui, const char* name, uint2 size = { 0, 0 }, uint2 pos = { 0, 0 }, Gui::WindowFlags flags = Gui::WindowFlags::Default);
            Window(Gui* pGui, const char* name, bool& open, uint2 size = { 0, 0 }, uint2 pos = { 0, 0 }, Gui::WindowFlags flags = Gui::WindowFlags::Default);

            /** Create a new window
                \param[in] w a reference to a Widgets object
                \param[in] size size in pixels of the window
                \param[in] pos position in pixels of the window
                \param[in] flags Window flags to apply
            */
            Window(const Widgets& w, const char* name, uint2 size = { 0, 0 }, uint2 pos = { 0, 0 }, Gui::WindowFlags flags = Gui::WindowFlags::Default) : Window(w.gui(), name, size, pos, flags) {}
            Window(const Widgets& w, const char* name, bool& open, uint2 size = { 0, 0 }, uint2 pos = { 0, 0 }, Gui::WindowFlags flags = Gui::WindowFlags::Default) : Window(w.gui(), name, open, size, pos, flags) {}

            ~Window();

            /** End the window.
            */
            void release();

            /** Begin a column within the current window
                \param[in] numColumns requires number of columns within the window.
             */
            void columns(uint32_t numColumns);

            /** Proceed to the next column within the window.
             */
            void nextColumn();

            /** Set the current window position in pixels
                \param[in] x horizontal window position in pixels
                \param[in] y vertical window position in pixels
            */
            void windowPos(uint32_t x, uint32_t y);

            /**  Set the size of the current window in pixels
                \param[in] width Window width in pixels
                \param[in] height Window height in pixels
            */
            void windowSize(uint32_t width, uint32_t height);
        };

        class FALCOR_API MainMenu : public Menu
        {
        public:
            /** Create a new main menu bar.
                \param[in] pGui a pointer to the current Gui object
            */
            MainMenu(Gui* pGui);

            ~MainMenu();

            /** End the main menu bar.
            */
            void release();
        };

        /** Create a new GUI object. Each object is essentially a container for a GUI window
        */
        static UniquePtr create(uint32_t width, uint32_t height, float scaleFactor = 1.0f);

        ~Gui();

        static float4 pickUniqueColor(const std::string& key);

        /** Add a font
        */
        void addFont(const std::string& name, const std::filesystem::path& path);

        /** Set the active font
        */
        void setActiveFont(const std::string& font);

        ImFont* getFont(std::string f = "");

        /** Start a new frame. Must be called at the start of each frame
        */
        void beginFrame();

        /** Set global font size scaling
        */
        static void setGlobalGuiScaling(float scale);

        /** Render the GUI
        */
        void render(RenderContext* pContext, const Fbo::SharedPtr& pFbo, float elapsedTime);

        /** Handle window resize events
        */
        void onWindowResize(uint32_t width, uint32_t height);

        /** Handle mouse events
        */
        bool onMouseEvent(const MouseEvent& event);

        /** Handle keyboard events
        */
        bool onKeyboardEvent(const KeyboardEvent& event);
    private:
        Gui() = default;
        GuiImpl* mpWrapper = nullptr;
    };

    /** Helper class to create a scope for ImGui IDs using PushID/PopID.
    */
    class FALCOR_API IDScope
    {
    public:
        IDScope(const void* id);
        ~IDScope();
    };

    FALCOR_ENUM_CLASS_OPERATORS(Gui::WindowFlags);
    FALCOR_ENUM_CLASS_OPERATORS(Gui::TextFlags);
}
