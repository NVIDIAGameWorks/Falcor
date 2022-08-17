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
#include "Window.h"
#include "Assert.h"
#include "GLFW.h"
#include "Platform/OS.h"
#include "Utils/Logger.h"
#include "Utils/UI/InputTypes.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    class ApiCallbacks
    {
    public:
        static void windowSizeCallback(GLFWwindow* pGlfwWindow, int width, int height)
        {
            // We also get here in case the window was minimized, so we need to ignore it
            if (width == 0 || height == 0)
            {
                return;
            }

            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                pWindow->resize(width, height); // Window callback is handled in here
            }
        }

        static void keyboardCallback(GLFWwindow* pGlfwWindow, int key, int scanCode, int action, int modifiers)
        {
            KeyboardEvent event;
            if (prepareKeyboardEvent(key, action, modifiers, event))
            {
                Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
                if (pWindow != nullptr)
                {
                    pWindow->mpCallbacks->handleKeyboardEvent(event);
                }
            }
        }

        static void charInputCallback(GLFWwindow* pGlfwWindow, uint32_t input)
        {
            KeyboardEvent event;
            event.type = KeyboardEvent::Type::Input;
            event.codepoint = input;

            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                pWindow->mpCallbacks->handleKeyboardEvent(event);
            }
        }

        static void mouseMoveCallback(GLFWwindow* pGlfwWindow, double mouseX, double mouseY)
        {
            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                // Prepare the mouse data
                MouseEvent event;
                event.type = MouseEvent::Type::Move;
                event.pos = calcMousePos(mouseX, mouseY, pWindow->getMouseScale());
                event.screenPos = { mouseX, mouseY };
                event.wheelDelta = float2(0, 0);

                pWindow->mpCallbacks->handleMouseEvent(event);
            }
        }

        static void mouseButtonCallback(GLFWwindow* pGlfwWindow, int button, int action, int modifiers)
        {
            MouseEvent event;
            // Prepare the mouse data
            MouseEvent::Type type = (action == GLFW_PRESS) ? MouseEvent::Type::ButtonDown : MouseEvent::Type::ButtonUp;
            switch (button)
            {
            case GLFW_MOUSE_BUTTON_LEFT:
                event.type = type;
                event.button = Input::MouseButton::Left;
                break;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                event.type = type;
                event.button = Input::MouseButton::Middle;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                event.type = type;
                event.button = Input::MouseButton::Right;
                break;
            default:
                // Other keys are not supported
                break;
            }

            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                // Modifiers
                event.mods = getModifierFlags(modifiers);
                double x, y;
                glfwGetCursorPos(pGlfwWindow, &x, &y);
                event.pos = calcMousePos(x, y, pWindow->getMouseScale());

                pWindow->mpCallbacks->handleMouseEvent(event);
            }
        }

        static void mouseWheelCallback(GLFWwindow* pGlfwWindow, double scrollX, double scrollY)
        {
            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                MouseEvent event;
                event.type = MouseEvent::Type::Wheel;
                double x, y;
                glfwGetCursorPos(pGlfwWindow, &x, &y);
                event.pos = calcMousePos(x, y, pWindow->getMouseScale());
                event.wheelDelta = (float2(float(scrollX), float(scrollY)));

                pWindow->mpCallbacks->handleMouseEvent(event);
            }
        }

        static void errorCallback(int errorCode, const char* pDescription)
        {
            // GLFW errors are always recoverable. Therefore we just log the error.
            logError("GLFW error {}: {}", errorCode, pDescription);
        }

        static void droppedFileCallback(GLFWwindow* pGlfwWindow, int count, const char** paths)
        {
            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow)
            {
                for (int i = 0; i < count; i++)
                {
                    std::filesystem::path path(paths[i]);
                    pWindow->mpCallbacks->handleDroppedFile(path);
                }
            }
        }
    private:

        static inline Input::Key glfwToFalcorKey(int glfwKey)
        {
            static_assert(GLFW_KEY_ESCAPE == 256, "GLFW_KEY_ESCAPE is expected to be 256");
            static_assert((uint32_t)Input::Key::Escape >= 256, "Input::Key::Escape is expected to be at least 256");

            if (glfwKey < GLFW_KEY_ESCAPE)
            {
                // Printable keys are expected to have the same value
                return (Input::Key)glfwKey;
            }

            switch (glfwKey)
            {
            case GLFW_KEY_ESCAPE:
                return Input::Key::Escape;
            case GLFW_KEY_ENTER:
                return Input::Key::Enter;
            case GLFW_KEY_TAB:
                return Input::Key::Tab;
            case GLFW_KEY_BACKSPACE:
                return Input::Key::Backspace;
            case GLFW_KEY_INSERT:
                return Input::Key::Insert;
            case GLFW_KEY_DELETE:
                return Input::Key::Del;
            case GLFW_KEY_RIGHT:
                return Input::Key::Right;
            case GLFW_KEY_LEFT:
                return Input::Key::Left;
            case GLFW_KEY_DOWN:
                return Input::Key::Down;
            case GLFW_KEY_UP:
                return Input::Key::Up;
            case GLFW_KEY_PAGE_UP:
                return Input::Key::PageUp;
            case GLFW_KEY_PAGE_DOWN:
                return Input::Key::PageDown;
            case GLFW_KEY_HOME:
                return Input::Key::Home;
            case GLFW_KEY_END:
                return Input::Key::End;
            case GLFW_KEY_CAPS_LOCK:
                return Input::Key::CapsLock;
            case GLFW_KEY_SCROLL_LOCK:
                return Input::Key::ScrollLock;
            case GLFW_KEY_NUM_LOCK:
                return Input::Key::NumLock;
            case GLFW_KEY_PRINT_SCREEN:
                return Input::Key::PrintScreen;
            case GLFW_KEY_PAUSE:
                return Input::Key::Pause;
            case GLFW_KEY_F1:
                return Input::Key::F1;
            case GLFW_KEY_F2:
                return Input::Key::F2;
            case GLFW_KEY_F3:
                return Input::Key::F3;
            case GLFW_KEY_F4:
                return Input::Key::F4;
            case GLFW_KEY_F5:
                return Input::Key::F5;
            case GLFW_KEY_F6:
                return Input::Key::F6;
            case GLFW_KEY_F7:
                return Input::Key::F7;
            case GLFW_KEY_F8:
                return Input::Key::F8;
            case GLFW_KEY_F9:
                return Input::Key::F9;
            case GLFW_KEY_F10:
                return Input::Key::F10;
            case GLFW_KEY_F11:
                return Input::Key::F11;
            case GLFW_KEY_F12:
                return Input::Key::F12;
            case GLFW_KEY_KP_0:
                return Input::Key::Keypad0;
            case GLFW_KEY_KP_1:
                return Input::Key::Keypad1;
            case GLFW_KEY_KP_2:
                return Input::Key::Keypad2;
            case GLFW_KEY_KP_3:
                return Input::Key::Keypad3;
            case GLFW_KEY_KP_4:
                return Input::Key::Keypad4;
            case GLFW_KEY_KP_5:
                return Input::Key::Keypad5;
            case GLFW_KEY_KP_6:
                return Input::Key::Keypad6;
            case GLFW_KEY_KP_7:
                return Input::Key::Keypad7;
            case GLFW_KEY_KP_8:
                return Input::Key::Keypad8;
            case GLFW_KEY_KP_9:
                return Input::Key::Keypad9;
            case GLFW_KEY_KP_DECIMAL:
                return Input::Key::KeypadDel;
            case GLFW_KEY_KP_DIVIDE:
                return Input::Key::KeypadDivide;
            case GLFW_KEY_KP_MULTIPLY:
                return Input::Key::KeypadMultiply;
            case GLFW_KEY_KP_SUBTRACT:
                return Input::Key::KeypadSubtract;
            case GLFW_KEY_KP_ADD:
                return Input::Key::KeypadAdd;
            case GLFW_KEY_KP_ENTER:
                return Input::Key::KeypadEnter;
            case GLFW_KEY_KP_EQUAL:
                return Input::Key::KeypadEqual;
            case GLFW_KEY_LEFT_SHIFT:
                return Input::Key::LeftShift;
            case GLFW_KEY_LEFT_CONTROL:
                return Input::Key::LeftControl;
            case GLFW_KEY_LEFT_ALT:
                return Input::Key::LeftAlt;
            case GLFW_KEY_LEFT_SUPER:
                return Input::Key::LeftSuper;
            case GLFW_KEY_RIGHT_SHIFT:
                return Input::Key::RightShift;
            case GLFW_KEY_RIGHT_CONTROL:
                return Input::Key::RightControl;
            case GLFW_KEY_RIGHT_ALT:
                return Input::Key::RightAlt;
            case GLFW_KEY_RIGHT_SUPER:
                return Input::Key::RightSuper;
            case GLFW_KEY_MENU:
                return Input::Key::Menu;
            default:
                return Input::Key::Unknown;
            }
        }

        static inline Input::ModifierFlags getModifierFlags(int modifiers)
        {
            // The GLFW mods should match the Input::ModifierFlags, but this is used for now to be safe if it changes in the future.
            Input::ModifierFlags flags = Input::ModifierFlags::None;
            if (modifiers & GLFW_MOD_ALT) flags |= Input::ModifierFlags::Alt;
            if (modifiers & GLFW_MOD_CONTROL) flags |= Input::ModifierFlags::Ctrl;
            if (modifiers & GLFW_MOD_SHIFT) flags |= Input::ModifierFlags::Shift;
            return flags;
        }

        /** GLFW reports modifiers inconsistently on different platforms.
            To make modifiers consistent we check the key action and adjust
            the modifiers due to changes from the current action.
        */
        static int fixGLFWModifiers(int modifiers, int key, int action)
        {
            int bit = 0;
            if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) bit = GLFW_MOD_SHIFT;
            if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) bit = GLFW_MOD_CONTROL;
            if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) bit = GLFW_MOD_ALT;
            return (action == GLFW_RELEASE) ? modifiers & (~bit) : modifiers | bit;
        }

        static inline float2 calcMousePos(double xPos, double yPos, const float2& mouseScale)
        {
            float2 pos = float2(float(xPos), float(yPos));
            pos *= mouseScale;
            return pos;
        }

        static inline bool prepareKeyboardEvent(int key, int action, int modifiers, KeyboardEvent& event)
        {
            if (key == GLFW_KEY_UNKNOWN)
            {
                return false;
            }

            modifiers = fixGLFWModifiers(modifiers, key, action);

            switch(action)
            {
            case GLFW_RELEASE:
                event.type = KeyboardEvent::Type::KeyReleased;
                break;
            case GLFW_PRESS:
                event.type = KeyboardEvent::Type::KeyPressed;
                break;
            case GLFW_REPEAT:
                event.type = KeyboardEvent::Type::KeyRepeated;
                break;
            default:
                FALCOR_UNREACHABLE();
                break;
            }
            event.key = glfwToFalcorKey(key);
            event.mods = getModifierFlags(modifiers);
            return true;
        }
    };

    Window::Window(ICallbacks* pCallbacks, const Desc& desc)
        : mpCallbacks(pCallbacks)
        , mDesc(desc)
        , mMouseScale(1.0f / (float)desc.width, 1.0f / (float)desc.height)
    {
    }

    void Window::updateWindowSize()
    {
        // Actual window size may be clamped to slightly lower than monitor resolution
        int32_t width, height;
        glfwGetWindowSize(mpGLFWWindow, &width, &height);
        setWindowSize(width, height);
    }

    void Window::setWindowSize(uint32_t width, uint32_t height)
    {
        FALCOR_ASSERT(width > 0 && height > 0);

        mDesc.width = width;
        mDesc.height = height;
        mMouseScale.x = 1.0f / (float)mDesc.width;
        mMouseScale.y = 1.0f / (float)mDesc.height;
    }

    Window::~Window()
    {
        glfwDestroyWindow(mpGLFWWindow);
        glfwTerminate();
    }

    void Window::shutdown()
    {
        glfwSetWindowShouldClose(mpGLFWWindow, 1);
    }

    Window::SharedPtr Window::create(const Desc& desc, ICallbacks* pCallbacks)
    {
        // Set error callback
        glfwSetErrorCallback(ApiCallbacks::errorCallback);

        // Init GLFW
        if (glfwInit() == GLFW_FALSE)
        {
            throw RuntimeError("Failed to initialize GLFW.");
        }

        // Create the window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        uint32_t w = desc.width;
        uint32_t h = desc.height;

        if (desc.mode == WindowMode::Fullscreen)
        {
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
            auto mon = glfwGetPrimaryMonitor();
            auto mod = glfwGetVideoMode(mon);
            w = mod->width;
            h = mod->height;
        }
        else if (desc.mode == WindowMode::Minimized)
        {
            // Start with window being invisible
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
        }

        if (desc.resizableWindow == false)
        {
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        }

        GLFWwindow* pGLFWWindow = glfwCreateWindow(w, h, desc.title.c_str(), nullptr, nullptr);
        if (!pGLFWWindow)
        {
            throw RuntimeError("Failed to create GLFW window.");
        }

        SharedPtr pWindow = SharedPtr(new Window(pCallbacks, desc));

        // Init handles
        pWindow->mpGLFWWindow = pGLFWWindow;
#if FALCOR_WINDOWS
        pWindow->mApiHandle = glfwGetWin32Window(pGLFWWindow);
        FALCOR_ASSERT(pWindow->mApiHandle);
#elif FALCOR_LINUX
        pWindow->mApiHandle.pDisplay = glfwGetX11Display();
        pWindow->mApiHandle.window = glfwGetX11Window(pGLFWWindow);
        FALCOR_ASSERT(pWindow->mApiHandle.pDisplay != nullptr);
#endif
        setMainWindowHandle(pWindow->mApiHandle);

        pWindow->updateWindowSize();

        glfwSetWindowUserPointer(pGLFWWindow, pWindow.get());

        // Set callbacks
        glfwSetWindowSizeCallback(pGLFWWindow, ApiCallbacks::windowSizeCallback);
        glfwSetKeyCallback(pGLFWWindow, ApiCallbacks::keyboardCallback);
        glfwSetMouseButtonCallback(pGLFWWindow, ApiCallbacks::mouseButtonCallback);
        glfwSetCursorPosCallback(pGLFWWindow, ApiCallbacks::mouseMoveCallback);
        glfwSetScrollCallback(pGLFWWindow, ApiCallbacks::mouseWheelCallback);
        glfwSetCharCallback(pGLFWWindow, ApiCallbacks::charInputCallback);
        glfwSetDropCallback(pGLFWWindow, ApiCallbacks::droppedFileCallback);

        if (desc.mode == WindowMode::Minimized)
        {
            // Iconify and show window to make it available if user clicks on it
            glfwIconifyWindow(pWindow->mpGLFWWindow);
            glfwShowWindow(pWindow->mpGLFWWindow);
        }

        return pWindow;
    }

    void Window::resize(uint32_t width, uint32_t height)
    {
        glfwSetWindowSize(mpGLFWWindow, width, height);

        // In minimized mode GLFW reports incorrect window size
        if (mDesc.mode == WindowMode::Minimized)
        {
            setWindowSize(width, height);
        }
        else
        {
            updateWindowSize();
        }

        mpCallbacks->handleWindowSizeChange();
    }

    void Window::msgLoop()
    {
        // Samples often rely on a size change event as part of initialization
        // This would have happened from a WM_SIZE message when calling ShowWindow on Win32
        mpCallbacks->handleWindowSizeChange();

        if (mDesc.mode != WindowMode::Minimized)
        {
            glfwShowWindow(mpGLFWWindow);
            glfwFocusWindow(mpGLFWWindow);
        }

        while (glfwWindowShouldClose(mpGLFWWindow) == false)
        {
            pollForEvents();
            mpCallbacks->handleRenderFrame();
        }
    }

    void Window::setWindowPos(int32_t x, int32_t y)
    {
        glfwSetWindowPos(mpGLFWWindow, x, y);
    }

    void Window::setWindowTitle(const std::string& title)
    {
        glfwSetWindowTitle(mpGLFWWindow, title.c_str());
    }

    void Window::pollForEvents()
    {
        glfwPollEvents();
        handleGamepadInput();
    }

    struct Window::GamepadData
    {
        static constexpr int kInvalidID = -1;
        bool initialized = false;
        int activeID = kInvalidID;
        GamepadState previousState;
    };

    void Window::handleGamepadInput()
    {
        // Perform one-time initialization.
        if (!mpGamepadData)
        {
            mpGamepadData = std::make_unique<GamepadData>();

            // Register mappings for NV controllers.
            static char nvPadMapping[] =
                "03000000550900001472000000000000,NVIDIA Controller v01.04,a:b11,b:b10,back:b13,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b7,leftstick:b5,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b6,rightstick:b4,righttrigger:a5,rightx:a3,righty:a6,start:b3,x:b9,y:b8,platform:Windows,\n"
                "03000000550900001072000000000000,NVIDIA Shield,a:b9,b:b8,back:b11,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b5,leftstick:b3,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b4,rightstick:b2,righttrigger:a4,rightx:a2,righty:a5,start:b0,x:b7,y:b6,platform:Windows,\n"
                "030000005509000000b4000000000000,NVIDIA Virtual Gamepad,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b8,lefttrigger:+a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b9,righttrigger:-a2,rightx:a3,righty:a4,start:b7,x:b2,y:b3,platform:Windows,";
            glfwUpdateGamepadMappings(nvPadMapping);
        }

        // Check if a gamepad is connected.
        if (mpGamepadData->activeID == GamepadData::kInvalidID)
        {
            for (int id = GLFW_JOYSTICK_1; id <= GLFW_JOYSTICK_LAST; ++id)
            {
                if (glfwJoystickPresent(id) && glfwJoystickIsGamepad(id))
                {
                    std::string name(glfwGetJoystickName(id));
                    logInfo("Gamepad '{}' connected.", name);
                    mpGamepadData->activeID = id;
                    mpGamepadData->previousState = {};

                    GamepadEvent event { GamepadEvent::Type::Connected };
                    mpCallbacks->handleGamepadEvent(event);

                    break;
                }
            }
        }

        // Check if gamepad is disconnected.
        if (mpGamepadData->activeID != GamepadData::kInvalidID)
        {
            if (!(glfwJoystickPresent(mpGamepadData->activeID) && glfwJoystickIsGamepad(mpGamepadData->activeID)))
            {
                logInfo("Gamepad disconnected.");
                mpGamepadData->activeID = GamepadData::kInvalidID;

                GamepadEvent event { GamepadEvent::Type::Disconnected };
                mpCallbacks->handleGamepadEvent(event);
            }
        }

        if (mpGamepadData->activeID == GamepadData::kInvalidID) return;

        GLFWgamepadstate glfwState;
        if (glfwGetGamepadState(mpGamepadData->activeID, &glfwState) != GLFW_TRUE) return;

        GamepadState currentState;
        currentState.leftX = glfwState.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
        currentState.leftY = glfwState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
        currentState.rightX = glfwState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
        currentState.rightY = glfwState.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
        currentState.leftTrigger = glfwState.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
        currentState.rightTrigger = glfwState.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
        currentState.buttons[(size_t)GamepadButton::A] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::B] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::X] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_X] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Y] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_Y] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::LeftBumper] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::RightBumper] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Back] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Start] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Guide] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_GUIDE] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::LeftThumb] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::RightThumb] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_THUMB] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Up] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Right] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Down] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS;
        currentState.buttons[(size_t)GamepadButton::Left] = glfwState.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS;

        auto &previousState = mpGamepadData->previousState;

        // Synthesize gamepad button events.
        for (uint32_t buttonIndex = 0; buttonIndex < (uint32_t)GamepadButton::Count; ++buttonIndex)
        {
            if (currentState.buttons[buttonIndex] && !previousState.buttons[buttonIndex])
            {
                GamepadEvent event { GamepadEvent::Type::ButtonDown, GamepadButton(buttonIndex) };
                mpCallbacks->handleGamepadEvent(event);
            }
            if (!currentState.buttons[buttonIndex] && previousState.buttons[buttonIndex])
            {
                GamepadEvent event { GamepadEvent::Type::ButtonUp, GamepadButton(buttonIndex) };
                mpCallbacks->handleGamepadEvent(event);
            }
        }

        previousState = currentState;

        mpCallbacks->handleGamepadState(currentState);
    }

    FALCOR_SCRIPT_BINDING(Window)
    {
        pybind11::class_<Window, Window::SharedPtr> window(m, "Window");
        window.def("setWindowPos", &Window::setWindowPos);

        pybind11::enum_<Window::WindowMode> windowMode(m, "WindowMode");
        windowMode.value("Normal", Window::WindowMode::Normal);
        windowMode.value("Fullscreen", Window::WindowMode::Fullscreen);
        windowMode.value("Minimized", Window::WindowMode::Minimized);

        ScriptBindings::SerializableStruct<Window::Desc> windowDesc(m, "WindowDesc");
#define field(f_) field(#f_, &Window::Desc::f_)
        windowDesc.field(width);
        windowDesc.field(height);
        windowDesc.field(title);
        windowDesc.field(mode);
        windowDesc.field(resizableWindow);
#undef field
    }
}
